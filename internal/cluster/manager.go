package cluster

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/Hamz04/nexus/internal/raft"
	"go.uber.org/zap"
)

// ModelDeployment represents a model deployed to the cluster.
type ModelDeployment struct {
	ModelID     string    `json:"model_id"`
	ModelName   string    `json:"model_name"`
	Version     string    `json:"version"`
	Nodes       []string  `json:"nodes"`        // assigned node IDs
	Replicas    int       `json:"replicas"`
	Shards      int       `json:"shards"`       // tensor parallel shards
	MaxBatch    int       `json:"max_batch"`
	TimeoutMs   int       `json:"timeout_ms"`
	Status      string    `json:"status"`       // deploying, active, draining, failed
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// ClusterState is the replicated state machine applied via Raft.
type ClusterState struct {
	mu          sync.RWMutex
	Models      map[string]*ModelDeployment `json:"models"`
	Nodes       map[string]*NodeInfo        `json:"nodes"`
	Version     uint64                      `json:"version"`
}

// NodeInfo stores metadata about a cluster member.
type NodeInfo struct {
	NodeID   string   `json:"node_id"`
	Addr     string   `json:"addr"`
	GPUs     int      `json:"gpus"`
	GPUType  string   `json:"gpu_type"`  // e.g. "A100-80GB"
	MemoryGB float64  `json:"memory_gb"`
	Labels   map[string]string `json:"labels"`
	JoinedAt time.Time `json:"joined_at"`
}

// Command types for the Raft log.
const (
	CmdDeployModel   = "deploy_model"
	CmdUndeployModel = "undeploy_model"
	CmdUpdateModel   = "update_model"
	CmdAddNode       = "add_node"
	CmdRemoveNode    = "remove_node"
)

// Command is a serializable state machine command.
type Command struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

// Manager orchestrates the cluster: Raft consensus, health checking,
// model registry, and node membership.
type Manager struct {
	raftNode    *raft.RaftNode
	healthCheck *HealthChecker
	state       *ClusterState
	logger      *zap.Logger

	// Model placement strategy
	placer Placer

	stopCh chan struct{}
}

// Placer decides which nodes to assign a model to.
type Placer interface {
	Place(model *ModelDeployment, nodes map[string]*NodeInfo, health map[string]NodeHealth) ([]string, error)
}

// ManagerConfig holds configuration for the cluster manager.
type ManagerConfig struct {
	RaftConfig   raft.Config
	HealthConfig HealthConfig
	LocalNode    NodeInfo
}

// NewManager creates a new cluster manager.
func NewManager(cfg ManagerConfig, transport raft.Transport, pinger Pinger, logger *zap.Logger) (*Manager, error) {
	// Create Raft node
	raftNode, err := raft.NewRaftNode(cfg.RaftConfig, transport, logger)
	if err != nil {
		return nil, fmt.Errorf("create raft node: %w", err)
	}

	// Create health checker
	hc := NewHealthChecker(cfg.HealthConfig, pinger, logger)

	mgr := &Manager{
		raftNode:    raftNode,
		healthCheck: hc,
		state: &ClusterState{
			Models: make(map[string]*ModelDeployment),
			Nodes:  make(map[string]*NodeInfo),
		},
		logger: logger.Named("cluster"),
		placer: &DefaultPlacer{},
		stopCh: make(chan struct{}),
	}

	// Wire up health callbacks
	hc.OnDead(func(nodeID string) {
		mgr.handleNodeFailure(nodeID)
	})
	hc.OnRecover(func(nodeID string) {
		mgr.handleNodeRecovery(nodeID)
	})

	// Register self
	local := cfg.LocalNode
	local.JoinedAt = time.Now()
	mgr.state.Nodes[local.NodeID] = &local

	return mgr, nil
}

// Start begins cluster operations.
func (m *Manager) Start() {
	m.raftNode.Start()
	m.healthCheck.Start()
	go m.applyLoop()
	m.logger.Info("cluster manager started")
}

// Stop gracefully shuts down the cluster manager.
func (m *Manager) Stop() {
	close(m.stopCh)
	m.healthCheck.Stop()
	m.raftNode.Stop()
	m.logger.Info("cluster manager stopped")
}

// DeployModel submits a model deployment to the cluster.
func (m *Manager) DeployModel(model *ModelDeployment) error {
	if !m.raftNode.IsLeader() {
		return fmt.Errorf("not leader: forward to leader")
	}

	// Determine placement
	health := m.healthCheck.GetAllHealth()
	nodes, err := m.placer.Place(model, m.state.Nodes, health)
	if err != nil {
		return fmt.Errorf("placement failed: %w", err)
	}

	model.Nodes = nodes
	model.Status = "deploying"
	model.CreatedAt = time.Now()
	model.UpdatedAt = time.Now()

	payload, err := json.Marshal(model)
	if err != nil {
		return fmt.Errorf("marshal model: %w", err)
	}

	cmd := Command{
		Type:    CmdDeployModel,
		Payload: payload,
	}
	cmdBytes, err := json.Marshal(cmd)
	if err != nil {
		return fmt.Errorf("marshal command: %w", err)
	}

	m.logger.Info("deploying model",
		zap.String("model", model.ModelID),
		zap.String("name", model.ModelName),
		zap.Strings("nodes", nodes),
		zap.Int("shards", model.Shards),
	)

	return m.raftNode.Propose(cmdBytes)
}

// UndeployModel removes a model from the cluster.
func (m *Manager) UndeployModel(modelID string) error {
	if !m.raftNode.IsLeader() {
		return fmt.Errorf("not leader")
	}

	payload, _ := json.Marshal(map[string]string{"model_id": modelID})
	cmd := Command{Type: CmdUndeployModel, Payload: payload}
	cmdBytes, _ := json.Marshal(cmd)

	m.logger.Info("undeploying model", zap.String("model", modelID))
	return m.raftNode.Propose(cmdBytes)
}

// AddNode registers a new node in the cluster.
func (m *Manager) AddNode(node NodeInfo) error {
	if !m.raftNode.IsLeader() {
		return fmt.Errorf("not leader")
	}

	node.JoinedAt = time.Now()
	payload, _ := json.Marshal(node)
	cmd := Command{Type: CmdAddNode, Payload: payload}
	cmdBytes, _ := json.Marshal(cmd)

	m.logger.Info("adding node",
		zap.String("node", node.NodeID),
		zap.Int("gpus", node.GPUs),
		zap.String("gpu_type", node.GPUType),
	)

	// Also register in health checker
	m.healthCheck.RegisterNode(node.NodeID, node.Addr)

	return m.raftNode.Propose(cmdBytes)
}

// RemoveNode deregisters a node from the cluster.
func (m *Manager) RemoveNode(nodeID string) error {
	if !m.raftNode.IsLeader() {
		return fmt.Errorf("not leader")
	}

	payload, _ := json.Marshal(map[string]string{"node_id": nodeID})
	cmd := Command{Type: CmdRemoveNode, Payload: payload}
	cmdBytes, _ := json.Marshal(cmd)

	m.logger.Info("removing node", zap.String("node", nodeID))
	m.healthCheck.DeregisterNode(nodeID)

	return m.raftNode.Propose(cmdBytes)
}

// GetModel returns deployment info for a model.
func (m *Manager) GetModel(modelID string) (*ModelDeployment, bool) {
	m.state.mu.RLock()
	defer m.state.mu.RUnlock()
	model, ok := m.state.Models[modelID]
	if !ok {
		return nil, false
	}
	copy := *model
	return &copy, true
}

// ListModels returns all deployed models.
func (m *Manager) ListModels() []*ModelDeployment {
	m.state.mu.RLock()
	defer m.state.mu.RUnlock()

	models := make([]*ModelDeployment, 0, len(m.state.Models))
	for _, model := range m.state.Models {
		copy := *model
		models = append(models, &copy)
	}
	return models
}

// ListNodes returns all cluster nodes.
func (m *Manager) ListNodes() []*NodeInfo {
	m.state.mu.RLock()
	defer m.state.mu.RUnlock()

	nodes := make([]*NodeInfo, 0, len(m.state.Nodes))
	for _, node := range m.state.Nodes {
		copy := *node
		nodes = append(nodes, &copy)
	}
	return nodes
}

// applyLoop reads committed Raft entries and applies them to the state machine.
func (m *Manager) applyLoop() {
	for {
		select {
		case entry := <-m.raftNode.ApplyCh():
			m.applyEntry(entry)
		case <-m.stopCh:
			return
		}
	}
}

func (m *Manager) applyEntry(entry raft.LogEntry) {
	var cmd Command
	if err := json.Unmarshal(entry.Command, &cmd); err != nil {
		m.logger.Error("failed to unmarshal command", zap.Error(err))
		return
	}

	m.state.mu.Lock()
	defer m.state.mu.Unlock()

	switch cmd.Type {
	case CmdDeployModel:
		var model ModelDeployment
		if err := json.Unmarshal(cmd.Payload, &model); err != nil {
			m.logger.Error("unmarshal deploy", zap.Error(err))
			return
		}
		model.Status = "active"
		model.UpdatedAt = time.Now()
		m.state.Models[model.ModelID] = &model
		m.logger.Info("model deployed",
			zap.String("model", model.ModelID),
			zap.Strings("nodes", model.Nodes),
		)

	case CmdUndeployModel:
		var payload map[string]string
		json.Unmarshal(cmd.Payload, &payload)
		modelID := payload["model_id"]
		delete(m.state.Models, modelID)
		m.logger.Info("model undeployed", zap.String("model", modelID))

	case CmdAddNode:
		var node NodeInfo
		if err := json.Unmarshal(cmd.Payload, &node); err != nil {
			m.logger.Error("unmarshal add_node", zap.Error(err))
			return
		}
		m.state.Nodes[node.NodeID] = &node
		m.logger.Info("node added", zap.String("node", node.NodeID))

	case CmdRemoveNode:
		var payload map[string]string
		json.Unmarshal(cmd.Payload, &payload)
		nodeID := payload["node_id"]
		delete(m.state.Nodes, nodeID)
		// Reassign models that were on this node
		for _, model := range m.state.Models {
			for i, n := range model.Nodes {
				if n == nodeID {
					model.Nodes = append(model.Nodes[:i], model.Nodes[i+1:]...)
					model.Status = "degraded"
					break
				}
			}
		}
		m.logger.Info("node removed", zap.String("node", nodeID))

	case CmdUpdateModel:
		var model ModelDeployment
		if err := json.Unmarshal(cmd.Payload, &model); err != nil {
			return
		}
		if existing, ok := m.state.Models[model.ModelID]; ok {
			existing.Nodes = model.Nodes
			existing.Status = model.Status
			existing.UpdatedAt = time.Now()
		}
	}

	m.state.Version++
}

// handleNodeFailure is called when the health checker declares a node dead.
func (m *Manager) handleNodeFailure(nodeID string) {
	m.logger.Error("handling node failure", zap.String("node", nodeID))

	// If we're leader, initiate model reassignment
	if !m.raftNode.IsLeader() {
		return
	}

	// Find all models on this node and trigger rebalance
	m.state.mu.RLock()
	affected := make([]string, 0)
	for id, model := range m.state.Models {
		for _, n := range model.Nodes {
			if n == nodeID {
				affected = append(affected, id)
				break
			}
		}
	}
	m.state.mu.RUnlock()

	for _, modelID := range affected {
		m.logger.Warn("model affected by node failure",
			zap.String("model", modelID),
			zap.String("failed_node", nodeID),
		)
		// In production: trigger rebalance via Raft proposal
	}
}

// handleNodeRecovery is called when a previously dead node comes back.
func (m *Manager) handleNodeRecovery(nodeID string) {
	m.logger.Info("node recovered", zap.String("node", nodeID))
}

// DefaultPlacer implements a simple GPU-aware placement strategy.
type DefaultPlacer struct{}

// Place selects nodes for a model deployment based on available GPU capacity.
func (p *DefaultPlacer) Place(model *ModelDeployment, nodes map[string]*NodeInfo, health map[string]NodeHealth) ([]string, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no nodes available")
	}

	needed := model.Replicas
	if needed == 0 {
		needed = 1
	}

	// Score nodes: prefer healthy nodes with more free GPU memory and lower queue depth
	type scored struct {
		nodeID string
		score  float64
	}

	var candidates []scored
	for id, node := range nodes {
		h, ok := health[id]
		if !ok || h.Status == NodeDead {
			continue
		}

		score := float64(node.GPUs) * 100.0
		if h.GPUMemTotal > 0 {
			score += float64(h.GPUMemFree) / float64(h.GPUMemTotal) * 50.0
		}
		if h.Status == NodeHealthy {
			score += 25.0
		}
		score -= float64(h.QueueDepth) * 2.0

		candidates = append(candidates, scored{nodeID: id, score: score})
	}

	if len(candidates) < needed {
		return nil, fmt.Errorf("not enough healthy nodes: need %d, have %d", needed, len(candidates))
	}

	// Sort by score descending (simple insertion sort for small N)
	for i := 1; i < len(candidates); i++ {
		for j := i; j > 0 && candidates[j].score > candidates[j-1].score; j-- {
			candidates[j], candidates[j-1] = candidates[j-1], candidates[j]
		}
	}

	result := make([]string, needed)
	for i := 0; i < needed; i++ {
		result[i] = candidates[i].nodeID
	}

	return result, nil
}
