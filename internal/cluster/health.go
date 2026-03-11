package cluster

import (
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// NodeStatus represents the health state of a cluster node.
type NodeStatus int

const (
	NodeHealthy NodeStatus = iota
	NodeSuspect
	NodeDead
)

func (s NodeStatus) String() string {
	switch s {
	case NodeHealthy:
		return "healthy"
	case NodeSuspect:
		return "suspect"
	case NodeDead:
		return "dead"
	default:
		return "unknown"
	}
}

// NodeHealth holds the current health info for a single node.
type NodeHealth struct {
	NodeID      string
	Addr        string
	Status      NodeStatus
	LastPing    time.Time
	LastPong    time.Time
	RTT         time.Duration
	FailCount   int
	GPUMemFree  uint64 // bytes
	GPUMemTotal uint64
	GPUUtil     float64 // 0-100%
	ModelsLoaded []string
	QueueDepth  int
}

// HealthConfig holds tunable parameters for the failure detector.
type HealthConfig struct {
	PingInterval    time.Duration
	PingTimeout     time.Duration
	SuspectAfter    int // consecutive failures before suspect
	DeadAfter       int // consecutive failures before dead
	RecoveryAfter   int // consecutive successes to recover
}

func DefaultHealthConfig() HealthConfig {
	return HealthConfig{
		PingInterval:  1 * time.Second,
		PingTimeout:   500 * time.Millisecond,
		SuspectAfter:  3,
		DeadAfter:     8,
		RecoveryAfter: 3,
	}
}

// HealthChecker implements a phi-accrual style failure detector.
type HealthChecker struct {
	mu       sync.RWMutex
	nodes    map[string]*NodeHealth
	config   HealthConfig
	logger   *zap.Logger
	pinger   Pinger

	// Callbacks
	onSuspect func(nodeID string)
	onDead    func(nodeID string)
	onRecover func(nodeID string)

	// Metrics
	nodeRTT       *prometheus.HistogramVec
	nodeStatus    *prometheus.GaugeVec
	pingFailures  *prometheus.CounterVec
	aliveNodes    prometheus.Gauge

	cancel context.CancelFunc
}

// Pinger defines how to check if a remote node is alive.
type Pinger interface {
	Ping(ctx context.Context, addr string) (PingResult, error)
}

// PingResult holds the response from a health ping.
type PingResult struct {
	RTT          time.Duration
	GPUMemFree   uint64
	GPUMemTotal  uint64
	GPUUtil      float64
	ModelsLoaded []string
	QueueDepth   int
}

// NewHealthChecker creates a new failure detector.
func NewHealthChecker(cfg HealthConfig, pinger Pinger, logger *zap.Logger) *HealthChecker {
	hc := &HealthChecker{
		nodes:  make(map[string]*NodeHealth),
		config: cfg,
		logger: logger.Named("health"),
		pinger: pinger,

		nodeRTT: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "nexus",
				Subsystem: "health",
				Name:      "ping_rtt_seconds",
				Help:      "Health check round-trip time.",
				Buckets:   prometheus.ExponentialBuckets(0.0001, 2, 12),
			},
			[]string{"node"},
		),
		nodeStatus: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "nexus",
				Subsystem: "health",
				Name:      "node_status",
				Help:      "Node status (0=healthy, 1=suspect, 2=dead).",
			},
			[]string{"node"},
		),
		pingFailures: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "nexus",
				Subsystem: "health",
				Name:      "ping_failures_total",
				Help:      "Health check ping failures per node.",
			},
			[]string{"node"},
		),
		aliveNodes: promauto.NewGauge(prometheus.GaugeOpts{
			Namespace: "nexus",
			Subsystem: "health",
			Name:      "alive_nodes",
			Help:      "Number of healthy nodes in the cluster.",
		}),
	}

	return hc
}

// RegisterNode adds a node to be monitored.
func (hc *HealthChecker) RegisterNode(nodeID, addr string) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	hc.nodes[nodeID] = &NodeHealth{
		NodeID: nodeID,
		Addr:   addr,
		Status: NodeHealthy,
	}
	hc.logger.Info("registered node", zap.String("node", nodeID), zap.String("addr", addr))
}

// DeregisterNode removes a node from monitoring.
func (hc *HealthChecker) DeregisterNode(nodeID string) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	delete(hc.nodes, nodeID)
	hc.logger.Info("deregistered node", zap.String("node", nodeID))
}

// OnSuspect sets the callback when a node becomes suspect.
func (hc *HealthChecker) OnSuspect(fn func(string)) { hc.onSuspect = fn }

// OnDead sets the callback when a node is declared dead.
func (hc *HealthChecker) OnDead(fn func(string)) { hc.onDead = fn }

// OnRecover sets the callback when a dead/suspect node recovers.
func (hc *HealthChecker) OnRecover(fn func(string)) { hc.onRecover = fn }

// Start begins the periodic health check loop.
func (hc *HealthChecker) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	hc.cancel = cancel
	go hc.run(ctx)
	hc.logger.Info("health checker started", zap.Duration("interval", hc.config.PingInterval))
}

// Stop halts the health check loop.
func (hc *HealthChecker) Stop() {
	if hc.cancel != nil {
		hc.cancel()
	}
	hc.logger.Info("health checker stopped")
}

// GetHealthy returns a list of all healthy node IDs.
func (hc *HealthChecker) GetHealthy() []string {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	var result []string
	for id, n := range hc.nodes {
		if n.Status == NodeHealthy {
			result = append(result, id)
		}
	}
	return result
}

// GetNodeHealth returns health info for a specific node.
func (hc *HealthChecker) GetNodeHealth(nodeID string) (*NodeHealth, bool) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	n, ok := hc.nodes[nodeID]
	if !ok {
		return nil, false
	}
	// Return copy
	copy := *n
	return &copy, true
}

// GetAllHealth returns health info for all nodes.
func (hc *HealthChecker) GetAllHealth() map[string]NodeHealth {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	result := make(map[string]NodeHealth, len(hc.nodes))
	for id, n := range hc.nodes {
		result[id] = *n
	}
	return result
}

func (hc *HealthChecker) run(ctx context.Context) {
	ticker := time.NewTicker(hc.config.PingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			hc.pingAll(ctx)
		}
	}
}

func (hc *HealthChecker) pingAll(ctx context.Context) {
	hc.mu.RLock()
	nodes := make([]*NodeHealth, 0, len(hc.nodes))
	for _, n := range hc.nodes {
		nodes = append(nodes, n)
	}
	hc.mu.RUnlock()

	var wg sync.WaitGroup
	for _, node := range nodes {
		wg.Add(1)
		go func(n *NodeHealth) {
			defer wg.Done()
			hc.pingNode(ctx, n)
		}(node)
	}
	wg.Wait()

	// Update alive count
	healthy := hc.GetHealthy()
	hc.aliveNodes.Set(float64(len(healthy)))
}

func (hc *HealthChecker) pingNode(ctx context.Context, node *NodeHealth) {
	pingCtx, cancel := context.WithTimeout(ctx, hc.config.PingTimeout)
	defer cancel()

	hc.mu.Lock()
	node.LastPing = time.Now()
	hc.mu.Unlock()

	result, err := hc.pinger.Ping(pingCtx, node.Addr)

	hc.mu.Lock()
	defer hc.mu.Unlock()

	if err != nil {
		// Ping failed
		node.FailCount++
		hc.pingFailures.WithLabelValues(node.NodeID).Inc()

		oldStatus := node.Status

		if node.FailCount >= hc.config.DeadAfter {
			node.Status = NodeDead
		} else if node.FailCount >= hc.config.SuspectAfter {
			node.Status = NodeSuspect
		}

		hc.nodeStatus.WithLabelValues(node.NodeID).Set(float64(node.Status))

		// Fire callbacks on state transitions
		if oldStatus != node.Status {
			switch node.Status {
			case NodeSuspect:
				hc.logger.Warn("node suspect",
					zap.String("node", node.NodeID),
					zap.Int("failures", node.FailCount),
				)
				if hc.onSuspect != nil {
					go hc.onSuspect(node.NodeID)
				}
			case NodeDead:
				hc.logger.Error("node declared dead",
					zap.String("node", node.NodeID),
					zap.Int("failures", node.FailCount),
				)
				if hc.onDead != nil {
					go hc.onDead(node.NodeID)
				}
			}
		}
		return
	}

	// Ping succeeded
	node.LastPong = time.Now()
	node.RTT = result.RTT
	node.GPUMemFree = result.GPUMemFree
	node.GPUMemTotal = result.GPUMemTotal
	node.GPUUtil = result.GPUUtil
	node.ModelsLoaded = result.ModelsLoaded
	node.QueueDepth = result.QueueDepth

	hc.nodeRTT.WithLabelValues(node.NodeID).Observe(result.RTT.Seconds())

	oldStatus := node.Status

	if node.Status != NodeHealthy {
		node.FailCount = 0
		node.Status = NodeHealthy
		hc.nodeStatus.WithLabelValues(node.NodeID).Set(0)

		if oldStatus != NodeHealthy {
			hc.logger.Info("node recovered",
				zap.String("node", node.NodeID),
				zap.Duration("rtt", result.RTT),
			)
			if hc.onRecover != nil {
				go hc.onRecover(node.NodeID)
			}
		}
	} else {
		node.FailCount = 0
	}
}
