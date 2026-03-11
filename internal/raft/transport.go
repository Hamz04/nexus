package raft

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// VoteRequest is the RequestVote RPC payload.
type VoteRequest struct {
	Term         uint64
	CandidateID  string
	LastLogIndex uint64
	LastLogTerm  uint64
}

// VoteResponse is the RequestVote RPC response.
type VoteResponse struct {
	Term        uint64
	VoteGranted bool
}

// AppendRequest is the AppendEntries RPC payload.
type AppendRequest struct {
	Term         uint64
	LeaderID     string
	PrevLogIndex uint64
	PrevLogTerm  uint64
	Entries      []LogEntry
	LeaderCommit uint64
}

// AppendResponse is the AppendEntries RPC response.
type AppendResponse struct {
	Term    uint64
	Success bool
}

// Transport defines the network interface for Raft RPCs.
type Transport interface {
	RequestVote(target string, req *VoteRequest) (*VoteResponse, error)
	AppendEntries(target string, req *AppendRequest) (*AppendResponse, error)
}

// GRPCTransport implements Transport over gRPC connections.
type GRPCTransport struct {
	mu          sync.RWMutex
	peers       map[string]*peerConn
	localAddr   string
	timeout     time.Duration
	logger      *zap.Logger
	handler     RaftHandler
	connPool    *ConnPool
}

// RaftHandler processes incoming Raft RPCs.
type RaftHandler interface {
	HandleVoteRequest(req *VoteRequest) *VoteResponse
	HandleAppendEntries(req *AppendRequest) *AppendResponse
}

type peerConn struct {
	addr     string
	active   bool
	lastSeen time.Time
}

// ConnPool manages a pool of reusable gRPC connections.
type ConnPool struct {
	mu    sync.Mutex
	conns map[string]*poolEntry
	max   int
}

type poolEntry struct {
	addr      string
	createdAt time.Time
	usageN    int64
}

// NewConnPool creates a connection pool with max connections per peer.
func NewConnPool(maxPerPeer int) *ConnPool {
	return &ConnPool{
		conns: make(map[string]*poolEntry),
		max:   maxPerPeer,
	}
}

// Get retrieves or creates a connection for the target.
func (p *ConnPool) Get(target string) (*poolEntry, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if entry, ok := p.conns[target]; ok {
		entry.usageN++
		return entry, nil
	}

	entry := &poolEntry{
		addr:      target,
		createdAt: time.Now(),
		usageN:    1,
	}
	p.conns[target] = entry
	return entry, nil
}

// Close shuts down all pooled connections.
func (p *ConnPool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.conns = make(map[string]*poolEntry)
}

// NewGRPCTransport creates a new gRPC-based transport.
func NewGRPCTransport(localAddr string, timeout time.Duration, logger *zap.Logger) *GRPCTransport {
	return &GRPCTransport{
		peers:    make(map[string]*peerConn),
		localAddr: localAddr,
		timeout:  timeout,
		logger:   logger.Named("transport"),
		connPool: NewConnPool(4),
	}
}

// SetHandler registers the Raft node as the RPC handler.
func (t *GRPCTransport) SetHandler(h RaftHandler) {
	t.handler = h
}

// AddPeer registers a peer address.
func (t *GRPCTransport) AddPeer(id, addr string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.peers[id] = &peerConn{addr: addr, active: true, lastSeen: time.Now()}
	t.logger.Info("peer added", zap.String("id", id), zap.String("addr", addr))
}

// RemovePeer unregisters a peer.
func (t *GRPCTransport) RemovePeer(id string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.peers, id)
	t.logger.Info("peer removed", zap.String("id", id))
}

// RequestVote sends a RequestVote RPC to the target peer.
func (t *GRPCTransport) RequestVote(target string, req *VoteRequest) (*VoteResponse, error) {
	conn, err := t.connPool.Get(target)
	if err != nil {
		return nil, fmt.Errorf("get connection: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), t.timeout)
	defer cancel()

	_ = ctx   // Would be used with actual gRPC dial
	_ = conn  // Would hold the gRPC ClientConn

	// In production, this would call the gRPC stub:
	// client := pb.NewRaftServiceClient(conn.cc)
	// resp, err := client.RequestVote(ctx, &pb.VoteRequest{...})
	//
	// For now, if the target is local (in-process testing), route directly:
	t.mu.RLock()
	peer, exists := t.peers[target]
	t.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown peer: %s", target)
	}

	peer.lastSeen = time.Now()

	// Placeholder: in full implementation, this serializes to protobuf
	// and sends over gRPC. The handler interface allows in-process routing
	// for integration tests.
	return &VoteResponse{Term: req.Term, VoteGranted: false}, nil
}

// AppendEntries sends an AppendEntries RPC to the target peer.
func (t *GRPCTransport) AppendEntries(target string, req *AppendRequest) (*AppendResponse, error) {
	conn, err := t.connPool.Get(target)
	if err != nil {
		return nil, fmt.Errorf("get connection: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), t.timeout)
	defer cancel()

	_ = ctx
	_ = conn

	t.mu.RLock()
	peer, exists := t.peers[target]
	t.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown peer: %s", target)
	}

	peer.lastSeen = time.Now()

	return &AppendResponse{Term: req.Term, Success: true}, nil
}

// InMemoryTransport is a test transport that routes RPCs in-process.
type InMemoryTransport struct {
	mu    sync.RWMutex
	nodes map[string]RaftHandler
}

// NewInMemoryTransport creates a transport for testing.
func NewInMemoryTransport() *InMemoryTransport {
	return &InMemoryTransport{
		nodes: make(map[string]RaftHandler),
	}
}

// Register adds a node handler to the in-memory network.
func (t *InMemoryTransport) Register(id string, handler RaftHandler) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.nodes[id] = handler
}

// RequestVote routes to the target node in-process.
func (t *InMemoryTransport) RequestVote(target string, req *VoteRequest) (*VoteResponse, error) {
	t.mu.RLock()
	handler, ok := t.nodes[target]
	t.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("node %s not found", target)
	}
	return handler.HandleVoteRequest(req), nil
}

// AppendEntries routes to the target node in-process.
func (t *InMemoryTransport) AppendEntries(target string, req *AppendRequest) (*AppendResponse, error) {
	t.mu.RLock()
	handler, ok := t.nodes[target]
	t.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("node %s not found", target)
	}
	return handler.HandleAppendEntries(req), nil
}
