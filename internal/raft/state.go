package raft

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"go.uber.org/zap"
)

// NodeState represents the current role of a Raft node.
type NodeState int

const (
	Follower NodeState = iota
	Candidate
	Leader
)

func (s NodeState) String() string {
	switch s {
	case Follower:
		return "Follower"
	case Candidate:
		return "Candidate"
	case Leader:
		return "Leader"
	default:
		return fmt.Sprintf("Unknown(%d)", int(s))
	}
}

// Config holds tunable Raft parameters.
type Config struct {
	NodeID            string
	Peers             []string
	ElectionMinMs     int
	ElectionMaxMs     int
	HeartbeatInterval time.Duration
	LogDir            string
}

func DefaultConfig(nodeID string, peers []string) Config {
	return Config{
		NodeID:            nodeID,
		Peers:             peers,
		ElectionMinMs:     150,
		ElectionMaxMs:     300,
		HeartbeatInterval: 50 * time.Millisecond,
		LogDir:            fmt.Sprintf("data/raft/%s", nodeID),
	}
}

// LogEntry is a single replicated log entry.
type LogEntry struct {
	Term    uint64
	Index   uint64
	Command []byte
}

// RaftNode implements the core Raft consensus algorithm.
type RaftNode struct {
	mu sync.RWMutex

	// Persistent state
	currentTerm uint64
	votedFor    string
	log         []LogEntry

	// Volatile state
	state       NodeState
	commitIndex uint64
	lastApplied uint64

	// Leader-only volatile state
	nextIndex  map[string]uint64
	matchIndex map[string]uint64

	// Node identity
	config Config

	// Channels
	heartbeatCh   chan struct{}
	voteGrantedCh chan struct{}
	stepDownCh    chan struct{}
	applyCh       chan LogEntry
	stopCh        chan struct{}

	// Transport
	transport Transport

	// WAL for durability
	wal *WAL

	// Logger
	logger *zap.Logger

	// Metrics
	metrics *RaftMetrics
}

// NewRaftNode creates a new Raft node with the given config.
func NewRaftNode(cfg Config, transport Transport, logger *zap.Logger) (*RaftNode, error) {
	wal, err := NewWAL(cfg.LogDir)
	if err != nil {
		return nil, fmt.Errorf("failed to create WAL: %w", err)
	}

	node := &RaftNode{
		config:        cfg,
		state:         Follower,
		nextIndex:     make(map[string]uint64),
		matchIndex:    make(map[string]uint64),
		heartbeatCh:   make(chan struct{}, 1),
		voteGrantedCh: make(chan struct{}, 1),
		stepDownCh:    make(chan struct{}, 1),
		applyCh:       make(chan LogEntry, 256),
		stopCh:        make(chan struct{}),
		transport:     transport,
		wal:           wal,
		logger:        logger.Named("raft").With(zap.String("node", cfg.NodeID)),
		metrics:       NewRaftMetrics(cfg.NodeID),
	}

	// Restore persisted state from WAL
	if err := node.restoreFromWAL(); err != nil {
		return nil, fmt.Errorf("failed to restore WAL: %w", err)
	}

	return node, nil
}

// Start begins the Raft event loop.
func (rn *RaftNode) Start() {
	rn.logger.Info("starting raft node",
		zap.String("state", rn.state.String()),
		zap.Uint64("term", rn.currentTerm),
		zap.Int("peers", len(rn.config.Peers)),
	)
	go rn.run()
}

// Stop gracefully shuts down the Raft node.
func (rn *RaftNode) Stop() {
	close(rn.stopCh)
	rn.wal.Close()
	rn.logger.Info("raft node stopped")
}

// ApplyCh returns the channel where committed entries are delivered.
func (rn *RaftNode) ApplyCh() <-chan LogEntry {
	return rn.applyCh
}

// Propose submits a new command to the Raft cluster.
// Returns error if this node is not the leader.
func (rn *RaftNode) Propose(command []byte) error {
	rn.mu.Lock()
	defer rn.mu.Unlock()

	if rn.state != Leader {
		return fmt.Errorf("not leader (current state: %s)", rn.state)
	}

	entry := LogEntry{
		Term:    rn.currentTerm,
		Index:   rn.lastLogIndex() + 1,
		Command: command,
	}

	// Persist to WAL before appending
	if err := rn.wal.Append(entry); err != nil {
		rn.logger.Error("failed to write WAL", zap.Error(err))
		return fmt.Errorf("WAL write failed: %w", err)
	}

	rn.log = append(rn.log, entry)
	rn.metrics.proposalsTotal.Inc()

	rn.logger.Debug("proposed entry",
		zap.Uint64("index", entry.Index),
		zap.Uint64("term", entry.Term),
		zap.Int("size", len(command)),
	)

	// Trigger immediate replication
	go rn.replicateToAll()

	return nil
}

// run is the main event loop for the Raft node.
func (rn *RaftNode) run() {
	for {
		select {
		case <-rn.stopCh:
			return
		default:
		}

		rn.mu.RLock()
		state := rn.state
		rn.mu.RUnlock()

		switch state {
		case Follower:
			rn.runFollower()
		case Candidate:
			rn.runCandidate()
		case Leader:
			rn.runLeader()
		}
	}
}

func (rn *RaftNode) runFollower() {
	timeout := rn.electionTimeout()
	rn.logger.Debug("follower waiting", zap.Duration("timeout", timeout))

	select {
	case <-time.After(timeout):
		// Election timeout -- become candidate
		rn.mu.Lock()
		rn.logger.Info("election timeout, becoming candidate",
			zap.Uint64("term", rn.currentTerm+1),
		)
		rn.state = Candidate
		rn.metrics.stateTransitions.WithLabelValues("candidate").Inc()
		rn.mu.Unlock()

	case <-rn.heartbeatCh:
		// Reset election timer on heartbeat
		return

	case <-rn.stopCh:
		return
	}
}

func (rn *RaftNode) runCandidate() {
	rn.mu.Lock()
	rn.currentTerm++
	rn.votedFor = rn.config.NodeID
	currentTerm := rn.currentTerm
	lastLogIndex := rn.lastLogIndex()
	lastLogTerm := rn.lastLogTerm()
	rn.mu.Unlock()

	// Persist vote
	rn.wal.PersistState(currentTerm, rn.config.NodeID)

	votes := 1 // Vote for self
	votesNeeded := (len(rn.config.Peers)+1)/2 + 1

	rn.logger.Info("starting election",
		zap.Uint64("term", currentTerm),
		zap.Int("votes_needed", votesNeeded),
	)

	// Request votes from all peers in parallel
	voteCh := make(chan bool, len(rn.config.Peers))
	for _, peer := range rn.config.Peers {
		go func(p string) {
			resp, err := rn.transport.RequestVote(p, &VoteRequest{
				Term:         currentTerm,
				CandidateID:  rn.config.NodeID,
				LastLogIndex: lastLogIndex,
				LastLogTerm:  lastLogTerm,
			})
			if err != nil {
				rn.logger.Warn("vote request failed",
					zap.String("peer", p),
					zap.Error(err),
				)
				voteCh <- false
				return
			}

			// If peer has higher term, step down
			if resp.Term > currentTerm {
				rn.stepDown(resp.Term)
				voteCh <- false
				return
			}

			voteCh <- resp.VoteGranted
		}(peer)
	}

	// Collect votes with election timeout
	timeout := rn.electionTimeout()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for i := 0; i < len(rn.config.Peers); i++ {
		select {
		case granted := <-voteCh:
			if granted {
				votes++
				rn.logger.Debug("vote received", zap.Int("total", votes))
			}
			if votes >= votesNeeded {
				rn.becomeLeader()
				return
			}

		case <-timer.C:
			rn.logger.Info("election timed out", zap.Int("votes", votes))
			return

		case <-rn.stepDownCh:
			return

		case <-rn.stopCh:
			return
		}
	}
}

func (rn *RaftNode) runLeader() {
	// Initialize nextIndex and matchIndex for all peers
	rn.mu.Lock()
	for _, peer := range rn.config.Peers {
		rn.nextIndex[peer] = rn.lastLogIndex() + 1
		rn.matchIndex[peer] = 0
	}
	rn.mu.Unlock()

	// Send initial heartbeat immediately
	rn.replicateToAll()

	heartbeatTicker := time.NewTicker(rn.config.HeartbeatInterval)
	defer heartbeatTicker.Stop()

	for {
		select {
		case <-heartbeatTicker.C:
			rn.replicateToAll()
			rn.advanceCommitIndex()

		case <-rn.stepDownCh:
			return

		case <-rn.stopCh:
			return
		}
	}
}

// replicateToAll sends AppendEntries RPCs to all peers.
func (rn *RaftNode) replicateToAll() {
	rn.mu.RLock()
	if rn.state != Leader {
		rn.mu.RUnlock()
		return
	}
	currentTerm := rn.currentTerm
	commitIndex := rn.commitIndex
	rn.mu.RUnlock()

	for _, peer := range rn.config.Peers {
		go func(p string) {
			rn.mu.RLock()
			nextIdx := rn.nextIndex[p]
			prevLogIndex := nextIdx - 1
			var prevLogTerm uint64
			if prevLogIndex > 0 && prevLogIndex <= uint64(len(rn.log)) {
				prevLogTerm = rn.log[prevLogIndex-1].Term
			}

			// Collect entries to send
			var entries []LogEntry
			if nextIdx <= uint64(len(rn.log)) {
				entries = rn.log[nextIdx-1:]
			}
			rn.mu.RUnlock()

			resp, err := rn.transport.AppendEntries(p, &AppendRequest{
				Term:         currentTerm,
				LeaderID:     rn.config.NodeID,
				PrevLogIndex: prevLogIndex,
				PrevLogTerm:  prevLogTerm,
				Entries:      entries,
				LeaderCommit: commitIndex,
			})
			if err != nil {
				rn.metrics.replicationErrors.WithLabelValues(p).Inc()
				return
			}

			if resp.Term > currentTerm {
				rn.stepDown(resp.Term)
				return
			}

			rn.mu.Lock()
			if resp.Success {
				rn.nextIndex[p] = nextIdx + uint64(len(entries))
				rn.matchIndex[p] = rn.nextIndex[p] - 1
				rn.metrics.replicationLag.WithLabelValues(p).Set(
					float64(rn.lastLogIndex() - rn.matchIndex[p]),
				)
			} else {
				// Decrement nextIndex and retry (log inconsistency)
				if rn.nextIndex[p] > 1 {
					rn.nextIndex[p]--
				}
			}
			rn.mu.Unlock()
		}(peer)
	}
}

// advanceCommitIndex checks if any new entries can be committed.
func (rn *RaftNode) advanceCommitIndex() {
	rn.mu.Lock()
	defer rn.mu.Unlock()

	for n := rn.commitIndex + 1; n <= rn.lastLogIndex(); n++ {
		if rn.log[n-1].Term != rn.currentTerm {
			continue
		}

		// Count replicas that have this entry
		matches := 1 // Leader always has it
		for _, peer := range rn.config.Peers {
			if rn.matchIndex[peer] >= n {
				matches++
			}
		}

		if matches > (len(rn.config.Peers)+1)/2 {
			rn.commitIndex = n
			rn.metrics.commitIndex.Set(float64(n))
		}
	}

	// Apply committed entries
	for rn.lastApplied < rn.commitIndex {
		rn.lastApplied++
		entry := rn.log[rn.lastApplied-1]
		rn.applyCh <- entry
		rn.metrics.appliedIndex.Set(float64(rn.lastApplied))
		rn.logger.Debug("applied entry",
			zap.Uint64("index", entry.Index),
			zap.Uint64("term", entry.Term),
		)
	}
}

// HandleVoteRequest processes an incoming RequestVote RPC.
func (rn *RaftNode) HandleVoteRequest(req *VoteRequest) *VoteResponse {
	rn.mu.Lock()
	defer rn.mu.Unlock()

	rn.logger.Debug("received vote request",
		zap.String("candidate", req.CandidateID),
		zap.Uint64("term", req.Term),
	)

	// Reply false if term < currentTerm
	if req.Term < rn.currentTerm {
		return &VoteResponse{Term: rn.currentTerm, VoteGranted: false}
	}

	// Step down if higher term
	if req.Term > rn.currentTerm {
		rn.currentTerm = req.Term
		rn.votedFor = ""
		rn.state = Follower
	}

	// Grant vote if we haven't voted or already voted for this candidate,
	// AND candidate's log is at least as up-to-date as ours
	if (rn.votedFor == "" || rn.votedFor == req.CandidateID) &&
		rn.isLogUpToDate(req.LastLogIndex, req.LastLogTerm) {

		rn.votedFor = req.CandidateID
		rn.wal.PersistState(rn.currentTerm, rn.votedFor)
		rn.metrics.votesGranted.Inc()

		// Reset election timer
		select {
		case rn.heartbeatCh <- struct{}{}:
		default:
		}

		return &VoteResponse{Term: rn.currentTerm, VoteGranted: true}
	}

	return &VoteResponse{Term: rn.currentTerm, VoteGranted: false}
}

// HandleAppendEntries processes an incoming AppendEntries RPC.
func (rn *RaftNode) HandleAppendEntries(req *AppendRequest) *AppendResponse {
	rn.mu.Lock()
	defer rn.mu.Unlock()

	// Reply false if term < currentTerm
	if req.Term < rn.currentTerm {
		return &AppendResponse{Term: rn.currentTerm, Success: false}
	}

	// Valid leader -- reset election timer
	select {
	case rn.heartbeatCh <- struct{}{}:
	default:
	}

	// Step down if higher or equal term from a leader
	if req.Term >= rn.currentTerm {
		rn.currentTerm = req.Term
		if rn.state != Follower {
			rn.state = Follower
			rn.metrics.stateTransitions.WithLabelValues("follower").Inc()
		}
	}

	// Log consistency check
	if req.PrevLogIndex > 0 {
		if req.PrevLogIndex > uint64(len(rn.log)) {
			return &AppendResponse{Term: rn.currentTerm, Success: false}
		}
		if rn.log[req.PrevLogIndex-1].Term != req.PrevLogTerm {
			// Delete conflicting entry and everything after
			rn.log = rn.log[:req.PrevLogIndex-1]
			return &AppendResponse{Term: rn.currentTerm, Success: false}
		}
	}

	// Append new entries (skip duplicates)
	for i, entry := range req.Entries {
		idx := req.PrevLogIndex + uint64(i) + 1
		if idx <= uint64(len(rn.log)) {
			if rn.log[idx-1].Term != entry.Term {
				rn.log = rn.log[:idx-1]
				rn.log = append(rn.log, req.Entries[i:]...)
				break
			}
		} else {
			rn.log = append(rn.log, req.Entries[i:]...)
			break
		}
	}

	// Persist new entries to WAL
	for _, entry := range req.Entries {
		rn.wal.Append(entry)
	}

	// Advance commit index
	if req.LeaderCommit > rn.commitIndex {
		oldCommit := rn.commitIndex
		if req.LeaderCommit < rn.lastLogIndex() {
			rn.commitIndex = req.LeaderCommit
		} else {
			rn.commitIndex = rn.lastLogIndex()
		}
		rn.metrics.commitIndex.Set(float64(rn.commitIndex))

		// Apply newly committed entries
		for i := oldCommit + 1; i <= rn.commitIndex; i++ {
			rn.lastApplied = i
			rn.applyCh <- rn.log[i-1]
		}
	}

	return &AppendResponse{Term: rn.currentTerm, Success: true}
}

// -- Helper methods --

func (rn *RaftNode) becomeLeader() {
	rn.mu.Lock()
	defer rn.mu.Unlock()

	rn.state = Leader
	rn.metrics.stateTransitions.WithLabelValues("leader").Inc()
	rn.metrics.leaderElections.Inc()
	rn.logger.Info("became leader",
		zap.Uint64("term", rn.currentTerm),
		zap.Int("log_length", len(rn.log)),
	)
}

func (rn *RaftNode) stepDown(newTerm uint64) {
	rn.mu.Lock()
	defer rn.mu.Unlock()

	rn.logger.Info("stepping down",
		zap.Uint64("old_term", rn.currentTerm),
		zap.Uint64("new_term", newTerm),
	)
	rn.currentTerm = newTerm
	rn.votedFor = ""
	rn.state = Follower
	rn.metrics.stateTransitions.WithLabelValues("follower").Inc()

	rn.wal.PersistState(newTerm, "")

	select {
	case rn.stepDownCh <- struct{}{}:
	default:
	}
}

func (rn *RaftNode) lastLogIndex() uint64 {
	return uint64(len(rn.log))
}

func (rn *RaftNode) lastLogTerm() uint64 {
	if len(rn.log) == 0 {
		return 0
	}
	return rn.log[len(rn.log)-1].Term
}

func (rn *RaftNode) isLogUpToDate(lastIndex, lastTerm uint64) bool {
	myLastTerm := rn.lastLogTerm()
	if lastTerm != myLastTerm {
		return lastTerm > myLastTerm
	}
	return lastIndex >= rn.lastLogIndex()
}

func (rn *RaftNode) electionTimeout() time.Duration {
	spread := rn.config.ElectionMaxMs - rn.config.ElectionMinMs
	ms := rn.config.ElectionMinMs + rand.Intn(spread)
	return time.Duration(ms) * time.Millisecond
}

func (rn *RaftNode) restoreFromWAL() error {
	state, err := rn.wal.LoadState()
	if err != nil {
		return err
	}
	if state != nil {
		rn.currentTerm = state.Term
		rn.votedFor = state.VotedFor
	}

	entries, err := rn.wal.LoadEntries()
	if err != nil {
		return err
	}
	rn.log = entries

	rn.logger.Info("restored from WAL",
		zap.Uint64("term", rn.currentTerm),
		zap.String("voted_for", rn.votedFor),
		zap.Int("entries", len(rn.log)),
	)
	return nil
}

// GetState returns current node state info (thread-safe).
func (rn *RaftNode) GetState() (NodeState, uint64, string) {
	rn.mu.RLock()
	defer rn.mu.RUnlock()
	return rn.state, rn.currentTerm, rn.config.NodeID
}

// IsLeader returns true if this node is the current leader.
func (rn *RaftNode) IsLeader() bool {
	rn.mu.RLock()
	defer rn.mu.RUnlock()
	return rn.state == Leader
}
