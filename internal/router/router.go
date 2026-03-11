package router

import (
	"context"
	"fmt"
	"hash/fnv"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// Priority levels for inference requests.
type Priority int

const (
	PriorityLow Priority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
)

func (p Priority) String() string {
	switch p {
	case PriorityLow:
		return "low"
	case PriorityNormal:
		return "normal"
	case PriorityHigh:
		return "high"
	case PriorityCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// InferenceRequest represents an incoming inference request.
type InferenceRequest struct {
	ID        string
	ModelID   string
	Input     []byte       // serialized tensor data
	Priority  Priority
	Deadline  time.Time
	CreatedAt time.Time
	ResultCh  chan *InferenceResult
	Stream    bool         // if true, results are streamed token-by-token
	StreamCh  chan *StreamToken
}

// InferenceResult is the output of an inference request.
type InferenceResult struct {
	RequestID  string
	Output     []byte
	LatencyMs  float64
	NodeID     string
	BatchSize  int
	Err        error
}

// StreamToken is a single token in a streaming inference response.
type StreamToken struct {
	Token     string
	Index     int
	LogProb   float64
	Finished  bool
}

// RouterConfig holds tunable parameters for the router.
type RouterConfig struct {
	// Adaptive batching
	MaxBatchSize     int
	BatchDeadlineMs  int           // max ms to wait for batch to fill
	BatchTimeoutMs   int           // max total time for batch execution

	// Rate limiting
	TokensPerSecond  float64
	BurstSize        int

	// Load shedding
	MaxQueueDepth    int
	ShedPriority     Priority      // shed requests at or below this priority

	// Circuit breaker
	CBFailThreshold  int
	CBResetTimeout   time.Duration

	// Consistent hashing
	VirtualNodes     int
}

func DefaultRouterConfig() RouterConfig {
	return RouterConfig{
		MaxBatchSize:    32,
		BatchDeadlineMs: 10,
		BatchTimeoutMs:  5000,
		TokensPerSecond: 1000,
		BurstSize:       100,
		MaxQueueDepth:   10000,
		ShedPriority:    PriorityLow,
		CBFailThreshold: 5,
		CBResetTimeout:  30 * time.Second,
		VirtualNodes:    150,
	}
}

// Router is the intelligent inference request router.
type Router struct {
	mu     sync.RWMutex
	config RouterConfig
	logger *zap.Logger

	// Consistent hash ring for model-to-node affinity
	hashRing *ConsistentHashRing

	// Per-model batch accumulators
	batchers map[string]*AdaptiveBatcher

	// Per-node circuit breakers
	circuitBreakers map[string]*CircuitBreaker

	// Global rate limiter
	rateLimiter *TokenBucketLimiter

	// Priority queue
	queue *PriorityQueue

	// Node backends
	backends map[string]Backend

	// Metrics
	metrics *RouterMetrics

	// Lifecycle
	stopCh chan struct{}
	wg     sync.WaitGroup
}

// Backend is the interface to a model-serving node.
type Backend interface {
	Infer(ctx context.Context, batch []*InferenceRequest) ([]*InferenceResult, error)
	InferStream(ctx context.Context, req *InferenceRequest) (<-chan *StreamToken, error)
	NodeID() string
	QueueDepth() int
}

// RouterMetrics holds Prometheus metrics for the router.
type RouterMetrics struct {
	requestsTotal     *prometheus.CounterVec
	requestLatency    *prometheus.HistogramVec
	batchSize         prometheus.Histogram
	batchWaitTime     prometheus.Histogram
	queueDepth        prometheus.Gauge
	queueWaitTime     *prometheus.HistogramVec
	requestsShed      *prometheus.CounterVec
	activeRequests    prometheus.Gauge
	routeDecisions    *prometheus.CounterVec
	streamTokens      prometheus.Counter
}

func newRouterMetrics() *RouterMetrics {
	return &RouterMetrics{
		requestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "nexus",
				Subsystem: "router",
				Name:      "requests_total",
				Help:      "Total inference requests by model and status.",
			},
			[]string{"model", "status"},
		),
		requestLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "nexus",
				Subsystem: "router",
				Name:      "request_duration_seconds",
				Help:      "End-to-end inference latency.",
				Buckets:   []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5},
			},
			[]string{"model", "priority"},
		),
		batchSize: promauto.NewHistogram(prometheus.HistogramOpts{
			Namespace: "nexus",
			Subsystem: "router",
			Name:      "batch_size",
			Help:      "Number of requests per batch.",
			Buckets:   []float64{1, 2, 4, 8, 16, 32, 64},
		}),
		batchWaitTime: promauto.NewHistogram(prometheus.HistogramOpts{
			Namespace: "nexus",
			Subsystem: "router",
			Name:      "batch_wait_seconds",
			Help:      "Time spent accumulating a batch.",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 10),
		}),
		queueDepth: promauto.NewGauge(prometheus.GaugeOpts{
			Namespace: "nexus",
			Subsystem: "router",
			Name:      "queue_depth",
			Help:      "Current request queue depth.",
		}),
		queueWaitTime: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "nexus",
				Subsystem: "router",
				Name:      "queue_wait_seconds",
				Help:      "Time requests spend in queue.",
				Buckets:   prometheus.ExponentialBuckets(0.001, 2, 12),
			},
			[]string{"priority"},
		),
		requestsShed: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "nexus",
				Subsystem: "router",
				Name:      "requests_shed_total",
				Help:      "Requests dropped due to load shedding.",
			},
			[]string{"model", "priority"},
		),
		activeRequests: promauto.NewGauge(prometheus.GaugeOpts{
			Namespace: "nexus",
			Subsystem: "router",
			Name:      "active_requests",
			Help:      "Currently in-flight inference requests.",
		}),
		routeDecisions: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "nexus",
				Subsystem: "router",
				Name:      "route_decisions_total",
				Help:      "Routing decisions by type (hash, fallback, overflow).",
			},
			[]string{"type"},
		),
		streamTokens: promauto.NewCounter(prometheus.CounterOpts{
			Namespace: "nexus",
			Subsystem: "router",
			Name:      "stream_tokens_total",
			Help:      "Total tokens emitted via streaming.",
		}),
	}
}

// NewRouter creates a new inference router.
func NewRouter(cfg RouterConfig, logger *zap.Logger) *Router {
	r := &Router{
		config:          cfg,
		logger:          logger.Named("router"),
		hashRing:        NewConsistentHashRing(cfg.VirtualNodes),
		batchers:        make(map[string]*AdaptiveBatcher),
		circuitBreakers: make(map[string]*CircuitBreaker),
		rateLimiter:     NewTokenBucketLimiter(cfg.TokensPerSecond, cfg.BurstSize),
		queue:           NewPriorityQueue(),
		backends:        make(map[string]Backend),
		metrics:         newRouterMetrics(),
		stopCh:          make(chan struct{}),
	}
	return r
}

// AddBackend registers a node backend for routing.
func (r *Router) AddBackend(nodeID string, b Backend) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.backends[nodeID] = b
	r.hashRing.Add(nodeID)
	r.circuitBreakers[nodeID] = NewCircuitBreaker(r.config.CBFailThreshold, r.config.CBResetTimeout)

	r.logger.Info("backend added",
		zap.String("node", nodeID),
		zap.Int("total_backends", len(r.backends)),
	)
}

// RemoveBackend removes a node backend.
func (r *Router) RemoveBackend(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.backends, nodeID)
	r.hashRing.Remove(nodeID)
	delete(r.circuitBreakers, nodeID)

	r.logger.Info("backend removed", zap.String("node", nodeID))
}

// Route submits an inference request through the router pipeline:
// rate limit -> priority queue -> batch accumulator -> node selection -> execution
func (r *Router) Route(ctx context.Context, req *InferenceRequest) (*InferenceResult, error) {
	req.CreatedAt = time.Now()
	req.ResultCh = make(chan *InferenceResult, 1)

	// 1. Rate limiting
	if !r.rateLimiter.Allow() {
		r.metrics.requestsShed.WithLabelValues(req.ModelID, req.Priority.String()).Inc()
		return nil, fmt.Errorf("rate limited")
	}

	// 2. Load shedding: reject low-priority requests when queue is deep
	qDepth := r.queue.Len()
	r.metrics.queueDepth.Set(float64(qDepth))
	if qDepth >= r.config.MaxQueueDepth && req.Priority <= r.config.ShedPriority {
		r.metrics.requestsShed.WithLabelValues(req.ModelID, req.Priority.String()).Inc()
		r.logger.Warn("shedding request",
			zap.String("model", req.ModelID),
			zap.String("priority", req.Priority.String()),
			zap.Int("queue_depth", qDepth),
		)
		return nil, fmt.Errorf("load shedding: queue depth %d exceeds limit", qDepth)
	}

	// 3. Streaming requests bypass batching
	if req.Stream {
		return r.routeStream(ctx, req)
	}

	// 4. Get or create batcher for this model
	batcher := r.getBatcher(req.ModelID)

	// 5. Submit to batcher (non-blocking)
	batcher.Submit(req)
	r.metrics.activeRequests.Inc()

	// 6. Wait for result
	select {
	case result := <-req.ResultCh:
		r.metrics.activeRequests.Dec()
		latency := time.Since(req.CreatedAt).Seconds()
		r.metrics.requestLatency.WithLabelValues(req.ModelID, req.Priority.String()).Observe(latency)
		if result.Err != nil {
			r.metrics.requestsTotal.WithLabelValues(req.ModelID, "error").Inc()
		} else {
			r.metrics.requestsTotal.WithLabelValues(req.ModelID, "success").Inc()
		}
		return result, result.Err

	case <-ctx.Done():
		r.metrics.activeRequests.Dec()
		r.metrics.requestsTotal.WithLabelValues(req.ModelID, "timeout").Inc()
		return nil, ctx.Err()
	}
}

// routeStream handles streaming inference (LLM token-by-token).
func (r *Router) routeStream(ctx context.Context, req *InferenceRequest) (*InferenceResult, error) {
	nodeID := r.selectNode(req.ModelID)
	if nodeID == "" {
		return nil, fmt.Errorf("no available backend for model %s", req.ModelID)
	}

	r.mu.RLock()
	backend, ok := r.backends[nodeID]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("backend %s not found", nodeID)
	}

	// Check circuit breaker
	r.mu.RLock()
	cb := r.circuitBreakers[nodeID]
	r.mu.RUnlock()
	if cb != nil && !cb.Allow() {
		return nil, fmt.Errorf("circuit breaker open for node %s", nodeID)
	}

	req.StreamCh = make(chan *StreamToken, 256)

	tokenCh, err := backend.InferStream(ctx, req)
	if err != nil {
		if cb != nil {
			cb.RecordFailure()
		}
		return nil, fmt.Errorf("stream inference failed: %w", err)
	}

	// Forward tokens from backend to request's stream channel
	go func() {
		defer close(req.StreamCh)
		for token := range tokenCh {
			req.StreamCh <- token
			r.metrics.streamTokens.Inc()
		}
	}()

	if cb != nil {
		cb.RecordSuccess()
	}

	return &InferenceResult{
		RequestID: req.ID,
		NodeID:    nodeID,
		LatencyMs: float64(time.Since(req.CreatedAt).Milliseconds()),
	}, nil
}

// getBatcher returns or creates an adaptive batcher for a model.
func (r *Router) getBatcher(modelID string) *AdaptiveBatcher {
	r.mu.Lock()
	defer r.mu.Unlock()

	if b, ok := r.batchers[modelID]; ok {
		return b
	}

	b := NewAdaptiveBatcher(AdaptiveBatcherConfig{
		MaxBatch:   r.config.MaxBatchSize,
		DeadlineMs: r.config.BatchDeadlineMs,
		TimeoutMs:  r.config.BatchTimeoutMs,
	}, func(batch []*InferenceRequest) {
		r.executeBatch(modelID, batch)
	}, r.logger)

	r.batchers[modelID] = b
	go b.Run(r.stopCh)

	r.logger.Info("created batcher", zap.String("model", modelID))
	return b
}

// executeBatch sends a batch of requests to the selected backend node.
func (r *Router) executeBatch(modelID string, batch []*InferenceRequest) {
	if len(batch) == 0 {
		return
	}

	r.metrics.batchSize.Observe(float64(len(batch)))

	// Select node using consistent hashing
	nodeID := r.selectNode(modelID)
	if nodeID == "" {
		err := fmt.Errorf("no available backend for model %s", modelID)
		for _, req := range batch {
			req.ResultCh <- &InferenceResult{RequestID: req.ID, Err: err}
		}
		return
	}

	r.mu.RLock()
	backend, ok := r.backends[nodeID]
	cb := r.circuitBreakers[nodeID]
	r.mu.RUnlock()

	if !ok {
		err := fmt.Errorf("backend %s disappeared", nodeID)
		for _, req := range batch {
			req.ResultCh <- &InferenceResult{RequestID: req.ID, Err: err}
		}
		return
	}

	// Circuit breaker check
	if cb != nil && !cb.Allow() {
		// Try fallback node
		nodeID = r.selectFallbackNode(modelID, nodeID)
		if nodeID == "" {
			err := fmt.Errorf("all backends exhausted for model %s", modelID)
			for _, req := range batch {
				req.ResultCh <- &InferenceResult{RequestID: req.ID, Err: err}
			}
			return
		}
		r.mu.RLock()
		backend = r.backends[nodeID]
		cb = r.circuitBreakers[nodeID]
		r.mu.RUnlock()
		r.metrics.routeDecisions.WithLabelValues("fallback").Inc()
	}

	// Execute batch on backend
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(r.config.BatchTimeoutMs)*time.Millisecond)
	defer cancel()

	results, err := backend.Infer(ctx, batch)
	if err != nil {
		if cb != nil {
			cb.RecordFailure()
		}
		for _, req := range batch {
			req.ResultCh <- &InferenceResult{RequestID: req.ID, Err: err}
		}
		return
	}

	if cb != nil {
		cb.RecordSuccess()
	}

	// Dispatch results
	for i, result := range results {
		result.NodeID = nodeID
		result.BatchSize = len(batch)
		if i < len(batch) {
			batch[i].ResultCh <- result
		}
	}
}

// selectNode uses consistent hashing to pick the primary node for a model.
func (r *Router) selectNode(modelID string) string {
	nodeID := r.hashRing.Get(modelID)
	if nodeID == "" {
		return ""
	}

	r.mu.RLock()
	_, ok := r.backends[nodeID]
	r.mu.RUnlock()

	if ok {
		r.metrics.routeDecisions.WithLabelValues("hash").Inc()
		return nodeID
	}
	return r.selectFallbackNode(modelID, nodeID)
}

// selectFallbackNode finds an alternate node, skipping the failed one.
func (r *Router) selectFallbackNode(modelID string, exclude string) string {
	nodes := r.hashRing.GetN(modelID, 3)
	for _, n := range nodes {
		if n == exclude {
			continue
		}
		r.mu.RLock()
		_, ok := r.backends[n]
		r.mu.RUnlock()
		if ok {
			cb := r.circuitBreakers[n]
			if cb == nil || cb.Allow() {
				r.metrics.routeDecisions.WithLabelValues("fallback").Inc()
				return n
			}
		}
	}
	return ""
}

// Stop gracefully shuts down the router.
func (r *Router) Stop() {
	close(r.stopCh)
	r.wg.Wait()
	r.logger.Info("router stopped")
}

// --- Consistent Hash Ring ---

// ConsistentHashRing implements consistent hashing with virtual nodes.
type ConsistentHashRing struct {
	mu           sync.RWMutex
	virtualNodes int
	hashMap      map[uint32]string
	keys         []uint32 // sorted
}

func NewConsistentHashRing(virtualNodes int) *ConsistentHashRing {
	return &ConsistentHashRing{
		virtualNodes: virtualNodes,
		hashMap:      make(map[uint32]string),
	}
}

func (c *ConsistentHashRing) Add(nodeID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := 0; i < c.virtualNodes; i++ {
		h := hashKey(fmt.Sprintf("%s#%d", nodeID, i))
		c.hashMap[h] = nodeID
		c.keys = append(c.keys, h)
	}
	sort.Slice(c.keys, func(i, j int) bool { return c.keys[i] < c.keys[j] })
}

func (c *ConsistentHashRing) Remove(nodeID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	newKeys := make([]uint32, 0, len(c.keys))
	for _, k := range c.keys {
		if c.hashMap[k] != nodeID {
			newKeys = append(newKeys, k)
		} else {
			delete(c.hashMap, k)
		}
	}
	c.keys = newKeys
}

func (c *ConsistentHashRing) Get(key string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.keys) == 0 {
		return ""
	}

	h := hashKey(key)
	idx := sort.Search(len(c.keys), func(i int) bool { return c.keys[i] >= h })
	if idx >= len(c.keys) {
		idx = 0
	}
	return c.hashMap[c.keys[idx]]
}

func (c *ConsistentHashRing) GetN(key string, n int) []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.keys) == 0 {
		return nil
	}

	h := hashKey(key)
	idx := sort.Search(len(c.keys), func(i int) bool { return c.keys[i] >= h })

	seen := make(map[string]bool)
	var result []string
	for i := 0; i < len(c.keys) && len(result) < n; i++ {
		pos := (idx + i) % len(c.keys)
		nodeID := c.hashMap[c.keys[pos]]
		if !seen[nodeID] {
			seen[nodeID] = true
			result = append(result, nodeID)
		}
	}
	return result
}

func hashKey(key string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(key))
	return h.Sum32()
}

// --- Adaptive Batcher ---

// AdaptiveBatcherConfig holds batcher configuration.
type AdaptiveBatcherConfig struct {
	MaxBatch   int
	DeadlineMs int
	TimeoutMs  int
}

// AdaptiveBatcher accumulates requests and dispatches as batches.
// It uses a deadline-based strategy: wait up to DeadlineMs for the batch
// to fill to MaxBatch, then dispatch whatever has accumulated.
type AdaptiveBatcher struct {
	mu       sync.Mutex
	config   AdaptiveBatcherConfig
	pending  []*InferenceRequest
	execFn   func([]*InferenceRequest)
	logger   *zap.Logger
	submitCh chan *InferenceRequest
}

func NewAdaptiveBatcher(cfg AdaptiveBatcherConfig, execFn func([]*InferenceRequest), logger *zap.Logger) *AdaptiveBatcher {
	return &AdaptiveBatcher{
		config:   cfg,
		execFn:   execFn,
		logger:   logger.Named("batcher"),
		submitCh: make(chan *InferenceRequest, 4096),
	}
}

func (b *AdaptiveBatcher) Submit(req *InferenceRequest) {
	b.submitCh <- req
}

func (b *AdaptiveBatcher) Run(stopCh <-chan struct{}) {
	deadline := time.Duration(b.config.DeadlineMs) * time.Millisecond
	var timer *time.Timer

	for {
		select {
		case req := <-b.submitCh:
			b.mu.Lock()
			b.pending = append(b.pending, req)

			if len(b.pending) >= b.config.MaxBatch {
				// Batch is full, dispatch immediately
				batch := b.pending
				b.pending = nil
				b.mu.Unlock()
				if timer != nil {
					timer.Stop()
					timer = nil
				}
				go b.execFn(batch)
			} else if len(b.pending) == 1 {
				// First request in new batch -- start deadline timer
				b.mu.Unlock()
				timer = time.AfterFunc(deadline, func() {
					b.mu.Lock()
					if len(b.pending) > 0 {
						batch := b.pending
						b.pending = nil
						b.mu.Unlock()
						go b.execFn(batch)
					} else {
						b.mu.Unlock()
					}
				})
			} else {
				b.mu.Unlock()
			}

		case <-stopCh:
			// Flush remaining
			b.mu.Lock()
			if len(b.pending) > 0 {
				batch := b.pending
				b.pending = nil
				b.mu.Unlock()
				b.execFn(batch)
			} else {
				b.mu.Unlock()
			}
			return
		}
	}
}

// --- Token Bucket Rate Limiter ---

type TokenBucketLimiter struct {
	tokens     float64
	maxTokens  float64
	refillRate float64 // tokens per second
	lastRefill time.Time
	mu         sync.Mutex
}

func NewTokenBucketLimiter(rate float64, burst int) *TokenBucketLimiter {
	return &TokenBucketLimiter{
		tokens:     float64(burst),
		maxTokens:  float64(burst),
		refillRate: rate,
		lastRefill: time.Now(),
	}
}

func (l *TokenBucketLimiter) Allow() bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(l.lastRefill).Seconds()
	l.tokens += elapsed * l.refillRate
	if l.tokens > l.maxTokens {
		l.tokens = l.maxTokens
	}
	l.lastRefill = now

	if l.tokens >= 1 {
		l.tokens--
		return true
	}
	return false
}

// --- Circuit Breaker ---

type CircuitBreakerState int

const (
	CBClosed CircuitBreakerState = iota
	CBOpen
	CBHalfOpen
)

type CircuitBreaker struct {
	state         atomic.Int32
	failCount     atomic.Int32
	successCount  atomic.Int32
	failThreshold int
	resetTimeout  time.Duration
	lastFailure   time.Time
	mu            sync.Mutex
}

func NewCircuitBreaker(threshold int, resetTimeout time.Duration) *CircuitBreaker {
	cb := &CircuitBreaker{
		failThreshold: threshold,
		resetTimeout:  resetTimeout,
	}
	cb.state.Store(int32(CBClosed))
	return cb
}

func (cb *CircuitBreaker) Allow() bool {
	state := CircuitBreakerState(cb.state.Load())

	switch state {
	case CBClosed:
		return true
	case CBOpen:
		cb.mu.Lock()
		elapsed := time.Since(cb.lastFailure)
		cb.mu.Unlock()

		if elapsed > cb.resetTimeout {
			cb.state.Store(int32(CBHalfOpen))
			cb.successCount.Store(0)
			return true
		}
		return false
	case CBHalfOpen:
		return true
	}
	return false
}

func (cb *CircuitBreaker) RecordSuccess() {
	state := CircuitBreakerState(cb.state.Load())

	if state == CBHalfOpen {
		if cb.successCount.Add(1) >= 3 {
			cb.state.Store(int32(CBClosed))
			cb.failCount.Store(0)
		}
	}
	if state == CBClosed {
		cb.failCount.Store(0)
	}
}

func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	cb.lastFailure = time.Now()
	cb.mu.Unlock()

	if cb.failCount.Add(1) >= int32(cb.failThreshold) {
		cb.state.Store(int32(CBOpen))
	}
}

func (cb *CircuitBreaker) State() CircuitBreakerState {
	return CircuitBreakerState(cb.state.Load())
}

// --- Priority Queue ---

// PriorityQueue is a thread-safe priority queue for inference requests.
type PriorityQueue struct {
	mu    sync.Mutex
	items []*InferenceRequest
	len_  int32
}

func NewPriorityQueue() *PriorityQueue {
	return &PriorityQueue{
		items: make([]*InferenceRequest, 0, 1024),
	}
}

func (pq *PriorityQueue) Push(req *InferenceRequest) {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	// Insert maintaining priority order (highest priority first)
	inserted := false
	for i, item := range pq.items {
		if req.Priority > item.Priority {
			pq.items = append(pq.items[:i+1], pq.items[i:]...)
			pq.items[i] = req
			inserted = true
			break
		}
	}
	if !inserted {
		pq.items = append(pq.items, req)
	}
	atomic.AddInt32(&pq.len_, 1)
}

func (pq *PriorityQueue) Pop() *InferenceRequest {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(pq.items) == 0 {
		return nil
	}

	req := pq.items[0]
	pq.items = pq.items[1:]
	atomic.AddInt32(&pq.len_, -1)
	return req
}

func (pq *PriorityQueue) Len() int {
	return int(atomic.LoadInt32(&pq.len_))
}
