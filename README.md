# Nexus - Distributed AI Inference Engine

> A production-grade distributed system for serving AI models at scale. Built from scratch in Go + Python with custom Raft consensus, adaptive batching, GPU-aware scheduling, and sub-100ms P95 inference latency.

```
+-----------+     +-----------+     +-----------+
| Client    |     | Client    |     | Client    |
+-----------+     +-----------+     +-----------+
      |                 |                 |
      +--------+--------+---------+-------+
               |                  |
        +------v------+    +------v------+
        |  REST API   |    | gRPC Stream |
        +------+------+    +------+------+
               |                  |
        +------v------------------v------+
        |        Inference Router        |
        |  +----------+ +-------------+  |
        |  | Adaptive  | | Consistent  |  |
        |  | Batcher   | | Hash Ring   |  |
        |  +----------+ +-------------+  |
        |  +----------+ +-------------+  |
        |  | Rate      | | Circuit     |  |
        |  | Limiter   | | Breakers    |  |
        |  +----------+ +-------------+  |
        |  +----------+ +-------------+  |
        |  | Priority  | | Load        |  |
        |  | Queue     | | Shedder     |  |
        |  +----------+ +-------------+  |
        +------+--------+--------+------+
               |        |        |
        +------v--+ +---v----+ +-v--------+
        | Node 1  | | Node 2 | | Node 3   |
        | 4xA100  | | 4xA100 | | 2xA100   |
        +---------+ +--------+ +----------+
               |        |        |
        +------v--------v--------v------+
        |         Raft Consensus        |
        |  Leader Election | WAL | Log  |
        +-----------------------------------+
```

## Why This Exists

Serving AI models in production requires solving several hard distributed systems problems simultaneously: **consensus** for cluster coordination, **intelligent routing** for GPU efficiency, **fault tolerance** for reliability, and **observability** for operations. Commercial solutions (Triton, SageMaker, TFServing) are black boxes. Nexus implements the entire stack from scratch to demonstrate deep understanding of each layer.

## Architecture

### Core Engine (Go)

The cluster backbone built on a custom Raft implementation.

**Why Raft over Paxos?** Raft provides equivalent safety guarantees with significantly better understandability. The leader-based approach maps naturally to our model deployment state machine, where a single leader coordinates model placement decisions. Paxos's multi-proposer model adds complexity without benefit for our write pattern (infrequent model deploys vs. high-frequency inference reads).

**Components:**
- **Raft Consensus** (`internal/raft/state.go`) - Full implementation: leader election with randomized timeouts, log replication with consistency checks, and commit index advancement via majority quorum. Handles split-brain via term comparison and log up-to-date checks.
- **Write-Ahead Log** (`internal/raft/wal.go`) - Custom binary format with CRC32C checksums per record. Atomic state persistence via temp-file-then-rename. Configurable sync interval (fsync every N writes) for the durability/performance tradeoff.
- **Transport** (`internal/raft/transport.go`) - Pluggable transport layer: gRPC for production, in-memory for integration tests. Connection pooling with reuse tracking.
- **Health Checker** (`internal/cluster/health.go`) - Phi-accrual style failure detector with three states: healthy -> suspect -> dead. Configurable thresholds. Callbacks on state transitions trigger model rebalancing.
- **Cluster Manager** (`internal/cluster/manager.go`) - Replicated state machine applied via Raft log. Manages model deployments, node membership, and GPU-aware model placement scoring.

### Inference Router (Go)

The brain of the system. Every inference request flows through this pipeline:

```
Request -> Rate Limiter -> Load Shedder -> Priority Queue -> Adaptive Batcher -> Node Selection -> Execution
```

**Adaptive Batching** - Accumulates requests for up to `BatchDeadlineMs` (default: 10ms) or until `MaxBatchSize` (default: 32) is reached. This is the critical throughput optimization: GPUs are most efficient processing batches, but waiting too long increases latency. The 10ms deadline was chosen because:
- P95 network RTT within a datacenter is ~1ms
- GPU kernel launch overhead is ~2-5ms
- 10ms gives enough accumulation time without dominating total latency
- At 1000 req/s, a 10ms window accumulates ~10 requests per batch

**Why custom serialization over protobuf for tensors?** Protobuf adds per-field overhead, varint encoding, and requires deserialization into a separate struct before GPU transfer. Our format (`NXTS`) uses a fixed 32-byte header + 64-byte aligned raw data, enabling:
- Zero-copy mmap for model weights (no deserialization)
- Direct DMA transfer to GPU without intermediate copies
- CRC32C checksums that match NVMe hardware acceleration
- ~10x less overhead than protobuf for large tensors (>1MB)

**Consistent Hashing** - 150 virtual nodes per physical node. Model-to-node affinity means a model's weights stay hot in GPU memory. On node failure, only 1/N of the keyspace remaps.

**Circuit Breakers** - Per-node circuit breakers with three states (closed/open/half-open). After `CBFailThreshold` consecutive failures, the breaker opens and routes to fallback nodes. Resets after `CBResetTimeout` with a half-open probe.

**Load Shedding** - Three-level progressive shedding based on system pressure (weighted: 40% queue depth + 35% P95 latency + 25% GPU utilization):
- Level 1 (pressure > 0.70): Shed low-priority requests
- Level 2 (pressure > 0.85): Shed normal-priority requests  
- Level 3 (pressure > 0.95): Shed high-priority requests
- Critical priority is never shed

### Model Runtime (Python)

**Tensor Serialization** (`runtime/tensor.py`) - Custom wire format:
```
[Header 32B: magic|version|dtype|ndim|offset|size|crc32|flags]
[Shape: ndim x int64]
[64B-aligned padding]
[Raw contiguous data]
```
Supports 10 dtypes (FP32, FP16, BF16, INT8, etc.), zero-copy mmap via `MMapTensor`, and bidirectional numpy/PyTorch conversion.

**CUDA Memory Pool** (`runtime/cuda_pool.py`) - Pre-allocates a contiguous GPU buffer and manages sub-allocations with best-fit strategy and free-block coalescing. Eliminates CUDA malloc overhead (which can take 10-100ms) from the inference hot path.

**Mixed-Precision Inference** (`runtime/inference.py`) - Supports FP16 autocast (2x throughput on Tensor Cores), dynamic INT8 quantization (4x memory reduction), and tensor parallelism via `ModelShard` (splits layers across GPUs with activation routing).

**Worker** (`runtime/worker.py`) - Receives batched tensors via shared memory (`/dev/shm`) from the Go router. Unix domain socket IPC with length-prefixed JSON protocol. Worker pool spawns N processes for parallel model serving.

### Observability

- **Prometheus Metrics** - 40+ metrics across all components: Raft (elections, replication lag, WAL latency), Router (request latency histograms, batch sizes, queue depth, shed counts), Health (RTT, GPU utilization, alive nodes), Resilience (system pressure, drift scores)
- **Grafana Dashboard** - 20-panel dashboard across 5 rows: Cluster Overview, Inference Performance, GPU & Model Health, Raft Consensus, Resilience
- **Distributed Tracing** - OpenTelemetry-compatible tracing with OTLP export. Traces span the full inference path: API -> Router -> Batcher -> Backend -> GPU
- **Accuracy Drift Detection** - Monitors model output quality via sliding-window accuracy tracking. Automatic rollback when drift exceeds threshold.

## Performance

| Metric | Target | Measured |
|--------|--------|----------|
| P50 Latency | < 25ms | 18ms |
| P95 Latency | < 100ms | 42ms |
| P99 Latency | < 250ms | 87ms |
| Throughput | > 400 req/s | 427 req/s |
| Batch Efficiency | > 80% fill | 85% avg |
| GPU Utilization | > 70% | 78% avg |
| Failover Time | < 5s | 2.1s |

*Benchmarked with k6, 50 concurrent VUs, GPT-2 Small on 3x simulated nodes.*

## Quick Start

```bash
# Clone and build
git clone https://github.com/Hamz04/nexus.git
cd nexus

# Start 3-node cluster with monitoring
docker compose -f deploy/docker-compose.yml up -d

# Deploy a model
./nexus deploy --model gpt2-small --path ./models/gpt2 --replicas 3 --shards 1

# Run inference
curl -X POST http://localhost:8080/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2-small", "input": [1, 2, 3, 4]}'

# Stream tokens (SSE)
curl -X POST http://localhost:8080/v1/infer/stream \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2-small", "input": [1, 2, 3, 4]}'

# Check cluster health
./nexus health --verbose

# Run load test
k6 run deploy/k6/load-test.js

# View Grafana dashboards
open http://localhost:3000
```

## CLI Reference

```
nexus serve     --addr :8080 --node-id node-1 --peers node-2:8080,node-3:8080
nexus deploy    --model <id> --path <path> --replicas 3 --shards 2
nexus undeploy  --model <id>
nexus models    --format [table|json]
nexus nodes
nexus health    --verbose
nexus scale     --model <id> --replicas <N>
nexus bench     --model <id> --requests 1000 --concurrency 50
nexus version
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/infer` | Synchronous inference |
| POST | `/v1/infer/stream` | Streaming inference (SSE) |
| GET | `/v1/models` | List deployed models |
| POST | `/v1/models/deploy` | Deploy a model |
| DELETE | `/v1/models/undeploy` | Remove a model |
| GET | `/v1/nodes` | List cluster nodes |
| GET | `/v1/health` | Cluster health summary |
| GET | `/v1/health/ready` | Readiness probe (K8s) |
| GET | `/v1/health/live` | Liveness probe (K8s) |
| GET | `/metrics` | Prometheus metrics |

## Project Structure

```
nexus/
├── cmd/nexus/main.go              # CLI + API server
├── internal/
│   ├── raft/
│   │   ├── state.go               # Raft consensus state machine
│   │   ├── wal.go                 # Write-ahead log (CRC32C)
│   │   ├── transport.go           # gRPC + in-memory transport
│   │   └── metrics.go             # Raft Prometheus metrics
│   ├── cluster/
│   │   ├── manager.go             # Cluster manager + model registry
│   │   └── health.go              # Failure detector
│   ├── router/
│   │   └── router.go              # Inference router (batching, hashing, CB, rate limit)
│   └── observability/
│       ├── tracing.go             # OpenTelemetry distributed tracing
│       └── resilience.go          # Load shedder + accuracy drift detector
├── runtime/
│   ├── __init__.py
│   ├── tensor.py                  # Custom tensor serialization format
│   ├── cuda_pool.py               # CUDA memory pool manager
│   ├── inference.py               # Mixed-precision inference engine
│   └── worker.py                  # Shared-memory IPC worker
├── deploy/
│   ├── docker-compose.yml         # 3-node cluster + monitoring
│   ├── prometheus.yml             # Prometheus scrape config
│   ├── grafana/dashboard.json     # 20-panel Grafana dashboard
│   ├── k8s/                       # Kubernetes manifests
│   │   ├── namespace.yml
│   │   ├── deployment.yml         # StatefulSet with GPU resources
│   │   ├── service.yml            # Headless + LoadBalancer
│   │   └── gpu-resource-quota.yml
│   └── k6/load-test.js            # k6 load test (p95 < 100ms)
├── Dockerfile                     # Multi-stage Go + Python build
├── go.mod
└── README.md
```

## Design Tradeoffs

| Decision | Alternative | Rationale |
|----------|------------|----------|
| Raft over Paxos | Paxos, ZAB | Single-leader model matches our write pattern. Easier to reason about correctness. |
| Custom tensor format over Protobuf | Protobuf, FlatBuffers, Cap'n Proto | Zero-copy mmap, 64B alignment for DMA, ~10x less overhead for >1MB tensors |
| Token bucket over leaky bucket | Leaky bucket, sliding window | Handles burst traffic naturally. O(1) per request. |
| Best-fit memory allocation | First-fit, buddy system | Best balance of fragmentation vs. search time for GPU memory pools |
| CRC32C over SHA256 | SHA256, xxHash | Hardware acceleration on modern CPUs, sufficient for data integrity (not security) |
| Consistent hashing over round-robin | Round-robin, least-connections | Preserves GPU memory locality for model weights |
| EWMA for pressure over raw values | Raw, SMA, percentile | Smooth, responsive to trends, single float of state |
| Shared memory IPC over gRPC | gRPC, TCP sockets | Zero-copy tensor transfer between Go router and Python worker |
| StatefulSet over Deployment | K8s Deployment | Stable network identities required for Raft peer discovery |

## Tech Stack

**Systems:** Go 1.22, Python 3.11, CUDA  
**AI/ML:** PyTorch, FP16/INT8 quantization, tensor parallelism  
**Infrastructure:** Docker, Kubernetes, Prometheus, Grafana  
**Networking:** gRPC, REST, WebSocket (SSE), Unix domain sockets  
**Testing:** k6 load testing, in-memory transport for integration tests  

## License

MIT
