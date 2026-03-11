# ============================================================================
# Nexus Distributed AI Inference Engine - Multi-stage Dockerfile
# ============================================================================
# Stage 1: Build Go binary
# Stage 2: Python runtime with PyTorch + Go binary
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder - compile Go binary
# ---------------------------------------------------------------------------
FROM golang:1.24-alpine AS builder

RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /build

# Cache dependency downloads
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w -X main.Version=$(git describe --tags --always --dirty 2>/dev/null || echo dev)" \
    -o /build/nexus \
    ./cmd/nexus/main.go

# ---------------------------------------------------------------------------
# Stage 2: Runtime - Python with PyTorch + Go binary
# ---------------------------------------------------------------------------
FROM python:3.11-slim

LABEL maintainer="nexus-team" \
      description="Nexus Distributed AI Inference Engine" \
      org.opencontainers.image.source="https://github.com/nexus-ai/nexus"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# PyTorch CPU-only build to keep image size reasonable
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy

# Create non-root user
RUN groupadd -r nexus && useradd -r -g nexus -d /app -s /sbin/nologin nexus

WORKDIR /app

# Copy Go binary from builder stage
COPY --from=builder /build/nexus ./nexus
RUN chmod +x ./nexus

# Copy Python runtime package
COPY runtime/ ./runtime/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R nexus:nexus /app

# Switch to non-root user
USER nexus

# Expose gRPC/HTTP port
EXPOSE 8080

# Health check - poll the liveness endpoint every 15s
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health/live || exit 1

ENTRYPOINT ["./nexus", "serve"]
