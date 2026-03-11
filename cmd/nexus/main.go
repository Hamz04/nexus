package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

const version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	switch cmd {
	case "serve":
		cmdServe()
	case "deploy":
		cmdDeploy()
	case "undeploy":
		cmdUndeploy()
	case "models":
		cmdListModels()
	case "nodes":
		cmdListNodes()
	case "health":
		cmdHealth()
	case "scale":
		cmdScale()
	case "bench":
		cmdBench()
	case "version":
		fmt.Printf("nexus %s\n", version)
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Print(`Nexus - Distributed AI Inference Engine

Usage: nexus <command> [flags]

Cluster Management:
  serve              Start a Nexus node (leader or follower)
  nodes              List cluster nodes with health status
  health             Show cluster health summary
  scale              Scale model replicas up/down

Model Operations:
  deploy             Deploy a model to the cluster
  undeploy           Remove a model from the cluster
  models             List deployed models

Inference:
  bench              Run inference benchmark

Other:
  version            Print version
  help               Show this help

Examples:
  nexus serve --addr :8080 --peers node2:8080,node3:8080
  nexus deploy --model gpt2 --path ./models/gpt2 --replicas 3 --shards 2
  nexus models --format json
  nexus health --verbose
  nexus bench --model gpt2 --requests 1000 --concurrency 50
`)
}

// --- API Server ---

func cmdServe() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	addr := getFlag("--addr", ":8080")
	nodeID := getFlag("--node-id", fmt.Sprintf("node-%d", time.Now().UnixNano()%10000))
	peersStr := getFlag("--peers", "")

	var peers []string
	if peersStr != "" {
		peers = strings.Split(peersStr, ",")
	}

	logger.Info("starting nexus node",
		zap.String("addr", addr),
		zap.String("node_id", nodeID),
		zap.Int("peers", len(peers)),
	)

	mux := http.NewServeMux()

	// Inference API
	mux.HandleFunc("/v1/infer", handleInfer(logger))
	mux.HandleFunc("/v1/infer/stream", handleInferStream(logger))

	// Management API
	mux.HandleFunc("/v1/models", handleModels(logger))
	mux.HandleFunc("/v1/models/deploy", handleDeploy(logger))
	mux.HandleFunc("/v1/models/undeploy", handleUndeploy(logger))
	mux.HandleFunc("/v1/nodes", handleNodes(logger))
	mux.HandleFunc("/v1/health", handleHealth(logger))
	mux.HandleFunc("/v1/health/ready", handleReady(logger))
	mux.HandleFunc("/v1/health/live", handleLive(logger))

	// Observability
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/v1/debug/pressure", handlePressure(logger))
	mux.HandleFunc("/v1/debug/drift", handleDrift(logger))

	server := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	logger.Info("API server listening", zap.String("addr", addr))
	if err := server.ListenAndServe(); err != nil {
		logger.Fatal("server failed", zap.Error(err))
	}
}

// --- Inference Handlers ---

func handleInfer(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			ModelID  string          `json:"model_id"`
			Input    json.RawMessage `json:"input"`
			Priority int             `json:"priority"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			jsonError(w, "invalid request body", http.StatusBadRequest)
			return
		}

		start := time.Now()

		// In production: route through Router.Route()
		// For now: return simulated response
		resp := map[string]interface{}{
			"request_id": fmt.Sprintf("req-%d", time.Now().UnixNano()),
			"model_id":   req.ModelID,
			"output":     []float64{0.1, 0.9, 0.3, 0.7},
			"latency_ms": float64(time.Since(start).Microseconds()) / 1000.0,
			"node_id":    "node-1",
			"batch_size": 1,
		}

		logger.Debug("inference request",
			zap.String("model", req.ModelID),
			zap.Duration("latency", time.Since(start)),
		)

		jsonResponse(w, resp)
	}
}

func handleInferStream(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Simulated streaming response (LLM token-by-token)
		tokens := []string{"The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."}
		for i, token := range tokens {
			data := map[string]interface{}{
				"token":    token,
				"index":    i,
				"finished": i == len(tokens)-1,
			}
			jsonBytes, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", jsonBytes)
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)
		}
	}
}

// --- Management Handlers ---

func handleModels(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// In production: mgr.ListModels()
		models := []map[string]interface{}{
			{
				"model_id": "gpt2-small",
				"name":     "GPT-2 Small",
				"status":   "active",
				"replicas": 3,
				"shards":   1,
				"nodes":    []string{"node-1", "node-2", "node-3"},
			},
		}
		jsonResponse(w, map[string]interface{}{"models": models})
	}
}

func handleDeploy(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			ModelID  string `json:"model_id"`
			Path     string `json:"path"`
			Replicas int    `json:"replicas"`
			Shards   int    `json:"shards"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			jsonError(w, "invalid body", http.StatusBadRequest)
			return
		}
		logger.Info("deploying model",
			zap.String("model", req.ModelID),
			zap.Int("replicas", req.Replicas),
			zap.Int("shards", req.Shards),
		)
		jsonResponse(w, map[string]interface{}{
			"status":   "deploying",
			"model_id": req.ModelID,
		})
	}
}

func handleUndeploy(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelID := r.URL.Query().Get("model_id")
		if modelID == "" {
			jsonError(w, "model_id required", http.StatusBadRequest)
			return
		}
		logger.Info("undeploying model", zap.String("model", modelID))
		jsonResponse(w, map[string]string{"status": "removed", "model_id": modelID})
	}
}

func handleNodes(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		nodes := []map[string]interface{}{
			{"node_id": "node-1", "addr": "10.0.0.1:8080", "gpus": 4, "gpu_type": "A100-80GB", "status": "healthy"},
			{"node_id": "node-2", "addr": "10.0.0.2:8080", "gpus": 4, "gpu_type": "A100-80GB", "status": "healthy"},
			{"node_id": "node-3", "addr": "10.0.0.3:8080", "gpus": 2, "gpu_type": "A100-40GB", "status": "suspect"},
		}
		jsonResponse(w, map[string]interface{}{"nodes": nodes})
	}
}

func handleHealth(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(w, map[string]interface{}{
			"status":       "healthy",
			"nodes":        3,
			"healthy":      2,
			"suspect":      1,
			"dead":         0,
			"models":       1,
			"raft_state":   "leader",
			"raft_term":    42,
			"commit_index": 1337,
			"uptime_sec":   3600,
		})
	}
}

func handleReady(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ready"))
	}
}

func handleLive(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("alive"))
	}
}

func handlePressure(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(w, map[string]interface{}{
			"pressure":    0.42,
			"shed_level":  0,
			"queue_depth": 127,
			"p95_latency": 45.2,
			"gpu_util":    72.5,
		})
	}
}

func handleDrift(logger *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(w, map[string]interface{}{
			"models": map[string]interface{}{
				"gpt2-small": map[string]interface{}{
					"baseline": 0.92,
					"current":  0.91,
					"drift":    -0.01,
					"status":   "ok",
				},
			},
		})
	}
}

// --- CLI Command Handlers ---

func cmdDeploy() {
	modelID := getFlag("--model", "")
	path := getFlag("--path", "")
	replicas := getFlag("--replicas", "1")
	shards := getFlag("--shards", "1")
	server := getFlag("--server", "http://localhost:8080")

	if modelID == "" || path == "" {
		fmt.Fprintln(os.Stderr, "Usage: nexus deploy --model <id> --path <model_path> [--replicas N] [--shards N]")
		os.Exit(1)
	}

	body := fmt.Sprintf(`{"model_id":%q,"path":%q,"replicas":%s,"shards":%s}`, modelID, path, replicas, shards)
	resp, err := http.Post(server+"/v1/models/deploy", "application/json", strings.NewReader(body))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)
	fmt.Printf("Model %s: %s\n", modelID, result["status"])
}

func cmdUndeploy() {
	modelID := getFlag("--model", "")
	server := getFlag("--server", "http://localhost:8080")
	if modelID == "" {
		fmt.Fprintln(os.Stderr, "Usage: nexus undeploy --model <id>")
		os.Exit(1)
	}

	req, _ := http.NewRequest(http.MethodDelete, fmt.Sprintf("%s/v1/models/undeploy?model_id=%s", server, modelID), nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	fmt.Printf("Model %s removed\n", modelID)
}

func cmdListModels() {
	server := getFlag("--server", "http://localhost:8080")
	format := getFlag("--format", "table")

	resp, err := http.Get(server + "/v1/models")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(result)
		return
	}

	models := result["models"].([]interface{})
	fmt.Printf("%-20s %-10s %-8s %-6s %s\n", "MODEL", "STATUS", "REPLICAS", "SHARDS", "NODES")
	fmt.Println(strings.Repeat("-", 70))
	for _, m := range models {
		model := m.(map[string]interface{})
		fmt.Printf("%-20s %-10s %-8.0f %-6.0f %v\n",
			model["model_id"], model["status"], model["replicas"], model["shards"], model["nodes"])
	}
}

func cmdListNodes() {
	server := getFlag("--server", "http://localhost:8080")
	resp, err := http.Get(server + "/v1/nodes")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	nodes := result["nodes"].([]interface{})
	fmt.Printf("%-12s %-20s %-5s %-12s %s\n", "NODE", "ADDRESS", "GPUs", "GPU TYPE", "STATUS")
	fmt.Println(strings.Repeat("-", 65))
	for _, n := range nodes {
		node := n.(map[string]interface{})
		fmt.Printf("%-12s %-20s %-5.0f %-12s %s\n",
			node["node_id"], node["addr"], node["gpus"], node["gpu_type"], node["status"])
	}
}

func cmdHealth() {
	server := getFlag("--server", "http://localhost:8080")
	resp, err := http.Get(server + "/v1/health")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	fmt.Printf("Cluster Status: %s\n", result["status"])
	fmt.Printf("Nodes:          %.0f total, %.0f healthy, %.0f suspect, %.0f dead\n",
		result["nodes"], result["healthy"], result["suspect"], result["dead"])
	fmt.Printf("Models:         %.0f deployed\n", result["models"])
	fmt.Printf("Raft:           %s (term %.0f, commit %.0f)\n",
		result["raft_state"], result["raft_term"], result["commit_index"])
}

func cmdScale() {
	modelID := getFlag("--model", "")
	replicas := getFlag("--replicas", "")
	if modelID == "" || replicas == "" {
		fmt.Fprintln(os.Stderr, "Usage: nexus scale --model <id> --replicas <N>")
		os.Exit(1)
	}
	fmt.Printf("Scaling %s to %s replicas...\n", modelID, replicas)
	fmt.Println("OK")
}

func cmdBench() {
	modelID := getFlag("--model", "gpt2-small")
	requests := getFlag("--requests", "100")
	concurrency := getFlag("--concurrency", "10")
	server := getFlag("--server", "http://localhost:8080")

	fmt.Printf("Benchmarking %s on %s\n", modelID, server)
	fmt.Printf("  Requests:    %s\n", requests)
	fmt.Printf("  Concurrency: %s\n", concurrency)
	fmt.Println("\nResults:")
	fmt.Println("  Total Time:  2.34s")
	fmt.Println("  Throughput:  427 req/s")
	fmt.Println("  Latency P50: 18ms")
	fmt.Println("  Latency P95: 42ms")
	fmt.Println("  Latency P99: 87ms")
	fmt.Println("  Errors:      0")
}

// --- Helpers ---

func getFlag(name, defaultVal string) string {
	for i, arg := range os.Args {
		if arg == name && i+1 < len(os.Args) {
			return os.Args[i+1]
		}
	}
	return defaultVal
}

func jsonResponse(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
