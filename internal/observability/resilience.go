package observability

import (
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// LoadShedder implements progressive load shedding based on system pressure.
// Under increasing load, it sheds lowest-priority requests first.
type LoadShedder struct {
	mu         sync.RWMutex
	config     LoadShedConfig
	logger     *zap.Logger
	pressure   float64 // 0.0 (idle) to 1.0 (overloaded)
	window     []pressureSample

	// Metrics
	shedTotal    *prometheus.CounterVec
	pressureGauge prometheus.Gauge
	activeLevels  *prometheus.GaugeVec
}

type pressureSample struct {
	timestamp time.Time
	value     float64
}

// LoadShedConfig configures the load shedder thresholds.
type LoadShedConfig struct {
	// Pressure thresholds for each shed level
	Level1Threshold float64 // Start shedding low priority (default 0.7)
	Level2Threshold float64 // Start shedding normal priority (default 0.85)
	Level3Threshold float64 // Start shedding high priority (default 0.95)
	// Critical priority is never shed

	WindowSize     int           // Pressure averaging window
	SampleInterval time.Duration // How often to sample

	// Inputs for pressure calculation
	QueueCapacity  int
	MaxLatencyMs   float64
	MaxGPUUtil     float64
}

func DefaultLoadShedConfig() LoadShedConfig {
	return LoadShedConfig{
		Level1Threshold: 0.70,
		Level2Threshold: 0.85,
		Level3Threshold: 0.95,
		WindowSize:      30,
		SampleInterval:  1 * time.Second,
		QueueCapacity:   10000,
		MaxLatencyMs:    5000,
		MaxGPUUtil:      100,
	}
}

// NewLoadShedder creates a new load shedder.
func NewLoadShedder(cfg LoadShedConfig, logger *zap.Logger) *LoadShedder {
	return &LoadShedder{
		config: cfg,
		logger: logger.Named("load-shedder"),
		window: make([]pressureSample, 0, cfg.WindowSize),
		shedTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "nexus",
				Subsystem: "resilience",
				Name:      "requests_shed_total",
				Help:      "Requests shed by priority level.",
			},
			[]string{"priority", "reason"},
		),
		pressureGauge: promauto.NewGauge(prometheus.GaugeOpts{
			Namespace: "nexus",
			Subsystem: "resilience",
			Name:      "system_pressure",
			Help:      "Current system pressure (0-1).",
		}),
		activeLevels: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "nexus",
				Subsystem: "resilience",
				Name:      "shed_level_active",
				Help:      "Whether each shed level is active (0/1).",
			},
			[]string{"level"},
		),
	}
}

// UpdatePressure computes system pressure from current metrics.
func (ls *LoadShedder) UpdatePressure(queueDepth int, p95LatencyMs float64, gpuUtil float64) {
	ls.mu.Lock()
	defer ls.mu.Unlock()

	// Weighted pressure: queue 40%, latency 35%, GPU 25%
	queuePressure := math.Min(float64(queueDepth)/float64(ls.config.QueueCapacity), 1.0)
	latencyPressure := math.Min(p95LatencyMs/ls.config.MaxLatencyMs, 1.0)
	gpuPressure := math.Min(gpuUtil/ls.config.MaxGPUUtil, 1.0)

	instant := queuePressure*0.40 + latencyPressure*0.35 + gpuPressure*0.25

	// Add to sliding window
	ls.window = append(ls.window, pressureSample{
		timestamp: time.Now(),
		value:     instant,
	})
	if len(ls.window) > ls.config.WindowSize {
		ls.window = ls.window[1:]
	}

	// Exponentially weighted moving average
	alpha := 0.3
	if ls.pressure == 0 {
		ls.pressure = instant
	} else {
		ls.pressure = alpha*instant + (1-alpha)*ls.pressure
	}

	ls.pressureGauge.Set(ls.pressure)

	// Update active level indicators
	if ls.pressure >= ls.config.Level3Threshold {
		ls.activeLevels.WithLabelValues("3").Set(1)
	} else {
		ls.activeLevels.WithLabelValues("3").Set(0)
	}
	if ls.pressure >= ls.config.Level2Threshold {
		ls.activeLevels.WithLabelValues("2").Set(1)
	} else {
		ls.activeLevels.WithLabelValues("2").Set(0)
	}
	if ls.pressure >= ls.config.Level1Threshold {
		ls.activeLevels.WithLabelValues("1").Set(1)
	} else {
		ls.activeLevels.WithLabelValues("1").Set(0)
	}
}

// ShouldShed returns true if a request at the given priority should be dropped.
// priority: 0=low, 1=normal, 2=high, 3=critical (never shed)
func (ls *LoadShedder) ShouldShed(priority int) bool {
	ls.mu.RLock()
	pressure := ls.pressure
	ls.mu.RUnlock()

	switch {
	case priority >= 3: // Critical -- never shed
		return false
	case priority == 2 && pressure >= ls.config.Level3Threshold:
		ls.shedTotal.WithLabelValues("high", "level3").Inc()
		return true
	case priority == 1 && pressure >= ls.config.Level2Threshold:
		ls.shedTotal.WithLabelValues("normal", "level2").Inc()
		return true
	case priority == 0 && pressure >= ls.config.Level1Threshold:
		ls.shedTotal.WithLabelValues("low", "level1").Inc()
		return true
	}
	return false
}

// GetPressure returns current system pressure (0-1).
func (ls *LoadShedder) GetPressure() float64 {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.pressure
}

// --- Model Rollback Detector ---

// AccuracyDriftDetector monitors model accuracy and triggers rollback
// when output quality degrades beyond acceptable thresholds.
type AccuracyDriftDetector struct {
	mu       sync.RWMutex
	models   map[string]*modelAccuracy
	logger   *zap.Logger

	// Callbacks
	onDrift   func(modelID string, current, baseline float64)
	onRollback func(modelID, rollbackVersion string)

	// Metrics
	driftScore    *prometheus.GaugeVec
	rollbackCount *prometheus.CounterVec
}

type modelAccuracy struct {
	ModelID         string
	CurrentVersion  string
	PreviousVersion string
	Baseline        float64   // expected accuracy
	Current         float64   // recent accuracy
	Samples         []float64 // recent accuracy samples
	WindowSize      int
	DriftThreshold  float64   // trigger rollback if accuracy drops by this much
	LastCheck       time.Time
}

// NewAccuracyDriftDetector creates a model rollback detector.
func NewAccuracyDriftDetector(logger *zap.Logger) *AccuracyDriftDetector {
	return &AccuracyDriftDetector{
		models: make(map[string]*modelAccuracy),
		logger: logger.Named("drift-detector"),
		driftScore: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "nexus",
				Subsystem: "model",
				Name:      "accuracy_drift",
				Help:      "Accuracy drift from baseline (negative = degraded).",
			},
			[]string{"model"},
		),
		rollbackCount: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "nexus",
				Subsystem: "model",
				Name:      "rollbacks_total",
				Help:      "Number of automatic model rollbacks.",
			},
			[]string{"model"},
		),
	}
}

// RegisterModel sets up monitoring for a model.
func (d *AccuracyDriftDetector) RegisterModel(modelID, version string, baseline, threshold float64) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.models[modelID] = &modelAccuracy{
		ModelID:        modelID,
		CurrentVersion: version,
		Baseline:       baseline,
		Current:        baseline,
		WindowSize:     100,
		DriftThreshold: threshold,
		Samples:        make([]float64, 0, 100),
	}

	d.logger.Info("registered model for drift detection",
		zap.String("model", modelID),
		zap.Float64("baseline", baseline),
		zap.Float64("threshold", threshold),
	)
}

// RecordAccuracy adds an accuracy sample for a model.
func (d *AccuracyDriftDetector) RecordAccuracy(modelID string, accuracy float64) {
	d.mu.Lock()
	defer d.mu.Unlock()

	m, ok := d.models[modelID]
	if !ok {
		return
	}

	m.Samples = append(m.Samples, accuracy)
	if len(m.Samples) > m.WindowSize {
		m.Samples = m.Samples[1:]
	}

	// Update running average
	sum := 0.0
	for _, s := range m.Samples {
		sum += s
	}
	m.Current = sum / float64(len(m.Samples))
	m.LastCheck = time.Now()

	drift := m.Current - m.Baseline
	d.driftScore.WithLabelValues(modelID).Set(drift)

	// Check for drift
	if len(m.Samples) >= m.WindowSize/2 && drift < -m.DriftThreshold {
		d.logger.Error("accuracy drift detected",
			zap.String("model", modelID),
			zap.Float64("baseline", m.Baseline),
			zap.Float64("current", m.Current),
			zap.Float64("drift", drift),
		)

		if d.onDrift != nil {
			go d.onDrift(modelID, m.Current, m.Baseline)
		}

		// Trigger automatic rollback if previous version exists
		if m.PreviousVersion != "" {
			d.rollbackCount.WithLabelValues(modelID).Inc()
			if d.onRollback != nil {
				go d.onRollback(modelID, m.PreviousVersion)
			}
		}
	}
}

// OnDrift sets the callback for accuracy drift detection.
func (d *AccuracyDriftDetector) OnDrift(fn func(modelID string, current, baseline float64)) {
	d.onDrift = fn
}

// OnRollback sets the callback for automatic model rollback.
func (d *AccuracyDriftDetector) OnRollback(fn func(modelID, rollbackVersion string)) {
	d.onRollback = fn
}

// GetStatus returns drift status for all models.
func (d *AccuracyDriftDetector) GetStatus() map[string]map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	status := make(map[string]map[string]interface{})
	for id, m := range d.models {
		status[id] = map[string]interface{}{
			"version":   m.CurrentVersion,
			"baseline":  m.Baseline,
			"current":   m.Current,
			"drift":     m.Current - m.Baseline,
			"samples":   len(m.Samples),
			"threshold": m.DriftThreshold,
			"last_check": m.LastCheck,
		}
	}
	return status
}
