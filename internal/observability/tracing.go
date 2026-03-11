package observability

import (
	"context"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// Span represents a distributed trace span.
type Span struct {
	TraceID    string
	SpanID     string
	ParentID   string
	Operation  string
	Service    string
	StartTime  time.Time
	EndTime    time.Time
	Duration   time.Duration
	Status     SpanStatus
	Attributes map[string]string
	Events     []SpanEvent
}

type SpanStatus int

const (
	SpanOK SpanStatus = iota
	SpanError
)

type SpanEvent struct {
	Name       string
	Timestamp  time.Time
	Attributes map[string]string
}

// Tracer provides distributed tracing for the inference pipeline.
type Tracer struct {
	serviceName string
	logger      *zap.Logger
	exporter    SpanExporter

	// Metrics about tracing itself
	spansCreated prometheus.Counter
	spansExported prometheus.Counter
	spanErrors   prometheus.Counter
}

// SpanExporter sends completed spans to a tracing backend.
type SpanExporter interface {
	Export(ctx context.Context, spans []Span) error
	Shutdown(ctx context.Context) error
}

// OTLPExporter exports spans in OpenTelemetry format.
type OTLPExporter struct {
	endpoint string
	batch    []Span
	batchMax int
	flushInt time.Duration
	stopCh   chan struct{}
}

func NewOTLPExporter(endpoint string) *OTLPExporter {
	e := &OTLPExporter{
		endpoint: endpoint,
		batchMax: 512,
		flushInt: 5 * time.Second,
		stopCh:   make(chan struct{}),
	}
	go e.flushLoop()
	return e
}

func (e *OTLPExporter) Export(ctx context.Context, spans []Span) error {
	e.batch = append(e.batch, spans...)
	if len(e.batch) >= e.batchMax {
		return e.flush(ctx)
	}
	return nil
}

func (e *OTLPExporter) flush(ctx context.Context) error {
	if len(e.batch) == 0 {
		return nil
	}
	// In production: serialize to OTLP protobuf and POST to endpoint
	// For now: clear batch (actual HTTP export would go here)
	e.batch = e.batch[:0]
	return nil
}

func (e *OTLPExporter) flushLoop() {
	ticker := time.NewTicker(e.flushInt)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			e.flush(context.Background())
		case <-e.stopCh:
			e.flush(context.Background())
			return
		}
	}
}

func (e *OTLPExporter) Shutdown(ctx context.Context) error {
	close(e.stopCh)
	return e.flush(ctx)
}

// NewTracer creates a distributed tracer.
func NewTracer(serviceName string, exporter SpanExporter, logger *zap.Logger) *Tracer {
	return &Tracer{
		serviceName: serviceName,
		logger:      logger.Named("tracer"),
		exporter:    exporter,
		spansCreated: promauto.NewCounter(prometheus.CounterOpts{
			Namespace: "nexus", Subsystem: "tracing", Name: "spans_created_total",
		}),
		spansExported: promauto.NewCounter(prometheus.CounterOpts{
			Namespace: "nexus", Subsystem: "tracing", Name: "spans_exported_total",
		}),
		spanErrors: promauto.NewCounter(prometheus.CounterOpts{
			Namespace: "nexus", Subsystem: "tracing", Name: "span_errors_total",
		}),
	}
}

// StartSpan begins a new trace span.
func (t *Tracer) StartSpan(ctx context.Context, operation string) (*ActiveSpan, context.Context) {
	span := &Span{
		TraceID:    generateID(),
		SpanID:     generateID(),
		Operation:  operation,
		Service:    t.serviceName,
		StartTime:  time.Now(),
		Status:     SpanOK,
		Attributes: make(map[string]string),
	}

	// Inherit trace context from parent
	if parent := spanFromContext(ctx); parent != nil {
		span.TraceID = parent.span.TraceID
		span.ParentID = parent.span.SpanID
	}

	t.spansCreated.Inc()

	active := &ActiveSpan{span: span, tracer: t}
	newCtx := context.WithValue(ctx, spanContextKey{}, active)
	return active, newCtx
}

// ActiveSpan is an in-progress span.
type ActiveSpan struct {
	span   *Span
	tracer *Tracer
}

func (s *ActiveSpan) SetAttribute(key, value string) {
	s.span.Attributes[key] = value
}

func (s *ActiveSpan) AddEvent(name string, attrs map[string]string) {
	s.span.Events = append(s.span.Events, SpanEvent{
		Name: name, Timestamp: time.Now(), Attributes: attrs,
	})
}

func (s *ActiveSpan) SetError(err error) {
	s.span.Status = SpanError
	s.span.Attributes["error"] = err.Error()
	s.tracer.spanErrors.Inc()
}

func (s *ActiveSpan) End() {
	s.span.EndTime = time.Now()
	s.span.Duration = s.span.EndTime.Sub(s.span.StartTime)

	if err := s.tracer.exporter.Export(context.Background(), []Span{*s.span}); err != nil {
		s.tracer.logger.Warn("failed to export span", zap.Error(err))
	} else {
		s.tracer.spansExported.Inc()
	}
}

type spanContextKey struct{}

func spanFromContext(ctx context.Context) *ActiveSpan {
	val := ctx.Value(spanContextKey{})
	if val == nil {
		return nil
	}
	return val.(*ActiveSpan)
}

var idCounter uint64

func generateID() string {
	idCounter++
	return fmt.Sprintf("%016x", idCounter)
}
