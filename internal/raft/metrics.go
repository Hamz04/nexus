package raft

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// RaftMetrics holds all Prometheus metrics for a Raft node.
type RaftMetrics struct {
	stateTransitions  *prometheus.CounterVec
	leaderElections   prometheus.Counter
	votesGranted      prometheus.Counter
	proposalsTotal    prometheus.Counter
	commitIndex       prometheus.Gauge
	appliedIndex      prometheus.Gauge
	replicationLag    *prometheus.GaugeVec
	replicationErrors *prometheus.CounterVec
	logSize           prometheus.Gauge
	walWriteLatency   prometheus.Histogram
	rpcLatency        *prometheus.HistogramVec
}

// NewRaftMetrics creates and registers all Raft metrics.
func NewRaftMetrics(nodeID string) *RaftMetrics {
	labels := prometheus.Labels{"node": nodeID}

	return &RaftMetrics{
		stateTransitions: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace:   "nexus",
				Subsystem:   "raft",
				Name:        "state_transitions_total",
				Help:        "Number of state transitions by target state.",
				ConstLabels: labels,
			},
			[]string{"to_state"},
		),

		leaderElections: promauto.NewCounter(prometheus.CounterOpts{
			Namespace:   "nexus",
			Subsystem:   "raft",
			Name:        "leader_elections_total",
			Help:        "Number of successful leader elections.",
			ConstLabels: labels,
		}),

		votesGranted: promauto.NewCounter(prometheus.CounterOpts{
			Namespace:   "nexus",
			Subsystem:   "raft",
			Name:        "votes_granted_total",
			Help:        "Number of votes granted to candidates.",
			ConstLabels: labels,
		}),

		proposalsTotal: promauto.NewCounter(prometheus.CounterOpts{
			Namespace:   "nexus",
			Subsystem:   "raft",
			Name:        "proposals_total",
			Help:        "Total number of proposals submitted.",
			ConstLabels: labels,
		}),

		commitIndex: promauto.NewGauge(prometheus.GaugeOpts{
			Namespace:   "nexus",
			Subsystem:   "raft",
			Name:        "commit_index",
			Help:        "Current commit index.",
			ConstLabels: labels,
		}),

		appliedIndex: promauto.NewGauge(prometheus.GaugeOpts{
			Namespace:   "nexus",
			Subsystem:   "raft",
			Name:        "applied_index",
			Help:        "Last applied index.",
			ConstLabels: labels,
		}),

		replicationLag: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace:   "nexus",
				Subsystem:   "raft",
				Name:        "replication_lag",
				Help:        "Replication lag per peer (entries behind leader).",
				ConstLabels: labels,
			},
			[]string{"peer"},
		),

		replicationErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace:   "nexus",
				Subsystem:   "raft",
				Name:        "replication_errors_total",
				Help:        "Replication RPC failures per peer.",
				ConstLabels: labels,
			},
			[]string{"peer"},
		),

		logSize: promauto.NewGauge(prometheus.GaugeOpts{
			Namespace:   "nexus",
			Subsystem:   "raft",
			Name:        "log_entries",
			Help:        "Current number of log entries.",
			ConstLabels: labels,
		}),

		walWriteLatency: promauto.NewHistogram(prometheus.HistogramOpts{
			Namespace:   "nexus",
			Subsystem:   "raft",
			Name:        "wal_write_duration_seconds",
			Help:        "WAL write latency.",
			ConstLabels: labels,
			Buckets:     prometheus.ExponentialBuckets(0.00001, 2, 15),
		}),

		rpcLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace:   "nexus",
				Subsystem:   "raft",
				Name:        "rpc_duration_seconds",
				Help:        "Raft RPC round-trip latency.",
				ConstLabels: labels,
				Buckets:     prometheus.ExponentialBuckets(0.0001, 2, 14),
			},
			[]string{"rpc_type", "peer"},
		),
	}
}
