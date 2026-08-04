"""
Prometheus metrics exported by the DeepREAP service.

The metric names mirror the dimensions surfaced in the paper's
Monitoring & Feedback Loop section: prediction accuracy, scheduling
latency, throughput, and resource utilization.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- REAP -------------------------------------------------------------
REAP_PREDICTIONS = Counter(
    "deepreap_reap_predictions_total",
    "Number of REAP demand predictions served.",
    ["target"],
)
REAP_LATENCY = Histogram(
    "deepreap_reap_latency_seconds",
    "Latency of a REAP prediction call.",
    ["target"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
REAP_PREDICTED_VALUE = Gauge(
    "deepreap_reap_predicted_value",
    "Most recent REAP prediction (per target).",
    ["target"],
)
REAP_OBSERVED_ERROR = Gauge(
    "deepreap_reap_observed_error",
    "Most recent absolute error reported back to /feedback (per target).",
    ["target"],
)
REAP_ONLINE_UPDATES = Counter(
    "deepreap_reap_online_weight_updates_total",
    "Online ensemble re-weighting updates applied from /feedback.",
    ["target"],
)

# --- Scheduler --------------------------------------------------------
SCHED_DECISIONS = Counter(
    "deepreap_scheduler_decisions_total",
    "Scheduling decisions made.",
    ["action_type"],   # 'schedule' or 'noop'
)
SCHED_LATENCY = Histogram(
    "deepreap_scheduler_latency_seconds",
    "Latency of a scheduling decision.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
SCHED_QUEUE_DEPTH = Gauge(
    "deepreap_scheduler_queue_depth",
    "Number of jobs visible + in backlog at last decision.",
)
SCHED_CLUSTER_UTIL = Gauge(
    "deepreap_cluster_utilization_ratio",
    "Cluster utilization ratio (0..1) at the current timestep.",
    ["resource"],
)

# --- Service ----------------------------------------------------------
SERVICE_INFO = Gauge(
    "deepreap_service_info",
    "Static service info (1 sample); labels carry version.",
    ["version"],
)
HTTP_REQUESTS = Counter(
    "deepreap_http_requests_total",
    "HTTP requests served.",
    ["endpoint", "status"],
)
