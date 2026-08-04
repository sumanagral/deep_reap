"""Data generation and industry-trace loaders for DeepREAP."""

from .production_traces import (
    generate_alibaba_like_trace,
    generate_google_like_trace,
)
from .synthetic_generator import generate_job_trace, generate_resource_usage
from .trace_loaders import load_job_trace

__all__ = [
    "generate_job_trace",
    "generate_resource_usage",
    "generate_google_like_trace",
    "generate_alibaba_like_trace",
    "load_job_trace",
]

# Public dumps → data/real/*.csv via: python -m data.fetch_public_traces
