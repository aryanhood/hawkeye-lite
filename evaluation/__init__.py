"""Evaluation, Video-In -> Video-Out Pipeline, Synthetic Generator, Benchmark, and CLI package."""

from evaluation.benchmark import (
    CAMERA_MODES,
    BenchmarkReport,
    MethodSummary,
    ScenarioCaseResult,
    run_benchmark,
    run_single_benchmark_case,
    seeded_rng,
)
from evaluation.cli import build_parser, main
from evaluation.export import build_web_studio_payload, export_all, render_benchmark_markdown
from evaluation.predict_video import (
    BounceEventReport,
    VideoPredictionResult,
    process_video,
    render_court_replay_frame,
)
from evaluation.synthetic_video import (
    SCENARIO_PRESETS,
    RallyScenarioSpec,
    generate_synthetic_rally_video,
    render_court_3d_frame,
    render_cutaway_frame,
)

__all__ = [
    "CAMERA_MODES",
    "SCENARIO_PRESETS",
    "BenchmarkReport",
    "BounceEventReport",
    "MethodSummary",
    "RallyScenarioSpec",
    "ScenarioCaseResult",
    "VideoPredictionResult",
    "build_parser",
    "build_web_studio_payload",
    "export_all",
    "generate_synthetic_rally_video",
    "main",
    "process_video",
    "render_benchmark_markdown",
    "render_court_3d_frame",
    "render_court_replay_frame",
    "render_cutaway_frame",
    "run_benchmark",
    "run_single_benchmark_case",
    "seeded_rng",
]
