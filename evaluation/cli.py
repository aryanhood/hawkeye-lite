"""Command-line interface for `hawkeye-lite`.

Supports:
1. `hawkeye-lite predict --input_video_path ... --output_video_path ... --minimap 1 --bounce 1`
2. `hawkeye-lite synth --scenario ao_sideline_in --output samples/input.mp4`
3. `hawkeye-lite benchmark`
4. `hawkeye-lite export [--check]`
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from evaluation.benchmark import run_benchmark
from evaluation.export import export_all
from evaluation.predict_video import process_video
from evaluation.synthetic_video import SCENARIO_PRESETS, generate_synthetic_rally_video


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="hawkeye-lite",
        description="Tennis Court Calibration, Multi-View DLT Triangulation & Line Calling Engine",
    )
    sub = parser.add_subparsers(dest="command")

    # 1. `predict` subcommand (ArtLabss flag compatible)
    p_pred = sub.add_parser("predict", help="Process an input tennis video (.mp4) into a Hawk-Eye output video (.mp4)")
    p_pred.add_argument("--input_video_path", "--input", "-i", required=True, help="Path to input .mp4 video")
    p_pred.add_argument("--output_video_path", "--output", "-o", required=True, help="Path to output .mp4 video")
    p_pred.add_argument(
        "--minimap", type=int, default=1, choices=[0, 1], help="Embed top-right 2D court minimap (1=on)"
    )
    p_pred.add_argument(
        "--bounce",
        type=int,
        default=1,
        choices=[0, 1],
        help="Detect bounces, call IN/OUT in mm, and append 3D Hawk-Eye replay (1=on)",
    )
    p_pred.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of frames to process")
    p_pred.add_argument("--report", type=str, default=None, help="Optional path to save JSON telemetry report")

    # 2. `synth` subcommand
    p_synth = sub.add_parser("synth", help="Generate a realistic synthetic tennis rally video (.mp4)")
    p_synth.add_argument(
        "--scenario",
        type=str,
        default="ao_sideline_in",
        choices=list(SCENARIO_PRESETS.keys()),
        help="Preset scenario name",
    )
    p_synth.add_argument("--output", "-o", required=True, help="Output .mp4 file path")

    # 3. `benchmark` subcommand
    p_bench = sub.add_parser("benchmark", help="Run the 300-scenario paired ablation benchmark")
    p_bench.add_argument("--seeds", type=int, default=100, help="Number of deterministic seeds (default: 100)")

    # 4. `export` subcommand
    p_exp = sub.add_parser("export", help="Export docs/benchmark.md and web/public/data/hawkeye_data.json")
    p_exp.add_argument("--check", action="store_true", help="Verify exported files match code without writing")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning exit code ``0`` on success."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        result = process_video(
            input_video_path=args.input_video_path,
            output_video_path=args.output_video_path,
            minimap=int(args.minimap),
            bounce=int(args.bounce),
            max_frames=args.max_frames,
        )
        if args.report:
            rep_path = Path(args.report)
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        print(
            f"[hawkeye-lite] Processed {result.total_input_frames} input frames "
            f"({result.active_court_frames} court, {result.cutaway_frames} cutaway, "
            f"{result.num_camera_shots} shots) -> {result.output_video_path} "
            f"({result.total_output_frames} output frames, {len(result.bounces)} bounces)."
        )
        for b in result.bounces:
            print(
                f"  Bounce @ frame {b.frame_index}: ({b.court_xy_m[0]:.2f}m, {b.court_xy_m[1]:.2f}m) -> "
                f"{b.decision.verdict} by {abs(b.decision.margin_mm):.1f} mm (+/- {b.decision.sigma_mm:.1f} mm) "
                f"on {b.decision.nearest_line_name}"
            )
        return 0

    if args.command == "synth":
        meta = generate_synthetic_rally_video(output_path=args.output, scenario=args.scenario)
        print(
            f"[hawkeye-lite] Generated synthetic video ({meta['scenario']}): {meta['output_path']} "
            f"({meta['num_frames']} frames, ground-truth bounce @ frame {meta['bounce_frame_gt']})"
        )
        return 0

    if args.command == "benchmark":
        report = run_benchmark(num_seeds=int(args.seeds))
        print(f"=== Hawk-Eye Lite 2D-Pixel-Only Paired Benchmark ({report.total_cases} scenarios) ===")
        for summary in (report.static_1cam, report.dynamic_1cam, report.dlt_2cam, report.dlt_4cam):
            print(
                f"- {summary.name}: median={summary.bounce_error_median_mm:.2f} mm, "
                f"p90={summary.bounce_error_p90_mm:.2f} mm, mean={summary.bounce_error_mean_mm:.2f} mm | "
                f"Call Acc={summary.line_call_accuracy_pct:.2f}% (Tight +-35mm: {summary.tight_call_accuracy_pct:.2f}%)"
            )
        print(
            f"- Skid ablation (2-Cam DLT): off={report.skid_ablation_2cam_no_skid_tight_pct:.2f}% vs "
            f"on={report.dlt_2cam.tight_call_accuracy_pct:.2f}% tight accuracy"
        )
        return 0

    if args.command == "export":
        repo_root = Path(__file__).resolve().parent.parent
        ok = export_all(repo_root=repo_root, check_only=bool(args.check))
        if args.check:
            if not ok:
                print(
                    "[hawkeye-lite] ERROR: Exported benchmark/web data is out of date. Run `hawkeye-lite export`.",
                    file=sys.stderr,
                )
                return 1
            print(
                "[hawkeye-lite] Export check passed: docs/benchmark.md and "
                "web/public/data/hawkeye_data.json are up to date."
            )
            return 0
        print("[hawkeye-lite] Exported docs/benchmark.md and web/public/data/hawkeye_data.json.")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
