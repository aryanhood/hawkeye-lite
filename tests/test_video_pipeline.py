"""End-to-end tests for Video-In -> Video-Out pipeline, Benchmark, Export, and CLI."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from evaluation.benchmark import run_benchmark, run_single_benchmark_case
from evaluation.cli import main
from evaluation.export import export_all
from evaluation.predict_video import process_video
from evaluation.synthetic_video import RallyScenarioSpec, generate_synthetic_rally_video


def test_end_to_end_video_in_to_video_out_and_cli(tmp_path: Path) -> None:
    in_mp4 = tmp_path / "input_rally.mp4"
    out_mp4 = tmp_path / "output_hawkeye.mp4"
    report_json = tmp_path / "report.json"

    spec = RallyScenarioSpec(
        name="fast_test_clip",
        description="Compact multi-shot clip for fast CI testing",
        pos0_m=(22.2, 3.5, 1.15),
        vel0_m_s=(-25.0, 7.8, 2.3),
        spin_rad_s=(0.0, 250.0, 0.0),
        num_frames=24,
        fps=30.0,
        include_cutaway_frames=4,
        include_angle_switch=True,
        pan_amplitude_m=0.15,
    )
    meta = generate_synthetic_rally_video(output_path=in_mp4, scenario=spec, width=640, height=360)
    assert in_mp4.exists()
    assert meta["num_frames"] == 24

    code = main(
        [
            "predict",
            "--input_video_path",
            str(in_mp4),
            "--output_video_path",
            str(out_mp4),
            "--minimap",
            "1",
            "--bounce",
            "1",
            "--report",
            str(report_json),
        ]
    )
    assert code == 0
    assert out_mp4.exists()
    assert report_json.exists()

    # Read back output .mp4 with cv2.VideoCapture and verify frames are valid
    cap = cv2.VideoCapture(str(out_mp4))
    assert cap.isOpened()
    out_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert out_count > 24  # includes appended 3D Hawk-Eye replay frames
    ret, sample_frame = cap.read()
    cap.release()
    assert ret
    assert sample_frame is not None
    assert sample_frame.shape == (360, 640, 3)

    with pytest.raises(FileNotFoundError):
        process_video(tmp_path / "nonexistent.mp4", out_mp4)


def test_cli_synth_benchmark_and_export(tmp_path: Path) -> None:
    synth_out = tmp_path / "synth_cli.mp4"
    assert main(["synth", "--scenario", "ao_sideline_in", "--output", str(synth_out)]) == 0
    assert synth_out.exists()

    assert main(["benchmark", "--seeds", "5"]) == 0

    case = run_single_benchmark_case(seed=1, camera_mode="panning_broadcast")
    assert case.err_dlt_4cam_mm < case.err_static_1cam_mm

    rep = run_benchmark(num_seeds=8)
    assert rep.total_cases == 24
    assert rep.dlt_2cam.bounce_error_median_mm < rep.static_1cam.bounce_error_median_mm
    assert rep.dlt_4cam.bounce_error_median_mm <= rep.dlt_2cam.bounce_error_median_mm

    assert export_all(repo_root=tmp_path, check_only=False)
    assert export_all(repo_root=tmp_path, check_only=True)

    # CLI `export` and `export --check` subcommands
    assert main(["export"]) == 0
    assert main(["export", "--check"]) == 0

    # No command prints help and returns 1
    assert main([]) == 1
