import { useEffect, useState } from "react";
import type { HawkeyePayload, ReplayScenario } from "./types";

const FALLBACK_PAYLOAD: HawkeyePayload = {
  version: "0.1.0",
  court: {
    length_m: 23.77,
    doubles_width_m: 10.97,
    world_coordinates_3d: [
      [0, 0, 0],
      [0, 1.37, 0],
      [0, 5.485, 0],
      [0, 9.6, 0],
      [0, 10.97, 0],
      [5.485, 1.37, 0],
      [5.485, 5.485, 0],
      [5.485, 9.6, 0],
      [11.885, 0, 0],
      [11.885, 1.37, 0],
      [11.885, 5.485, 0],
      [11.885, 9.6, 0],
      [11.885, 10.97, 0],
      [18.285, 1.37, 0],
      [18.285, 5.485, 0],
      [18.285, 9.6, 0],
      [23.77, 0, 0],
      [23.77, 1.37, 0],
      [23.77, 5.485, 0],
      [23.77, 9.6, 0],
      [23.77, 10.97, 0],
      [11.885, 0, 1.2],
      [11.885, 1.37, 1.2],
      [11.885, 5.485, 1.2],
      [11.885, 9.6, 1.2],
      [11.885, 10.97, 1.2],
    ],
    wireframe_groups: [
      [0, 4, 20, 16, 0],
      [1, 3, 19, 17, 1],
      [5, 7, 15, 13, 5],
      [6, 14],
      [8, 21, 25, 12],
    ],
    configurations_12: {
      "1": [
        [286, 561],
        [1379, 561],
        [286, 2935],
        [1379, 2935],
      ],
      "2": [
        [423, 561],
        [1242, 561],
        [423, 2935],
        [1242, 2935],
      ],
    },
  },
  benchmark: {
    total_cases: 300,
    tight_cases_count: 242,
    tight_in_rate_pct: 49.59,
    static_1cam: {
      name: "1-Cam Static Frame-0 Homography",
      bounce_error_median_mm: 3625.14,
      bounce_error_p90_mm: 7259.74,
      bounce_error_mean_mm: 3990.64,
      line_call_accuracy_pct: 51.33,
      tight_call_accuracy_pct: 53.72,
      brier_score: null,
      ci95_coverage_pct: null,
    },
    dynamic_1cam: {
      name: "1-Cam Dynamic Homography + 2D Outlier Interp",
      bounce_error_median_mm: 2708.21,
      bounce_error_p90_mm: 8029.8,
      bounce_error_mean_mm: 3260.23,
      line_call_accuracy_pct: 53.33,
      tight_call_accuracy_pct: 54.96,
      brier_score: null,
      ci95_coverage_pct: null,
    },
    dlt_2cam: {
      name: "2-Cam DLT 3D + KNN + Sub-Frame Ballistic Fit",
      bounce_error_median_mm: 52.64,
      bounce_error_p90_mm: 146.97,
      bounce_error_mean_mm: 82.58,
      line_call_accuracy_pct: 79.33,
      tight_call_accuracy_pct: 76.86,
      brier_score: 0.1568,
      ci95_coverage_pct: 76.33,
    },
    dlt_4cam: {
      name: "4-Cam Stadium DLT 3D + KNN + Sub-Frame Ballistic Fit",
      bounce_error_median_mm: 31.27,
      bounce_error_p90_mm: 87.62,
      bounce_error_mean_mm: 58.87,
      line_call_accuracy_pct: 83.33,
      tight_call_accuracy_pct: 80.17,
      brier_score: 0.1149,
      ci95_coverage_pct: 84.67,
    },
    skid_ablation_2cam_no_skid_acc_pct: 73.33,
    skid_ablation_2cam_no_skid_tight_pct: 70.25,
    mean_error_reduction_2cam_vs_1cam_mm: 3177.65,
    mean_error_reduction_4cam_vs_2cam_mm: 23.71,
  },
  replays: [
    {
      id: "ao_sideline_in",
      name: "ao_sideline_in",
      description:
        "Australian Open crosscourt forehand near the right singles sideline (2-Cam DLT reconstruction)",
      fps: 30,
      cutaway_frames: 0,
      has_angle_switch: false,
      bounce_index: 26,
      bounce_x_m: 4.12,
      bounce_y_m: 9.588,
      verdict: "IN",
      is_in: true,
      margin_mm: 18.4,
      sigma_mm: 24.5,
      nearest_line: "right_singles_sideline",
      raw_points_3d: [
        [22.4, 3.2, 1.15],
        [14.2, 6.1, 1.48],
        [4.12, 9.588, 0.034],
        [1.1, 10.6, 0.78],
      ],
      knn_points_3d: [
        [22.4, 3.2, 1.15],
        [14.2, 6.1, 1.48],
        [4.12, 9.588, 0.034],
        [1.1, 10.6, 0.78],
      ],
      fitted_curve_3d: [
        [22.4, 3.2, 1.15],
        [14.2, 6.1, 1.48],
        [4.12, 9.588, 0.034],
        [1.1, 10.6, 0.78],
      ],
      ground_truth_3d: [
        [22.4, 3.2, 1.15],
        [14.2, 6.1, 1.48],
        [4.12, 9.588, 0.034],
        [1.1, 10.6, 0.78],
      ],
    },
  ],
};

function projectIsometric(
  pt3d: number[],
  yawDeg: number,
  pitchDeg: number,
  width: number,
  height: number,
): [number, number] {
  const cx = 11.885;
  const cy = 5.485;
  const x = pt3d[0] - cx;
  const y = pt3d[1] - cy;
  const z = pt3d[2];

  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;

  const xr = x * Math.cos(yaw) - y * Math.sin(yaw);
  const yr = x * Math.sin(yaw) + y * Math.cos(yaw);

  const u = width * 0.5 + xr * 21.5;
  const v = height * 0.56 + yr * Math.sin(pitch) * 21.5 - z * Math.cos(pitch) * 38.0;
  return [u, v];
}

export function App() {
  const [data, setData] = useState<HawkeyePayload>(FALLBACK_PAYLOAD);
  const [activeTab, setActiveTab] = useState<"replay" | "configs" | "benchmark">("replay");
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>("ao_sideline_in");
  const [stageMode, setStageMode] = useState<"raw" | "knn" | "fitted">("fitted");
  const [yawDeg, setYawDeg] = useState<number>(32);
  const [selectedConf, setSelectedConf] = useState<string>("1");

  useEffect(() => {
    fetch("/data/hawkeye_data.json")
      .then((res) => (res.ok ? res.json() : null))
      .then((json: HawkeyePayload | null) => {
        if (json?.replays && json.replays.length > 0) {
          setData(json);
          setSelectedScenarioId(json.replays[0].id);
        }
      })
      .catch(() => {
        // Fallback payload already loaded for offline/test environments
      });
  }, []);

  const scenario: ReplayScenario =
    data.replays.find((r) => r.id === selectedScenarioId) ?? data.replays[0];

  const activePoints3d =
    stageMode === "raw"
      ? scenario.raw_points_3d
      : stageMode === "knn"
        ? scenario.knn_points_3d
        : scenario.fitted_curve_3d;

  return (
    <div className="studio-shell">
      <header className="studio-header">
        <div>
          <h1 className="brand-title">
            Hawk-Eye Lite — Multi-View 3D Trajectory &amp; Line-Call Studio
          </h1>
          <p className="brand-subtitle">
            Video Input (.mp4) → 12-Config CourtDetector + Minimap + Multi-View DLT 3D Trajectory
            Reconstruction
          </p>
        </div>
        <nav className="tab-bar" aria-label="Studio navigation">
          <button
            type="button"
            className={`tab-btn ${activeTab === "replay" ? "active" : ""}`}
            onClick={() => setActiveTab("replay")}
          >
            3D Reconstruction & Minimap
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "configs" ? "active" : ""}`}
            onClick={() => setActiveTab("configs")}
          >
            12-Config Court Calibration
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "benchmark" ? "active" : ""}`}
            onClick={() => setActiveTab("benchmark")}
          >
            300-Scenario Benchmark
          </button>
        </nav>
      </header>

      {activeTab === "replay" && (
        <div className="grid-2">
          <section className="card">
            <div className="card-title">
              <span>SEU 3D Wireframe Court & Trajectory Reconstruction</span>
              <span className={`badge ${scenario.is_in ? "badge-in" : "badge-out"}`}>
                {scenario.verdict} BY {Math.abs(scenario.margin_mm).toFixed(1)} mm (±
                {scenario.sigma_mm.toFixed(1)} mm)
              </span>
            </div>

            <div className="controls-row">
              <label htmlFor="scenario-select">Rally:</label>
              <select
                id="scenario-select"
                className="select-input"
                value={scenario.id}
                onChange={(e) => setSelectedScenarioId(e.target.value)}
              >
                {data.replays.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name} ({r.verdict})
                  </option>
                ))}
              </select>

              <button
                type="button"
                className={`btn-small ${stageMode === "raw" ? "active" : ""}`}
                onClick={() => setStageMode("raw")}
              >
                1. Raw Noisy 3D
              </button>
              <button
                type="button"
                className={`btn-small ${stageMode === "knn" ? "active" : ""}`}
                onClick={() => setStageMode("knn")}
              >
                2. Multi-Plane KNN
              </button>
              <button
                type="button"
                className={`btn-small ${stageMode === "fitted" ? "active" : ""}`}
                onClick={() => setStageMode("fitted")}
              >
                3. Constrained 3D Curve
              </button>

              <label htmlFor="yaw-slider">Orbit:</label>
              <input
                id="yaw-slider"
                type="range"
                min={-65}
                max={65}
                value={yawDeg}
                onChange={(e) => setYawDeg(Number(e.target.value))}
              />
            </div>

            <svg viewBox="0 0 680 380" className="svg-viewport" role="img" aria-label="3D Court">
              {data.court.wireframe_groups.map((grp, gIdx) => {
                const polyPts = grp
                  .map((ptIdx) => {
                    const coord = data.court.world_coordinates_3d[ptIdx] ?? [0, 0, 0];
                    const [u, v] = projectIsometric(coord, yawDeg, 32, 680, 380);
                    return `${u.toFixed(1)},${v.toFixed(1)}`;
                  })
                  .join(" ");
                return (
                  <polyline
                    key={`wf-${gIdx.toString()}`}
                    points={polyPts}
                    fill="none"
                    stroke={gIdx === 4 ? "#38bdf8" : "#94a3b8"}
                    strokeWidth={gIdx === 4 ? 2.5 : 1.8}
                  />
                );
              })}

              {stageMode === "fitted" && activePoints3d.length > 1 && (
                <polyline
                  points={activePoints3d
                    .map((pt) => {
                      const [u, v] = projectIsometric(pt, yawDeg, 32, 680, 380);
                      return `${u.toFixed(1)},${v.toFixed(1)}`;
                    })
                    .join(" ")}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth={3}
                />
              )}

              {activePoints3d.map((pt, idx) => {
                const [u, v] = projectIsometric(pt, yawDeg, 32, 680, 380);
                return (
                  <circle
                    key={`pt-${idx.toString()}`}
                    cx={u}
                    cy={v}
                    r={stageMode === "raw" ? 3.5 : 3}
                    fill={stageMode === "raw" ? "#f87171" : "#38bdf8"}
                  />
                );
              })}

              {(() => {
                const [bu, bv] = projectIsometric(
                  [scenario.bounce_x_m, scenario.bounce_y_m, 0.0335],
                  yawDeg,
                  32,
                  680,
                  380,
                );
                return (
                  <g>
                    <circle
                      cx={bu}
                      cy={bv}
                      r={7}
                      fill={scenario.is_in ? "#22c55e" : "#ef4444"}
                      stroke="#ffffff"
                      strokeWidth={2}
                    />
                  </g>
                );
              })()}
            </svg>
            <p className="brand-subtitle" style={{ marginTop: 10 }}>
              {scenario.description}
            </p>
          </section>

          <section className="card">
            <div className="card-title">
              <span>2D Top-Down Court Minimap & Oblique Skid Footprint</span>
              <span className="badge badge-in">{scenario.nearest_line}</span>
            </div>
            <svg viewBox="0 0 320 420" className="svg-viewport" role="img" aria-label="2D Minimap">
              <rect
                x={40}
                y={24}
                width={240}
                height={372}
                fill="#1d4ed8"
                stroke="#e2e8f0"
                strokeWidth={2}
              />
              <rect
                x={70}
                y={24}
                width={180}
                height={372}
                fill="#1e3a8a"
                stroke="#ffffff"
                strokeWidth={2}
              />
              <line x1={70} y1={110} x2={250} y2={110} stroke="#ffffff" strokeWidth={2} />
              <line x1={70} y1={310} x2={250} y2={310} stroke="#ffffff" strokeWidth={2} />
              <line x1={160} y1={110} x2={160} y2={310} stroke="#ffffff" strokeWidth={2} />
              <line x1={40} y1={210} x2={280} y2={210} stroke="#38bdf8" strokeWidth={3} />

              {scenario.fitted_curve_3d.map((pt, idx) => {
                const mx = 40 + (pt[1] / 10.97) * 240;
                const my = 24 + (pt[0] / 23.77) * 372;
                return (
                  <circle key={`m-${idx.toString()}`} cx={mx} cy={my} r={2.8} fill="#fde047" />
                );
              })}

              {(() => {
                const bx = 40 + (scenario.bounce_y_m / 10.97) * 240;
                const by = 24 + (scenario.bounce_x_m / 23.77) * 372;
                return (
                  <ellipse
                    cx={bx}
                    cy={by}
                    rx={9}
                    ry={6}
                    fill={scenario.is_in ? "#22c55e" : "#ef4444"}
                    stroke="#ffffff"
                    strokeWidth={2}
                  />
                );
              })()}
            </svg>
            <table className="metric-table">
              <tbody>
                <tr>
                  <th>3D Bounce Coordinates</th>
                  <td>
                    X = {scenario.bounce_x_m.toFixed(3)} m, Y = {scenario.bounce_y_m.toFixed(3)} m
                  </td>
                </tr>
                <tr>
                  <th>Line Call Verdict</th>
                  <td>
                    <strong>{scenario.verdict}</strong> ({scenario.margin_mm.toFixed(1)} mm ±{" "}
                    {scenario.sigma_mm.toFixed(1)} mm)
                  </td>
                </tr>
                <tr>
                  <th>Broadcast Camera Handling</th>
                  <td>
                    {scenario.cutaway_frames > 0
                      ? `${scenario.cutaway_frames} cutaway frames gated + Multi-Angle Cut`
                      : "Dynamic Pan-Compensated H(t)"}
                  </td>
                </tr>
              </tbody>
            </table>
          </section>
        </div>
      )}

      {activeTab === "configs" && (
        <section className="card">
          <div className="card-title">
            <span>
              12 Four-Point Court Configurations (ArtLabss CourtReference + 26 SEU 3D Points)
            </span>
            <span className="badge badge-in">Conf #{selectedConf}</span>
          </div>
          <p className="brand-subtitle">
            When a broadcast camera zooms or pans so only part of the court is visible,{" "}
            <code>CourtDetector._find_homography</code> automatically scores all 12 four-point
            configurations using <code>c_p - 0.5 * w_p</code> to lock the homography.
          </p>
          <div className="conf-grid">
            {Object.entries(data.court.configurations_12).map(([confId, pts]) => (
              <button
                type="button"
                key={confId}
                className={`conf-card ${selectedConf === confId ? "active" : ""}`}
                onClick={() => setSelectedConf(confId)}
              >
                <strong>Configuration #{confId}</strong>
                <div className="brand-subtitle" style={{ fontSize: "0.75rem", marginTop: 4 }}>
                  {pts.map((p) => `(${p[0]},${p[1]})`).join(" · ")}
                </div>
              </button>
            ))}
          </div>
        </section>
      )}

      {activeTab === "benchmark" && (
        <section className="card">
          <div className="card-title">
            <span>
              300-Scenario 2D-Pixel-Only Benchmark ({data.benchmark.total_cases} Cases · Tight IN{" "}
              {data.benchmark.tight_in_rate_pct.toFixed(1)}%)
            </span>
            <span className="badge badge-in">
              -{data.benchmark.mean_error_reduction_2cam_vs_1cam_mm.toFixed(1)} mm (2-Cam vs 1-Cam)
            </span>
          </div>
          <p className="brand-subtitle" style={{ marginBottom: 12 }}>
            Controlled 2-Cam Skid Ablation: center-point only{" "}
            <code>{data.benchmark.skid_ablation_2cam_no_skid_acc_pct.toFixed(2)}%</code> overall /{" "}
            <code>{data.benchmark.skid_ablation_2cam_no_skid_tight_pct.toFixed(2)}%</code> tight vs.
            oblique skid ellipse{" "}
            <code>{data.benchmark.dlt_2cam.line_call_accuracy_pct.toFixed(2)}%</code> overall /{" "}
            <code>{data.benchmark.dlt_2cam.tight_call_accuracy_pct.toFixed(2)}%</code> tight.
          </p>
          <table className="metric-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Median Error</th>
                <th>p90 Error</th>
                <th>Mean Error</th>
                <th>Overall Acc</th>
                <th>Tight Acc (±35 mm)</th>
                <th>Brier</th>
                <th>95% CI Cov</th>
              </tr>
            </thead>
            <tbody>
              {[
                data.benchmark.static_1cam,
                data.benchmark.dynamic_1cam,
                data.benchmark.dlt_2cam,
                data.benchmark.dlt_4cam,
              ].map((m) => (
                <tr key={m.name}>
                  <td>
                    <strong>{m.name}</strong>
                  </td>
                  <td>{m.bounce_error_median_mm.toFixed(2)} mm</td>
                  <td>{m.bounce_error_p90_mm.toFixed(2)} mm</td>
                  <td>{m.bounce_error_mean_mm.toFixed(2)} mm</td>
                  <td>{m.line_call_accuracy_pct.toFixed(2)}%</td>
                  <td>{m.tight_call_accuracy_pct.toFixed(2)}%</td>
                  <td>{m.brier_score != null ? m.brier_score.toFixed(4) : "—"}</td>
                  <td>
                    {m.ci95_coverage_pct != null ? `${m.ci95_coverage_pct.toFixed(2)}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
