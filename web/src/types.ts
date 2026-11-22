export interface MethodSummary {
  name: string;
  bounce_error_median_mm: number;
  bounce_error_p90_mm: number;
  bounce_error_mean_mm: number;
  line_call_accuracy_pct: number;
  tight_call_accuracy_pct: number;
  brier_score?: number | null;
  ci95_coverage_pct?: number | null;
}

export interface ReplayScenario {
  id: string;
  name: string;
  description: string;
  fps: number;
  cutaway_frames: number;
  has_angle_switch: boolean;
  bounce_index: number;
  bounce_x_m: number;
  bounce_y_m: number;
  verdict: "IN" | "OUT";
  is_in: boolean;
  margin_mm: number;
  sigma_mm: number;
  nearest_line: string;
  raw_points_3d: number[][];
  knn_points_3d: number[][];
  fitted_curve_3d: number[][];
  ground_truth_3d: number[][];
}

export interface HawkeyePayload {
  version: string;
  court: {
    length_m: number;
    doubles_width_m: number;
    world_coordinates_3d: number[][];
    wireframe_groups: number[][];
    configurations_12: Record<string, number[][]>;
  };
  benchmark: {
    total_cases: number;
    tight_cases_count: number;
    tight_in_rate_pct: number;
    static_1cam: MethodSummary;
    dynamic_1cam: MethodSummary;
    dlt_2cam: MethodSummary;
    dlt_4cam: MethodSummary;
    skid_ablation_2cam_no_skid_acc_pct: number;
    skid_ablation_2cam_no_skid_tight_pct: number;
    mean_error_reduction_2cam_vs_1cam_mm: number;
    mean_error_reduction_4cam_vs_2cam_mm: number;
  };
  replays: ReplayScenario[];
}
