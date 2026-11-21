import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("Hawk-Eye Lite v2.0 Studio", () => {
  it("renders 3D reconstruction studio, switches trajectory stages, and navigates tabs", () => {
    render(<App />);
    expect(
      screen.getByText(/Hawk-Eye Lite v2.0 — 3D Broadcast & Line-Call Studio/i),
    ).toBeInTheDocument();

    const knnBtn = screen.getByRole("button", { name: /2. Multi-Plane KNN/i });
    fireEvent.click(knnBtn);
    expect(knnBtn.className).toContain("active");

    const confTab = screen.getByRole("button", { name: /12-Config Court Calibration/i });
    fireEvent.click(confTab);
    expect(screen.getByText(/12 Four-Point Court Configurations/i)).toBeInTheDocument();

    const benchTab = screen.getByRole("button", { name: /300-Scenario Benchmark/i });
    fireEvent.click(benchTab);
    expect(screen.getByText(/300-Scenario Paired Ablation Benchmark/i)).toBeInTheDocument();
  });
});
