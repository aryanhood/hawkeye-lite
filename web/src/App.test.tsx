import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("Hawk-Eye Lite Studio", () => {
  it("renders 3D reconstruction studio, switches trajectory stages, and navigates tabs", () => {
    render(<App />);
    expect(
      screen.getByText(/Hawk-Eye Lite — Multi-View 3D Trajectory & Line-Call Studio/i),
    ).toBeInTheDocument();

    const knnBtn = screen.getByRole("button", { name: /2. Multi-Plane KNN/i });
    fireEvent.click(knnBtn);
    expect(knnBtn.className).toContain("active");

    const confTab = screen.getByRole("button", { name: /12-Config Court Calibration/i });
    fireEvent.click(confTab);
    expect(screen.getByText(/12 Four-Point Court Configurations/i)).toBeInTheDocument();

    const benchTab = screen.getByRole("button", { name: /300-Scenario Benchmark/i });
    fireEvent.click(benchTab);
    expect(screen.getByText(/300-Scenario 2D-Pixel-Only Benchmark/i)).toBeInTheDocument();
    expect(screen.getByText(/1-Cam Static Frame-0 Homography/i)).toBeInTheDocument();
    expect(screen.getByText(/1-Cam Dynamic Homography/i)).toBeInTheDocument();
    expect(screen.getByText(/2-Cam DLT 3D/i)).toBeInTheDocument();
    expect(screen.getByText(/4-Cam Stadium DLT 3D/i)).toBeInTheDocument();
  });
});
