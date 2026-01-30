
# hawkeye-lite — Real-time multi-camera tennis tracking (student MAANG demo)

A reproducible, interview-grade implementation of the core ideas behind Hawk-Eye: camera calibration,
deterministic per-camera detection, multi-view triangulation, physics-aware tracking, and line-calling —
built for clarity, testability, and real-time reasoning on consumer hardware.

Goals:
- Demonstrate mastery of multi-view geometry, numerical stability, and physics-aware filters.
- Provide clean, modular code suitable for interview walkthroughs.
- Supply synthetic data and tests so reviewers can reproduce results locally.

Quickstart:
1. Install: `pip install -r requirements.txt`
2. Run unit tests: `pytest -q`
3. Launch demo notebook: `jupyter lab demo/demo.ipynb`
