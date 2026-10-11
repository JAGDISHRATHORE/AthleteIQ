"""Run the AthleteIQ pipeline.

    python scripts/run_pipeline.py download   # fetch the public dataset (~330 MB)
    python scripts/run_pipeline.py build      # quality checks, metrics, features, clusters
    python scripts/run_pipeline.py train      # baselines + models + intervals + SHAP
    python scripts/run_pipeline.py all        # all of the above
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from athleteiq import pipeline  # noqa: E402
from athleteiq.data_loader import download_raw  # noqa: E402

STEPS = {"download": download_raw, "build": pipeline.build, "train": pipeline.train}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STEPS if step == "all" else [step]):
        print(f"== {name}")
        STEPS[name]()
