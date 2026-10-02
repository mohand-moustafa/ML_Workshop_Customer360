"""
main.py - run the whole Customer 360 workshop solution end to end.

Usage (from the repository root):
    python solutions/main.py                 # run every step
    python solutions/main.py eda regression  # run selected steps only

Steps (in order): eda, classification, regression, unsupervised, integration
"""

import sys
import time
from importlib import import_module
from pathlib import Path

# Make `import common` and the numbered modules importable no matter where the
# script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS = {
    "eda": ("01_eda", "Part A   - EDA"),
    "classification": ("02_classification", "Part C   - Churn classification"),
    "regression": ("03_regression", "Part B+D - Revenue regression"),
    "unsupervised": ("04_unsupervised", "Part E+F - Segmentation, PCA, anomalies"),
    "integration": ("05_customer360_integration", "Part G   - Customer 360 integration"),
}


def main(selected: list[str]) -> None:
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        sys.exit(f"Unknown step(s): {unknown}. Choose from: {list(STEPS)}")

    steps = selected or list(STEPS)
    total_start = time.time()
    for key in steps:
        module_name, label = STEPS[key]
        print("\n" + "#" * 78)
        print(f"# {label}")
        print("#" * 78)
        start = time.time()
        import_module(module_name).run()
        print(f"\n[done] {label} in {time.time() - start:.1f}s")

    print(f"\nAll requested steps finished in {time.time() - total_start:.1f}s")
    print("Outputs are in the outputs/ folder.")


if __name__ == "__main__":
    main(sys.argv[1:])
