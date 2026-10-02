import sys
from importlib import import_module
from pathlib import Path

MIN_PYTHON = (3, 9)
PACKAGES = ["numpy", "pandas", "matplotlib", "seaborn", "scipy", "sklearn"]
MIN_SKLEARN = (1, 2)
DATA_PATH = Path(__file__).resolve().parent / "data" / "customer_360_ml_workshop.csv"


def version_tuple(version):
    parts = []
    for piece in version.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def main():
    ok = True

    python_version = sys.version_info[:2]
    status = "OK" if python_version >= MIN_PYTHON else "FAIL"
    print(f"[{status}] Python {sys.version.split()[0]} (required >= {'.'.join(map(str, MIN_PYTHON))})")
    ok &= python_version >= MIN_PYTHON

    for name in PACKAGES:
        try:
            module = import_module(name)
            print(f"[OK]   {name} {module.__version__}")
            if name == "sklearn" and version_tuple(module.__version__) < MIN_SKLEARN:
                print(f"[FAIL] scikit-learn >= {'.'.join(map(str, MIN_SKLEARN))} is required")
                ok = False
        except ImportError:
            print(f"[FAIL] {name} is not installed")
            ok = False

    if DATA_PATH.exists():
        print(f"[OK]   dataset found: {DATA_PATH.relative_to(DATA_PATH.parents[1])}")
    else:
        print(f"[FAIL] dataset not found: {DATA_PATH}")
        ok = False

    print()
    if ok:
        print("Environment is ready.")
    else:
        print("Environment has problems. Run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
