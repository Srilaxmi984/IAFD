"""Train and evaluate the final ensemble.

    python scripts/run_final_ensemble.py --dev      # quick check on the dev sample
    python scripts/run_final_ensemble.py            # full data (use this for reported numbers)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iafd.final_ensemble import main  # noqa: E402

if __name__ == "__main__":
    main()
