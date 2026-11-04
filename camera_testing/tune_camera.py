from pathlib import Path
from itertools import product
from datetime import datetime
import ast
import contextlib
import csv
import io
import math
import re

ROOT = Path(__file__).resolve().parent
TEST_FILE = ROOT / "test_camera_accuracy.py"

OUTPUT = (
    ROOT / "camera_accuracy_results"
    / f"tuning_{datetime.now():%Y%m%d_%H%M%S_%f}"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

WINDOW_SIZES = [4, 6, 8]
RELATIVE_SIZES = [0.3, 0.4, 0.5]

source = TEST_FILE.read_text(encoding="utf-8-sig")


class ChangeSettings(ast.NodeTransformer):
    """Replace settings in memory without modifying the original file."""

    def __init__(self, window, relative):
        self.values = {
            "cornerRefinementWinSize": window,
            "relativeCornerRefinmentWinSize": relative,
        }
        self.found = set()

    def visit_Assign(self, node):
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "parameters"
                and target.attr in self.values
            ):
                node.value = ast.copy_location(
                    ast.Constant(self.values[target.attr]), node.value
                )
                self.found.add(target.attr)

        return node


# Read the per-position summaries printed by the existing test.
number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

summary_pattern = re.compile(
    r"X=\s*" + number
    + r":\s*(\d+)/(\d+)\s+valid"
    + r"\s*\|\s*mean range error=\s*" + number + r"\s*cm"
    + r"\s*\|\s*range RMSE=\s*" + number + r"\s*cm"
    + r"\s*\|\s*max range error=\s*" + number + r"\s*cm"
    + r"\s*\|\s*bearing RMSE=\s*" + number + r"\s*deg"
)

results = []

for run, (window, relative) in enumerate(
    product(WINDOW_SIZES, RELATIVE_SIZES), start=1
):
    label = f"window_{window}_relative_{relative:.1f}"
    print(f"\n[{run}/9] Testing {label} ...", flush=True)

    tree = ast.parse(source, filename=str(TEST_FILE))
    changer = ChangeSettings(window, relative)
    tree = changer.visit(tree)
    ast.fix_missing_locations(tree)

    missing = set(changer.values) - changer.found
    if missing:
        raise RuntimeError(
            f"Could not find these assignments in {TEST_FILE.name}: "
            f"{sorted(missing)}. Add them before creating ArucoDetector."
        )

    # Fresh globals for each run, with the original script's file path.
    namespace = {
        "__name__": "__main__",
        "__file__": str(TEST_FILE),
    }
    captured = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured):
            exec(compile(tree, str(TEST_FILE), "exec"), namespace)
    finally:
        # Preserve the complete output even if a run fails.
        (OUTPUT / f"{label}.txt").write_text(
            captured.getvalue(), encoding="utf-8"
        )

    text = captured.getvalue()
    matches = summary_pattern.findall(text)
    counts = re.search(r"Valid estimates:\s*(\d+)/(\d+)", text)

    if not matches or counts is None:
        raise RuntimeError(
            f"Could not read the test summary. Inspect {label}.txt."
        )

    valid, total = map(int, counts.groups())

    # Each row: X, valid count, total count, bias, range RMSE,
    # maximum absolute range error, bearing RMSE.
    groups = [
        {
            "n": int(m[1]),
            "bias": float(m[3]),
            "range_rmse": float(m[4]),
            "max_error": float(m[5]),
            "bearing_rmse": float(m[6]),
        }
        for m in matches
        if int(m[1]) > 0
    ]

    if valid == 0 or sum(g["n"] for g in groups) != valid:
        raise RuntimeError("Summary counts are incomplete or inconsistent.")

    # Combine group statistics, weighted by valid measurements.
    bias = sum(g["n"] * g["bias"] for g in groups) / valid
    range_rmse = math.sqrt(
        sum(g["n"] * g["range_rmse"] ** 2 for g in groups) / valid
    )
    bearing_rmse = math.sqrt(
        sum(g["n"] * g["bearing_rmse"] ** 2 for g in groups) / valid
    )
    max_error = max(g["max_error"] for g in groups)

    results.append({
        "window": window,
        "relative": relative,
        "valid": valid,
        "total": total,
        "mean_range_error_cm": bias,
        "range_rmse_cm": range_rmse,
        "max_range_error_cm": max_error,
        "bearing_rmse_deg": bearing_rmse,
    })

    print(
        f"  Valid: {valid}/{total}"
        f" | range RMSE: {range_rmse:.3f} cm"
        f" | bias: {bias:+.3f} cm"
        f" | max error: {max_error:.3f} cm"
        f" | bearing RMSE: {bearing_rmse:.4f} deg",
        flush=True,
    )

    # Save progress after every completed combination.
    with (OUTPUT / "tuning_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


print("\nCOMPARISON")
print("Sorted by most valid estimates, then lowest range RMSE.")
print("Window  Relative  Valid     RMSE(cm)  Bias(cm)  Max(cm)  Bearing(deg)")

ranked = sorted(
    results,
    key=lambda row: (-row["valid"], row["range_rmse_cm"]),
)

for row in ranked:
    print(
        f"{row['window']:>6}"
        f"  {row['relative']:>8.1f}"
        f"  {row['valid']:>3}/{row['total']:<3}"
        f"  {row['range_rmse_cm']:>10.3f}"
        f"  {row['mean_range_error_cm']:>+8.3f}"
        f"  {row['max_range_error_cm']:>7.3f}"
        f"  {row['bearing_rmse_deg']:>12.4f}"
    )

print("\nThese combined metrics use rounded per-position summaries.")
print("Use the original per-run results.csv files for exact analysis.")
print(f"Comparison and logs saved to: {OUTPUT}")