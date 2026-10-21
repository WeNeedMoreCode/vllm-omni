# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Print a coarse step breakdown and optionally compare two captured profiles."""

import argparse
import json
from pathlib import Path

LABELS = {
    "attention_core": "Attention core",
    "mlp": "MLP",
    "attention_projections": "Q/K/V/output projections",
    "normalization": "Layer normalization",
    "other": "Other",
    "total": "Whole sampling step",
}


def parts(report):
    total = next(step["wall_seconds_excluding_checkpoint_io"] for step in report["step_times"] if step["index"] == 1)
    result = {
        key: report["module_aggregates"].get("step1." + key, {}).get("device_interval_seconds", 0.0)
        for key in LABELS
        if key not in {"other", "total"}
    }
    result["other"] = total - sum(result.values())
    result["total"] = total
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    current = json.loads(args.report.read_text())
    values = parts(current)
    print(f"Current: {current['device']}, {current['dtype']}; step 1, including both CFG branches")
    if args.reference is None:
        print(f"{'Component':29} {'Seconds':>10} {'Share':>9}")
        for key, label in LABELS.items():
            print(f"{label:29} {values[key]:10.3f} {values[key] / values['total']:8.1%}")
    else:
        reference = json.loads(args.reference.read_text())
        if reference["sampling"] != current["sampling"] or reference["case"] != current["case"]:
            raise ValueError("Sampling setup or case differs; do not compare as the same workload")
        previous = parts(reference)
        print(f"Reference: {reference['device']}, {reference['dtype']}")
        print(f"{'Component':29} {'Current s':>10} {'Reference s':>12} {'Ref/current':>12}")
        for key, label in LABELS.items():
            ratio = f"{previous[key] / values[key]:.2f}x" if values[key] > 0 else "n/a"
            print(f"{label:29} {values[key]:10.3f} {previous[key]:12.3f} {ratio:>12}")
        print("Different device dtypes/execution paths: this is a performance breakdown, not a quality parity check.")
    print("Stream intervals include launch gaps. Parent GEN-layer totals are excluded to avoid double counting.")
    print("Checkpoint I/O is excluded; instrumentation overhead remains. No full video decoding in this probe.")


if __name__ == "__main__":
    main()
