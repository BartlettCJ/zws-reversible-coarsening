from __future__ import annotations

import argparse
import json
from pathlib import Path

from zws_nullity_lab.atomic import atomic_write_json
from zws_nullity_lab.density_threshold_campaign import (
    load_and_verify_density_manifest,
    run_density_campaign,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
MANIFEST = EXPERIMENT_DIR / "FINITE_SIZE_MANIFEST.json"
RESULT_PARENT = ROOT / "results" / "homogeneous_density_threshold_20260826"
SUMMARY_PATH = RESULT_PARENT / "finite_size_campaign_summary.json"
STAGES = ("finite_size_512", "finite_size_1024", "finite_size_2048")


def validate_frozen_stages(manifest: dict) -> None:
    observed = tuple(manifest.get("stages", ()))
    if len(observed) != len(STAGES) or set(observed) != set(STAGES):
        raise RuntimeError("finite-size manifest stages differ from the frozen stage set")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gated finite-size check in order.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest, digest, _ = load_and_verify_density_manifest(MANIFEST)
    # atomic_write_json sorts object keys, so mapping iteration order is not the
    # scientific execution order. Validate the exact frozen set, then execute
    # using STAGES below (512 -> 1024 -> 2048).
    validate_frozen_stages(manifest)
    stage_summaries = []
    for stage_id in STAGES:
        summary = run_density_campaign(
            MANIFEST,
            RESULT_PARENT / stage_id,
            stage_id=stage_id,
            device=args.device,
        )
        stage_summaries.append(summary)
        combined = {
            "schema": "zwslab-density-threshold-finite-size-campaign-summary-v1",
            "status": "running" if summary["status"] == "complete" else summary["status"],
            "manifest_sha256": digest,
            "completed_stage_count": sum(
                item["status"] == "complete" for item in stage_summaries
            ),
            "planned_stage_count": len(STAGES),
            "stage_summaries": stage_summaries,
        }
        atomic_write_json(SUMMARY_PATH, combined)
        if summary["status"] != "complete":
            print(json.dumps(combined, indent=2))
            raise SystemExit(1 if summary["status"] == "failed" else 2)
    combined["status"] = "complete"
    atomic_write_json(SUMMARY_PATH, combined)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
