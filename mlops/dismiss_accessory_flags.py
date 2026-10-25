"""
One-shot script: dismiss auto-flagged accessory items that were flagged
by cross-style mismatches (score 0.15-0.40), not because they are broken.

Run while gateway is up:
    python mlops/dismiss_accessory_flags.py --gateway-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

FLAGS_PATH = Path(__file__).parent.parent / "gateway" / "low_score_flags.json"
_ACCESSORY_CATS_IN_NAME = (
    "shoe", "sneaker", "trainer", "boot", "sandal", "heel", "pump",
    "flat", "loafer", "mule", "slipper", "hat", "cap", "bag", "handbag",
    "tote", "purse", "backpack",
)


def _is_accessory(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in _ACCESSORY_CATS_IN_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be dismissed without calling the API")
    args = parser.parse_args()

    flags = json.loads(FLAGS_PATH.read_text())

    to_dismiss = []
    for f in flags:
        score  = f.get("judge_score")
        manual = f.get("manual", False)
        name   = f.get("name", "")
        pid    = f.get("product_id", "")

        if manual:
            print(f"  [SKIP]    manual flag: {name}")
            continue
        if score is not None and score < 0.10:
            print(f"  [KEEP]    genuinely bad (score={score}): {name}")
            continue
        if not _is_accessory(name):
            print(f"  [SKIP]    not an accessory: {name}")
            continue
        print(f"  [DISMISS] cross-style false-positive (score={score}): {name}")
        to_dismiss.append({"product_id": pid, "name": name})

    if not to_dismiss:
        print("Nothing to dismiss.")
        sys.exit(0)

    if args.dry_run:
        print(f"\nDry run — would dismiss {len(to_dismiss)} item(s).")
        sys.exit(0)

    print(f"\nDismissing {len(to_dismiss)} falsely-flagged accessory item(s)…")
    for item in to_dismiss:
        try:
            r = requests.post(
                f"{args.gateway_url}/low-score-flags/dismiss",
                json={"product_id": item["product_id"], "name": item["name"]},
                timeout=15,
            )
            r.raise_for_status()
            print(f"  OK: {item['name']}")
        except Exception as e:
            print(f"  FAIL: {item['name']} — {e}")

    print("Done.")


if __name__ == "__main__":
    main()
