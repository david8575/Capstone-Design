"""
실행 (AI 폴더에서): python -m eval.make_splits
"""
import csv
import json
import random
from collections import defaultdict

from eval.labels import AI_DIR, OUT_CSV

SEED = 42
DEV_RATIO = 0.3
SPLITS_JSON = AI_DIR / "eval" / "splits.json"


def main():
    with OUT_CSV.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val" and r["video_path"]]

    by_place = defaultdict(list)
    for r in rows:
        by_place[r["place_folder"]].append(r["video_name"])

    rng = random.Random(SEED)
    dev, test = [], []
    for place, names in sorted(by_place.items()):
        names = sorted(names)          # 순서 고정 후 셔플 → 항상 같은 결과
        rng.shuffle(names)
        k = round(len(names) * DEV_RATIO)
        dev += names[:k]
        test += names[k:]
        print(f"{place}: dev {k} / test {len(names) - k}")

    SPLITS_JSON.write_text(
        json.dumps({"seed": SEED, "dev_ratio": DEV_RATIO,
                    "dev": sorted(dev), "test": sorted(test)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장: {SPLITS_JSON} (dev {len(dev)} / test {len(test)})")


if __name__ == "__main__":
    main()