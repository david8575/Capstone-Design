"""
실행 (AI 폴더에서): python -m eval.labels
"""
import csv
import json
from collections import Counter
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = AI_DIR / "data" / "095.교통사고_영상_데이터" / "01.데이터"
OUT_CSV = AI_DIR / "data" / "labels.csv"

SPLIT_DIRS = {"train": "1.Training", "val": "2.Validation"}
FIELDS = [
    "video_name", "split", "place_folder", "video_path",
    "filming_way", "point_of_view",
    "accident_type", "accident_place", "accident_place_feature",
    "vehicle_a_progress", "vehicle_b_progress",
    "rate_a", "rate_b", "schema",
]


def normalize(v: dict) -> dict:
    """라벨 스키마 변형 3종을 하나로 통일"""
    accident_type = v.get("traffic_accident_type", v.get("accident_type"))

    if "accident_negligence_rateA" in v:
        rate_a = v["accident_negligence_rateA"]
        rate_b = v["accident_negligence_rateB"]
        schema = "rateAB"
    else:
        # 단일 필드 = A 과실 (같은 사고유형의 rateA와 993/993 일치로 검증)
        rate_a = v["accident_negligence_rate"]
        rate_b = 100 - rate_a
        schema = "rate_single"

    if (accident_type is None or accident_type < 0 or not (0 <= rate_a <= 100) or rate_a + rate_b != 100):
        raise ValueError(f"비정상 라벨: {v.get('video_name')} (type={accident_type}, rate_a={rate_a})")

    return {
        "video_name": v["video_name"],
        "filming_way": v["filming_way"],              # bb=블랙박스, cc=CCTV
        "point_of_view": v["video_point_of_view"],    # 1=1인칭, 3=3인칭
        "accident_type": accident_type,
        "accident_place": v["accident_place"],
        "accident_place_feature": v["accident_place_feature"],
        "vehicle_a_progress": v["vehicle_a_progress_info"],
        "vehicle_b_progress": v["vehicle_b_progress_info"],
        "rate_a": rate_a,
        "rate_b": rate_b,
        "schema": schema,
    }


def load_labels() -> list[dict]:
    rows = []
    for split, split_dir in SPLIT_DIRS.items():
        label_root = DATA_ROOT / split_dir / "라벨링데이터_231108_add"
        for json_path in sorted(label_root.glob("*/*.json")):
            folder = json_path.parent.name                    # 예: VL_차대차_영상_주차장(...)
            v = json.loads(json_path.read_text(encoding="utf-8"))["video"]
            try:
                row = normalize(v)
            except ValueError as e:
                print(f"[제외] {e}")
                continue

            # 원천 영상 폴더: VL_ → VS_, TL_ → TS_
            video_folder = folder[0] + "S_" + folder[3:]
            video = DATA_ROOT / split_dir / "원천데이터_231108_add" / video_folder / f"{v['video_name']}.mp4"

            row["split"] = split
            row["place_folder"] = folder.split("_영상_", 1)[1]
            row["video_path"] = str(video.relative_to(AI_DIR)) if video.exists() else ""
            rows.append(row)
    return rows


def main():
    rows = load_labels()

    names = [r["video_name"] for r in rows]
    assert len(names) == len(set(names)), "video_name 중복 존재"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"저장: {OUT_CSV} ({len(rows)}건)")
    for split in SPLIT_DIRS:
        part = [r for r in rows if r["split"] == split]
        with_video = sum(1 for r in part if r["video_path"])
        print(f"\n[{split}] 라벨 {len(part)}건 / 영상 있음 {with_video}건")
        for place, n in Counter(r["place_folder"] for r in part).most_common():
            print(f"  {place}: {n}")
    print("\n스키마:", dict(Counter(r["schema"] for r in rows)))


if __name__ == "__main__":
    main()