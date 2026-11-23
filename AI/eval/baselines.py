"""
실행 (AI 폴더에서): python -m eval.baselines --split dev
"""
import argparse
import json
from collections import Counter, defaultdict

from eval.dataset import RESULTS_DIR, load_labels, load_split, load_types
from eval.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()

    labels = load_labels()
    gts = {r["video_name"]: r for r in labels}
    train = [r for r in labels if r["split"] == "train"]
    types_tbl = load_types()

    # Training 통계
    global_major = Counter(int(r["rate_a"]) for r in train).most_common(1)[0][0]
    place_rate, place_types = defaultdict(Counter), defaultdict(Counter)
    for r in train:
        place_rate[r["place_folder"]][int(r["rate_a"])] += 1
        place_types[r["place_folder"]][int(r["accident_type"])] += 1

    systems = {
        # 하한선: 아무 정보 없이 찍기
        "always_50": lambda g: {"rate_a": 50},
        "global_majority": lambda g: {"rate_a": global_major},
        # 사용자가 도로 유형만 알려준 경우 (= 현재 앱의 road_type 입력)
        "place_majority": lambda g: {
            "rate_a": place_rate[g["place_folder"]].most_common(1)[0][0],
            "types": [t for t, _ in place_types[g["place_folder"]].most_common(5)],
        },
        # 상한선: 사고유형을 정확히 안다면 (영상 분석이 완벽한 경우)
        "oracle_type": lambda g: {
            "rate_a": types_tbl[int(g["accident_type"])]["rate_a"],
            "types": [int(g["accident_type"])],
        },
    }

    names = load_split(args.split)
    results = {}
    print(f"[{args.split}] n={len(names)}")
    print(f"{'system':18s} {'MAE':>6s} {'exact':>7s} {'±10':>6s} {'side':>6s} {'top1':>6s} {'top5':>6s}")
    for sys_name, predict in systems.items():
        preds = [{"video_name": n, **predict(gts[n])} for n in names]
        m = compute_metrics(preds, gts)
        results[sys_name] = m
        print(f"{sys_name:18s} {m['rate_mae']:6.1f} {m['rate_exact']:7.3f} {m['rate_within10']:6.3f} "
              f"{m['side_acc']:6.3f} {m.get('type_top1', float('nan')):6.3f} {m.get('type_top5', float('nan')):6.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"baselines_{args.split}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()