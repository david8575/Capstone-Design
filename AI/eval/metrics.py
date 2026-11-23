"""평가 지표

예측 형식 (시스템 공통):
    {"video_name": str,
     "rate_a": int | None,      # A 과실 예측 (None = 시스템이 판단 실패)
     "types": list[int] | None} # 사고유형 후보, 확률 높은 순 (선택)
"""
import random


def side(rate_a: int) -> int:
    """과실이 더 큰 쪽: 1=A, -1=B, 0=동일(50:50)"""
    return (rate_a > 50) - (rate_a < 50)


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> list[float]:
    """평균의 95% 신뢰구간 (부트스트랩)"""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)]


def compute_metrics(preds: list[dict], gts: dict[str, dict]) -> dict:
    """preds: 예측 리스트 / gts: video_name → labels.csv 행"""
    assert preds, "예측이 비어 있음"

    abs_err, exact, within10, side_ok = [], [], [], []
    top1, top5 = [], []
    n_fail = 0

    for p in preds:
        g = gts[p["video_name"]]
        gt_a = int(g["rate_a"])

        pred_a = p.get("rate_a")
        if pred_a is None:        # 판단 실패 → 50:50으로 대체하고 커버리지에 반영
            n_fail += 1
            pred_a = 50

        err = abs(pred_a - gt_a)
        abs_err.append(err)
        exact.append(float(err == 0))
        within10.append(float(err <= 10))
        side_ok.append(float(side(pred_a) == side(gt_a)))

        types = p.get("types")
        if types is not None:
            gt_t = int(g["accident_type"])
            top1.append(float(types[:1] == [gt_t]))
            top5.append(float(gt_t in types[:5]))

    n = len(preds)
    mean = lambda xs: round(sum(xs) / len(xs), 4)
    out = {
        "n": n,
        "coverage": round(1 - n_fail / n, 4),
        "rate_mae": mean(abs_err),
        "rate_mae_ci95": bootstrap_ci(abs_err),
        "rate_exact": mean(exact),
        "rate_exact_ci95": bootstrap_ci(exact),
        "rate_within10": mean(within10),
        "side_acc": mean(side_ok),
    }
    if top1:
        out |= {"type_n": len(top1), "type_top1": mean(top1), "type_top5": mean(top5)}
    return out