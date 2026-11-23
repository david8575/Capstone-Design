"""평가용 데이터 로더 (라벨, dev/test 분할, 사고유형 표)"""
import csv
import json

from eval.labels import AI_DIR, OUT_CSV as LABELS_CSV
from eval.make_splits import SPLITS_JSON
from eval.type_table import TYPES_CSV

RESULTS_DIR = AI_DIR / "eval" / "results"


def load_labels() -> list[dict]:
    with LABELS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(name: str) -> list[str]:
    """'dev' 또는 'test' 에 속한 video_name 목록"""
    return json.loads(SPLITS_JSON.read_text(encoding="utf-8"))[name]


def load_types() -> dict[int, dict]:
    """사고유형 번호 → {description, rate_a, rate_b}"""
    with TYPES_CSV.open(encoding="utf-8") as f:
        return {
            int(r["type_id"]): {**r, "rate_a": int(r["rate_a"]), "rate_b": int(r["rate_b"])}
            for r in csv.DictReader(f)
        }