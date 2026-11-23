"""
실행 (AI 폴더에서): python -m eval.type_table
"""
import csv
import re
from collections import Counter

from pypdf import PdfReader

from eval.labels import AI_DIR, OUT_CSV as LABELS_CSV

DOCS_DIR = AI_DIR / "data" / "docs"
TYPES_CSV = AI_DIR / "data" / "accident_types.csv"
OBJECTS = ("차대차", "차대보행자", "차대자전거", "차대이륜차", "차대 이륜차")
ROW_END = re.compile(r"(\d+) (\d+) (\d+)$")          # ... 과실A 과실B 사고유형번호


def extract_text() -> str:
    pdf = next(DOCS_DIR.glob("*데이터설명서*.pdf"))
    return "\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)


def parse_types(text: str) -> dict[int, dict]:
    start = text.index("사고객체 사고장소 사고장소특징")
    end = text.index("3 라벨링데이터 실제예시")

    types, buf, obj = {}, "", ""
    for line in text[start:end].splitlines():
        line = line.strip(" \u3000")
        # 페이지 구분선, 쪽 번호, 표 머리글은 건너뜀
        if not line or re.fullmatch(r"- \d+ -", line) or line.startswith("사고객체 사고장소"):
            continue
        buf = f"{buf} {line}".strip()      # PDF에서 한 행이 여러 줄로 쪼개진 경우 이어붙임
        m = ROW_END.search(buf)
        if not m:
            continue
        body = buf[:m.start()].strip()
        for o in OBJECTS:                  # 사고대상은 그룹 첫 행에만 적혀 있음
            if body.startswith(o + " "):
                obj, body = o.replace(" ", ""), body[len(o):].strip()
        type_id = int(m.group(3))
        types[type_id] = {
            "type_id": type_id,
            "object": obj,
            "description": body,
            "rate_a": int(m.group(1)),
            "rate_b": int(m.group(2)),
        }
        buf = ""
    return types


def main():
    types = parse_types(extract_text())
    missing = [i for i in range(434) if i not in types]
    assert not missing, f"누락된 사고유형: {missing}"

    with TYPES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type_id", "object", "description", "rate_a", "rate_b"])
        writer.writeheader()
        writer.writerows(types[i] for i in sorted(types))
    print(f"저장: {TYPES_CSV} ({len(types)}종)", dict(Counter(t["object"] for t in types.values())))

    # 검증: 라벨의 과실비율이 표와 일치하는지
    with LABELS_CSV.open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    mismatch = Counter(
        (int(r["accident_type"]), int(r["rate_a"]))
        for r in labels if types[int(r["accident_type"])]["rate_a"] != int(r["rate_a"])
    )
    n_bad = sum(mismatch.values())
    print(f"라벨-표 과실 일치: {len(labels) - n_bad}/{len(labels)} "
          f"({(len(labels) - n_bad) / len(labels):.2%})")
    print("불일치 상위 (사고유형, 라벨 rate_a):", mismatch.most_common(5))


if __name__ == "__main__":
    main()