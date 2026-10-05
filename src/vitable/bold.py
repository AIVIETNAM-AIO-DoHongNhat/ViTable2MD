"""Phát hiện chữ in đậm dựa trên độ dày nét chữ."""

import numpy as np


# =============================================================================
# =============================================================================

def fit(stats: list[tuple[float, bool]]) -> dict:
    """Học ngưỡng độ dày nét tốt nhất (F1 cao nhất) để tách chữ đậm và chữ thường."""
    strokes = np.array([stroke for stroke, _ in stats], dtype=float)
    labels = np.array([bold for _, bold in stats], dtype=bool)
    best = {"threshold": float("inf"), "f1": 0.0}
    for threshold in np.unique(np.round(strokes, 2)):
        predicted = strokes >= threshold
        hits = int(np.sum(predicted & labels))
        f1 = 2 * hits / max(1, int(predicted.sum()) + int(labels.sum()))
        if f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": round(f1, 4)}
    best["bold_rate"] = round(float(labels.mean()), 4) if len(labels) else 0.0
    return best


def decide(table: dict, rule: dict, row_vote: float = 0.6, edge_rows: bool = False,
           header_block: bool = False) -> set[tuple[int, int]]:
    """Chọn các ô in đậm theo ngưỡng, rồi cho đậm cả hàng nếu đa số ô trong hàng đậm.

    edge_rows: luôn cho đậm hàng đầu và hàng cuối (trong nhãn M1/M2, 100% ô ở hai hàng này in đậm).
    header_block: hàng đầu có ô gộp ngang [[H]] (bảng M2: tiêu đề + nhóm cột + tên cột) -> 3 hàng đầu đậm,
        ngược lại chỉ hàng đầu. Đúng 100% trên 783 bảng nhãn M1/M2.
    """
    cells = {position: cell for position, cell in table["cells"].items() if cell["lines"]}
    bold = {position for position, cell in cells.items() if cell["stroke"] >= rule["threshold"]}
    by_row: dict[int, list[tuple[int, int]]] = {}
    for position in cells:
        by_row.setdefault(position[0], []).append(position)
    for row, positions in by_row.items():
        share = sum(position in bold for position in positions) / len(positions)
        if share >= row_vote:
            bold.update(positions)
    if edge_rows:
        bold.update(position for position in cells if position[0] in (0, table["rows"] - 1))
    if header_block:
        header = 3 if "[[H]]" in table["matrix"][0] else 1
        bold.update(position for position in cells if position[0] < header)
    return bold
