"""Thước đo local: Document_Score = 0.9 * TEDS + 0.1 * Bold-F1."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re

from apted import APTED, Config
import Levenshtein

from .data import parse_markdown


# =============================================================================
# =============================================================================

BOLD_RE = re.compile(r"^\*\*(.*)\*\*$", re.DOTALL)


# [baseline, đã sửa: dùng hằng MERGE_TOKENS thay cho tập {'[[H]]', '[[V]]'}]
def anchor_spans(table: list[list[str]]) -> dict[tuple[int, int], tuple[int, int]]:
    """Từ ma trận có [[H]]/[[V]], tính mỗi ô gốc chiếm bao nhiêu hàng × bao nhiêu cột."""
    rows, cols = len(table), len(table[0])
    spans: dict[tuple[int, int], tuple[int, int]] = {}
    for row in range(rows):
        for col in range(cols):
            if table[row][col] in MERGE_TOKENS:
                continue
            col_span = 1
            while col + col_span < cols and table[row][col + col_span] == "[[H]]":
                col_span += 1
            row_span = 1
            while row + row_span < rows and table[row + row_span][col] == "[[V]]":
                row_span += 1
            spans[(row, col)] = (row_span, col_span)
    return spans


# [baseline, đã sửa: đổi tên từ clean_label_cell, thêm .strip()]
def cell_text(value: str) -> str:
    """Lấy chữ thuần của ô: bỏ **, đổi <br> thành xuống dòng, bỏ escape \|."""
    match = BOLD_RE.match(value)
    if match:
        value = match.group(1)
    return value.replace("<br>", "\n").replace("\\|", "|").strip()


# =============================================================================
# =============================================================================

MERGE_TOKENS = {"[[H]]", "[[V]]"}


def is_bold(value: str) -> bool:
    """Kiểm tra một ô có được bọc **...** (in đậm) hay không."""
    return bool(BOLD_RE.match(value)) and len(value) > 4


@dataclass
class Node:
    """Một nút của cây HTML: table / tr / td, kèm colspan, rowspan và nội dung chữ."""
    tag: str
    colspan: int = 1
    rowspan: int = 1
    content: str = ""
    children: list["Node"] = field(default_factory=list)


class TableConfig(Config):
    """Quy định chi phí khi so hai cây: khác span tốn 1, cùng span tốn theo độ khác nhau của chữ."""
    def rename(self, node1: Node, node2: Node) -> float:
        """Chi phí đổi nút này thành nút kia (0 = giống hệt, 1 = khác hoàn toàn)."""
        if node1.tag != node2.tag or node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        if node1.tag == "td":
            longest = max(len(node1.content), len(node2.content))
            if longest == 0:
                return 0.0
            return Levenshtein.distance(node1.content, node2.content) / longest
        return 0.0

    def children(self, node: Node) -> list[Node]:
        """Cho thư viện APTED biết các nút con của một nút."""
        return node.children


def markdown_to_tree(markdown: str) -> tuple[Node, int]:
    """Chuyển Markdown (có [[H]]/[[V]]) thành cây HTML để tính TEDS; trả về cả số nút."""
    root = Node("body")
    count = 1
    try:
        tables = parse_markdown(markdown)
    except (ValueError, IndexError):
        return root, count
    for table in tables:
        table_node = Node("table")
        count += 1
        spans = anchor_spans(table)
        for row_index, row in enumerate(table):
            tr = Node("tr")
            count += 1
            for col_index, value in enumerate(row):
                if (row_index, col_index) not in spans:
                    continue
                row_span, col_span = spans[(row_index, col_index)]
                tr.children.append(Node("td", col_span, row_span, cell_text(value)))
                count += 1
            table_node.children.append(tr)
        root.children.append(table_node)
    return root, count


def teds(pred_markdown: str, true_markdown: str) -> float:
    """TEDS = 1 − (số phép sửa cây / số nút của cây lớn hơn); 1 là giống hệt nhãn."""
    pred_tree, pred_count = markdown_to_tree(pred_markdown)
    true_tree, true_count = markdown_to_tree(true_markdown)
    distance = APTED(pred_tree, true_tree, TableConfig()).compute_edit_distance()
    return max(0.0, 1.0 - distance / max(pred_count, true_count))


def bold_cells(markdown: str) -> Counter:
    """Lấy danh sách nội dung các ô in đậm trong một tài liệu (đếm cả lặp lại)."""
    try:
        tables = parse_markdown(markdown)
    except (ValueError, IndexError):
        return Counter()
    return Counter(cell_text(value) for table in tables for row in table for value in row if is_bold(value))


def bold_f1(pred_markdown: str, true_markdown: str) -> float:
    """F1 giữa các ô in đậm của dự đoán và của nhãn."""
    pred, true = bold_cells(pred_markdown), bold_cells(true_markdown)
    if not pred and not true:
        return 1.0
    hits = sum((pred & true).values())
    return 2 * hits / (sum(pred.values()) + sum(true.values()))


def document_score(pred_markdown: str, true_markdown: str) -> dict[str, float]:
    """Điểm một tài liệu = 0.9 × TEDS + 0.1 × Bold-F1."""
    t, b = teds(pred_markdown, true_markdown), bold_f1(pred_markdown, true_markdown)
    return {"teds": t, "bold_f1": b, "score": 0.9 * t + 0.1 * b}


def evaluate(pred_dir: Path, split_dir: Path, records: list[dict]) -> tuple[dict[str, float], list[dict]]:
    """Chấm cả thư mục dự đoán so với nhãn: điểm trung bình (thang 100) + điểm từng tài liệu."""
    details = []
    for record in records:
        true = (split_dir / record["label_path"]).read_text(encoding="utf-8")
        pred_path = pred_dir / f"{record['id']}.md"
        pred = pred_path.read_text(encoding="utf-8") if pred_path.exists() else ""
        details.append({"id": record["id"], "difficulty": record.get("difficulty"),
                        "border": record.get("attributes", {}).get("border_mode"),
                        **document_score(pred, true)})
    summary = {key: 100 * sum(item[key] for item in details) / max(1, len(details))
               for key in ("teds", "bold_f1", "score")}
    return summary, details


def breakdown(details: list[dict], key: str) -> dict[str, float]:
    """Điểm trung bình theo nhóm (độ khó / kiểu viền) để biết đang mất điểm ở đâu."""
    groups: dict[str, list[float]] = {}
    for item in details:
        groups.setdefault(str(item[key]), []).append(item["score"])
    return {name: round(100 * sum(values) / len(values), 2) for name, values in sorted(groups.items())}
