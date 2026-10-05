"""Đọc dữ liệu, chia train/val, đọc/ghi Markdown, nối bảng qua trang, ghi nhật ký, đóng gói file nộp."""

import csv
import json
from pathlib import Path
import random
import unicodedata
import zipfile

import Levenshtein
import numpy as np


# =============================================================================
# LẤY TỪ BASELINE (mục 3 'Tiện ích dữ liệu và đóng gói kết quả' của baseline.ipynb)
# =============================================================================

SEED = 2026080701


def set_seed(seed: int = SEED) -> None:
    """Cố định seed ngẫu nhiên để mỗi lần chạy ra cùng kết quả."""
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl(path: Path) -> list[dict]:
    """Đọc file .jsonl: mỗi dòng là một đối tượng JSON."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest(split_dir: Path) -> list[dict]:
    """Đọc manifest.jsonl của một tập (danh sách tài liệu, đường dẫn ảnh và nhãn)."""
    path = split_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy manifest: {path}")
    return load_jsonl(path)


def read_label(split_dir: Path, record: dict) -> str:
    """Đọc file nhãn Markdown của một tài liệu (chuẩn hoá Unicode NFC)."""
    path = split_dir / record["label_path"]
    return unicodedata.normalize("NFC", path.read_text(encoding="utf-8")).strip() + "\n"


def split_markdown_row(line: str) -> list[str]:
    """Tách một dòng Markdown `| a | b |` thành danh sách ô, giữ nguyên dấu \| đã escape."""
    if not (line.startswith("|") and line.endswith("|")):
        raise ValueError("Dòng Markdown không bắt đầu/kết thúc bằng dấu |")
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line[1:-1]:
        if escaped:
            current.append("\\" + character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def parse_markdown(markdown: str) -> list[list[list[str]]]:
    """Đọc Markdown thành danh sách bảng (ma trận ô) và báo lỗi nếu sai định dạng."""
    tables: list[list[list[str]]] = []
    for block in unicodedata.normalize("NFC", markdown).strip().split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError("Bảng có ít hơn ba dòng")
        rows = [split_markdown_row(line) for line in lines]
        width = len(rows[0])
        if width < 2 or any(len(row) != width for row in rows):
            raise ValueError("Số ô giữa các dòng không nhất quán")
        if any(cell != "---" for cell in rows[1]):
            raise ValueError("Thiếu dòng phân cách Markdown")
        tables.append([rows[0], *rows[2:]])
    if not tables:
        raise ValueError("Không có bảng")
    return tables


def merge_errors(table: list[list[str]]) -> list[str]:
    """Lỗi cấu trúc ô gộp (BTC chấm 0 điểm cả tài liệu nếu sai).

    Quy ước rút từ 1100 nhãn: ô gốc + [[H]] liền bên phải + [[V]] liền bên dưới tạo thành một hình chữ nhật;
    phần trong hình chữ nhật là [[H]]; mọi [[H]]/[[V]] phải thuộc đúng một hình chữ nhật như vậy.
    """
    rows, cols = len(table), len(table[0])
    tokens = ("[[H]]", "[[V]]")
    owner: dict[tuple[int, int], tuple[int, int]] = {}
    errors: list[str] = []
    for r in range(rows):
        for c in range(cols):
            if table[r][c] in tokens:
                continue
            col_span = 1
            while c + col_span < cols and table[r][c + col_span] == "[[H]]":
                col_span += 1
            row_span = 1
            while r + row_span < rows and table[r + row_span][c] == "[[V]]":
                row_span += 1
            for rr in range(r, r + row_span):
                for cc in range(c, c + col_span):
                    if (rr, cc) == (r, c):
                        continue
                    if (rr, cc) in owner:
                        errors.append(f"({rr},{cc}) thuộc hai vùng gộp")
                    elif rr > r and cc > c and table[rr][cc] != "[[H]]":
                        errors.append(f"({rr},{cc}) trong vùng gộp phải là [[H]]")
                    owner[(rr, cc)] = (r, c)
    errors.extend(f"{table[r][c]} mồ côi ở ({r},{c})" for r in range(rows) for c in range(cols)
                  if table[r][c] in tokens and (r, c) not in owner)
    return errors


def drop_invalid_merges(table: list[list[str]]) -> list[list[str]]:
    """Bảng có ô gộp sai cấu trúc -> bỏ hết ô gộp (thành ô trống): mất ít điểm thay vì 0 điểm cả tài liệu."""
    if not merge_errors(table):
        return table
    return [["" if cell in ("[[H]]", "[[V]]") else cell for cell in row] for row in table]


def is_valid_markdown(markdown: str) -> bool:
    """Kiểm tra Markdown có hợp lệ theo định dạng BTC yêu cầu hay không (cú pháp + cấu trúc ô gộp)."""
    try:
        tables = parse_markdown(markdown)
    except (ValueError, IndexError):
        return False
    return "```" not in markdown and not any(merge_errors(table) for table in tables)


def write_predictions(output_dir: Path, records: list[dict], markdowns: list[str]) -> None:
    """Ghi mỗi tài liệu dự đoán ra một file .md (kiểm tra hợp lệ trước khi ghi)."""
    if len(records) != len(markdowns):
        raise ValueError("Số prediction không khớp manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"{record['id']}.md" for record in records}
    for existing in output_dir.glob("*.md"):
        if existing.name not in expected:
            existing.unlink()
    for record, markdown in zip(records, markdowns, strict=True):
        normalized = unicodedata.normalize("NFC", markdown).strip() + "\n"
        if not is_valid_markdown(normalized):
            raise ValueError(f"Prediction không hợp lệ: {record['id']}")
        # newline="\n": trên Windows write_text mặc định đổi \n thành \r\n -> "\r\n\r\n" giữa hai bảng
        # không còn là dòng trống "\n\n" như nhãn của BTC, tài liệu nhiều bảng bị đọc sai cú pháp.
        (output_dir / f"{record['id']}.md").write_text(normalized, encoding="utf-8", newline="\n")


def make_predictions_zip(output_dir: Path, zip_path: Path) -> None:
    """Nén các file .md thành file zip để nộp (file nằm ngay gốc zip)."""
    files = sorted(output_dir.glob("*.md"))
    if not files:
        raise ValueError("Thư mục prediction rỗng")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)   # tệp .md phải nằm ngay gốc ZIP


# =============================================================================
# =============================================================================

VAL_SEED = 2026
VAL_SIZE = 100


def find_root(start: Path | None = None) -> Path:
    """Tìm thư mục gốc repo (nơi chứa data/) để code chạy được dù mở từ src/ hay từ gốc."""
    start = (start or Path.cwd()).resolve()
    for folder in [start, *start.parents]:
        if (folder / "data" / "training_set").is_dir():
            return folder
    raise FileNotFoundError("Không tìm thấy thư mục data/training_set")


def train_val_split(records: list[dict], val_size: int = VAL_SIZE, seed: int = VAL_SEED):
    """Chia 1100 tài liệu train thành ~1000 train + 100 val, giữ đúng tỉ lệ độ khó M1–M4."""
    rng = random.Random(seed)
    by_level: dict[str, list[dict]] = {}
    for record in sorted(records, key=lambda item: item["id"]):
        by_level.setdefault(record.get("difficulty", "?"), []).append(record)
    val_ids: set[str] = set()
    for level, items in sorted(by_level.items()):
        take = round(val_size * len(items) / len(records))
        val_ids.update(item["id"] for item in rng.sample(items, take))
    train = [record for record in records if record["id"] not in val_ids]
    val = [record for record in records if record["id"] in val_ids]
    return train, val


def table_to_markdown(cells: list[list[str]]) -> str:
    """Ghép ma trận ô thành một bảng Markdown: dòng đầu, dòng ---, rồi các dòng còn lại."""
    """Ma trận ô (đã escape `|`) -> một khối bảng Markdown."""
    cols = len(cells[0])
    lines = ["| " + " | ".join(cells[0]) + " |", "| " + " | ".join(["---"] * cols) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in cells[1:])
    return "\n".join(lines)


def _similar(a: str, b: str) -> float:
    """Độ giống nhau của hai chuỗi (1 = giống hệt, 0 = khác hoàn toàn)."""
    if not a and not b:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def _unbold(cell: str) -> str:
    """Bỏ dấu ** bao quanh nội dung ô (nếu có)."""
    return cell[2:-2] if len(cell) >= 4 and cell.startswith("**") and cell.endswith("**") else cell


def stitch_pages(pages: list[list[list[list[str]]]]) -> list[list[list[str]]]:
    """Nối bảng bị ngắt sang trang sau và bỏ các hàng header bị lặp lại.

    Hàng cuối của trang trước không còn là hàng cuối bảng -> bỏ in đậm (luật in đậm chạy theo từng trang
    nên đã tô đậm nó; trong 36/36 nhãn M2 hai trang, hàng này là hàng thường).
    """
    tables: list[list[list[str]]] = []
    for page_index, page in enumerate(pages):
        for table_index, table in enumerate(page):
            if page_index > 0 and table_index == 0 and tables:
                previous = tables[-1]
                if len(previous[0]) == len(table[0]) and \
                        _similar(" ".join(previous[0]), " ".join(table[0])) >= 0.8:
                    repeat = 0
                    while repeat < min(len(previous), len(table) - 1) and \
                            _similar(" ".join(previous[repeat]), " ".join(table[repeat])) >= 0.8:
                        repeat += 1
                    previous[-1] = [_unbold(cell) for cell in previous[-1]]
                    previous.extend(table[repeat:])
                    continue
            tables.append(table)
    return tables


LOG_COLUMNS = ["time", "name", "note", "score", "teds", "bold_f1",
               "score_M1M2",   # Score chỉ trên tài liệu M1+M2 = đúng loại tài liệu của public/private test
               "score_M1","score_M2", "score_M3", "score_M4",
               "score_grid", "score_horizontal", "score_zebra", "score_none",
               "pages_with_table", "pages", "ocr_samples", "ocr_cer", "train_seconds", "predict_seconds",
               "config"]


def logged_names(path: Path) -> set[str]:
    """Tên các thí nghiệm đã có trong report/experiments.csv."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["name"] for row in csv.DictReader(handle)}


def append_log(path: Path, result: dict) -> None:
    """Thêm một dòng kết quả vào report/experiments.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        if header != LOG_COLUMNS:
            raise ValueError(f"Tiêu đề cột của {path.name} khác LOG_COLUMNS -> cập nhật file CSV trước khi ghi")
    row = {key: result.get(key, "") for key in LOG_COLUMNS}
    row["config"] = json.dumps(result.get("config", {}), ensure_ascii=False)
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
