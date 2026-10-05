"""Phân tích bố cục trang: cắt ô, tách dòng, đo độ dày nét, trích mẫu huấn luyện OCR."""

from pathlib import Path

import cv2
import numpy as np

from .data import parse_markdown
from .grid import (GridTable, _runs, binarize, boundary_presence, deskew_gray, deskew_gray_fast,
                   detect_grid_tables, line_masks, load_gray, merge_matrix)
from .metric import BOLD_RE, anchor_spans, cell_text, is_bold


# =============================================================================
# PHẦN 1: LẤY TỪ BASELINE (mục 5 'Mô hình nhận dạng ô' của baseline.ipynb)
# =============================================================================

# [baseline: số 40 viết cứng trong normalize_crop/collate_cells, nay đặt thành hằng]
CROP_HEIGHT = 40


MAX_CROP_WIDTH = 384


# [baseline: điều kiện len(text) > 52 trong collect_cell_samples]
MAX_TRAIN_TEXT = 52          # như baseline: bỏ mẫu quá dài


# [baseline, đã sửa: tách phần tìm mực ra ink_mask(); thêm tham số clean]
def normalize_crop(crop: np.ndarray, target_height: int = CROP_HEIGHT, max_width: int = MAX_CROP_WIDTH,
                   clean: bool = True) -> np.ndarray:
    """Chuẩn hoá ảnh ô cho OCR: cắt sát chữ, đưa về cao 40 px, giới hạn bề rộng."""
    if crop.size == 0:
        return np.full((target_height, 16), 255, dtype=np.uint8)
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if clean:
        crop = clean_crop(crop)
    ink = ink_mask(crop, clean)
    ys, xs = np.nonzero(ink)
    if not len(xs):
        return np.full((target_height, 16), 255, dtype=np.uint8)
    y0, y1 = max(0, int(ys.min()) - 2), min(crop.shape[0], int(ys.max()) + 3)
    x0, x1 = max(0, int(xs.min()) - 3), min(crop.shape[1], int(xs.max()) + 4)
    trimmed = cv2.GaussianBlur(crop, (3, 3), 0)[y0:y1, x0:x1]
    scale = target_height / max(1, trimmed.shape[0])
    width = max(4, min(max_width, round(trimmed.shape[1] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(trimmed, (width, target_height), interpolation=interpolation)


def encode_png(image: np.ndarray) -> bytes:
    """Nén ảnh thành bytes PNG để lưu mẫu tốn ít RAM."""
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Không mã hóa được crop")
    return data.tobytes()


def decode_png(data: bytes) -> np.ndarray:
    """Giải nén bytes PNG ngược lại thành ảnh."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Không giải mã được crop")
    return image


# =============================================================================
# PHẦN 2: VIẾT MỚI
# =============================================================================

def ink_mask(crop: np.ndarray, blank_check: bool = True) -> np.ndarray:
    """Ảnh nhị phân phần mực (chữ) trong ô, xoá viền mỏng để không dính đường kẻ."""
    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Ô trống: Otsu vẫn chia đôi nhiễu nền -> coi là không có mực.
    if blank_check and int(blurred.max()) - int(blurred.min()) < 40:
        ink[:] = 0
    border = max(2, round(min(crop.shape) * 0.04))
    ink[:border] = 0
    ink[-border:] = 0
    ink[:, :border] = 0
    ink[:, -border:] = 0
    return ink


def clean_crop(crop: np.ndarray) -> np.ndarray:
    """Đảo màu ô nền tối chữ sáng thành chữ tối trên nền sáng."""
    if crop.size and float(np.median(crop)) < 110:
        return 255 - crop
    return crop


def is_blank(crop: np.ndarray) -> bool:
    """Kiểm tra ô có trống (không có mực) hay không."""
    return crop.size == 0 or not np.any(ink_mask(clean_crop(crop)))


def split_lines(crop: np.ndarray) -> list[np.ndarray]:
    """Cắt ô nhiều dòng thành từng dòng chữ dựa vào khoảng trắng giữa các dòng."""
    if crop.size == 0:
        return [crop]
    ink = ink_mask(clean_crop(crop))
    profile = np.count_nonzero(ink, axis=1)
    bands = [list(band) for band in _runs(profile > 0, gap=1)]
    if len(bands) <= 1:
        return [crop]
    typical = float(np.median([end - start + 1 for start, end in bands]))
    typical = max(typical, max(end - start + 1 for start, end in bands) * 0.6)
    # Dải quá mỏng (dấu thanh tiếng Việt) hoặc quá sát dòng bên cạnh -> gộp vào dòng đó.
    merged = [bands[0]]
    for start, end in bands[1:]:
        previous = merged[-1]
        gap = start - previous[1] - 1
        small_prev = previous[1] - previous[0] + 1 < 0.45 * typical
        small_this = end - start + 1 < 0.45 * typical
        if gap < 0.25 * typical or small_prev or small_this:
            previous[1] = end
        else:
            merged.append([start, end])
    if len(merged) <= 1:
        return [crop]
    # Cắt tại điểm giữa khoảng trắng giữa hai dòng.
    cuts = [0]
    for (_, end), (start, _) in zip(merged, merged[1:]):
        cuts.append((end + start) // 2)
    cuts.append(crop.shape[0])
    return [crop[top:bottom] for top, bottom in zip(cuts, cuts[1:])]


def stroke_width(line_image: np.ndarray) -> float:
    """Đo độ dày nét chữ trung bình (chữ đậm có nét dày hơn chữ thường)."""
    ink = cv2.threshold(line_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if not np.any(ink):
        return 0.0
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ridge = distance[(distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))) & (ink > 0)]
    return float(2 * np.mean(ridge)) if ridge.size else 0.0


def prepare_page(path: Path, exp: dict) -> np.ndarray:
    """Đọc ảnh trang ở dạng xám rồi chỉnh nghiêng."""
    gray = load_gray(path)
    return (deskew_gray_fast if exp.get("fast_deskew") else deskew_gray)(gray)[0]


# [mới, chứa bộ lọc 7 cột x 15 hàng lấy từ mục 7 của baseline]
def find_tables(gray: np.ndarray, exp: dict) -> list[GridTable]:
    """Tìm các bảng kẻ lưới trên trang (có thể bật bộ lọc kích thước của baseline)."""
    tables = detect_grid_tables(gray)
    if exp.get("size_filter"):
        # Bộ lọc của baseline: chỉ giữ bảng tối đa 7 cột x 15 hàng.
        tables = [table for table in tables if 2 <= table.cols <= 7 and 2 <= table.rows <= 15]
    return tables


def table_matrix(table: GridTable, binary, horizontal, vertical, exp: dict) -> list[list[str | None]]:
    """Dựng ma trận ô gộp cho một bảng (toàn ô đơn nếu tắt công tắc merge)."""
    if not exp.get("merge"):
        return [[None] * table.cols for _ in range(table.rows)]
    v_line, h_line = boundary_presence(table, binary, horizontal, vertical)
    return merge_matrix(table.rows, table.cols, v_line, h_line)


# [mới, tách ra từ đoạn cắt ô (+2/-2 px) của baseline]
def cell_crop(gray: np.ndarray, table: GridTable, row: int, col: int, row_span: int = 1,
              col_span: int = 1) -> np.ndarray:
    """Cắt vùng ảnh của một ô (hoặc cả vùng ô gộp) theo toạ độ lưới."""
    x0, x1 = table.x_edges[col] + 2, table.x_edges[col + col_span] - 2
    y0, y1 = table.y_edges[row] + 2, table.y_edges[row + row_span] - 2
    return gray[y0:y1, x0:x1]


def cell_lines(crop: np.ndarray, exp: dict) -> list[np.ndarray]:
    """Biến ảnh một ô thành danh sách ảnh dòng chữ đã chuẩn hoá để đưa vào OCR."""
    clean = bool(exp.get("clean_crops"))
    width = exp.get("max_width", MAX_CROP_WIDTH)
    if clean and is_blank(crop):
        return []
    parts = split_lines(crop) if exp.get("multiline") else [crop]
    lines = [normalize_crop(part, max_width=width, clean=clean) for part in parts]
    if clean:
        lines = [line for line in lines if line.shape[1] > 16 or np.any(line < 128)]
    return lines


# [mới, viết lại từ grid_page_markdown của baseline (tách phần OCR ra ngoài)]
def analyze_page(args: tuple[str, dict]) -> list[dict]:
    """Phân tích một trang: tìm bảng, ô gộp, cắt ô thành ảnh dòng chữ (chưa OCR)."""
    path, exp = args
    gray = prepare_page(Path(path), exp)
    binary = binarize(gray)
    horizontal, vertical = line_masks(binary)
    result = []
    for table in find_tables(gray, exp):
        matrix = table_matrix(table, binary, horizontal, vertical, exp)
        spans = anchor_spans([[value or "" for value in row] for row in matrix])
        cells = {}
        for (row, col), (row_span, col_span) in spans.items():
            lines = cell_lines(cell_crop(gray, table, row, col, row_span, col_span), exp)
            cells[(row, col)] = {"lines": lines,
                                 "stroke": stroke_width(lines[0]) if lines else 0.0}
        result.append({"bbox": table.bbox, "rows": table.rows, "cols": table.cols,
                       "x_edges": table.x_edges, "y_edges": table.y_edges,
                       "matrix": matrix, "cells": cells})
    return result


# [mới, viết lại từ collect_cell_samples của baseline (xử lý từng trang, thêm tách dòng)]
def page_samples(args: tuple[str, str, dict]) -> tuple[list[tuple[bytes, str]], list[tuple[float, bool]]]:
    """Tạo mẫu huấn luyện OCR (ảnh dòng chữ, chữ đúng) từ một trang train và nhãn của nó."""
    path, label_path, exp = args
    gray = prepare_page(Path(path), exp)
    detected = detect_grid_tables(gray)
    labels = parse_markdown(Path(label_path).read_text(encoding="utf-8"))
    if [(item.rows, item.cols) for item in detected] != [(len(table), len(table[0])) for table in labels]:
        return [], []
    clean = bool(exp.get("clean_crops"))
    width = exp.get("max_width", MAX_CROP_WIDTH)
    samples, bold_stats = [], []
    for grid, label in zip(detected, labels):
        for (row, col), (row_span, col_span) in anchor_spans(label).items():
            raw = label[row][col]
            text = cell_text(raw)
            if not text:
                continue
            crop = cell_crop(gray, grid, row, col, row_span, col_span)
            if "\n" in text:
                if not exp.get("multiline"):
                    continue
                parts = split_lines(crop)
                texts = text.split("\n")
                if len(parts) != len(texts):
                    continue
                pairs = list(zip(parts, texts))
            else:
                pairs = [(crop, text)]
            first = None
            for part, part_text in pairs:
                if len(part_text) > MAX_TRAIN_TEXT:
                    continue
                image = normalize_crop(part, max_width=width, clean=clean)
                first = image if first is None else first
                samples.append((encode_png(image), part_text))
            if first is not None:
                bold_stats.append((stroke_width(first), is_bold(raw)))
    return samples, bold_stats


def draw_tables(gray: np.ndarray, tables: list[dict]) -> np.ndarray:
    """Vẽ kết quả dò bảng lên ảnh: viền đỏ = từng ô, nền vàng = ô gộp (ảnh RGB)."""
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = image.copy()
    boxes = []
    for table in tables:
        xs, ys = table["x_edges"], table["y_edges"]
        spans = anchor_spans([[value or "" for value in row] for row in table["matrix"]])
        for (row, col), (row_span, col_span) in spans.items():
            box = ((xs[col], ys[row]), (xs[col + col_span], ys[row + row_span]))
            boxes.append(box)
            if row_span > 1 or col_span > 1:
                cv2.rectangle(overlay, *box, (255, 215, 0), -1)
    image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    for box in boxes:
        cv2.rectangle(image, *box, (220, 0, 0), 2)
    for table in tables:
        x0, y0 = table["bbox"][:2]
        cv2.putText(image, f"{table['rows']} hang x {table['cols']} cot", (x0, max(30, y0 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 0, 0), 3)
    return image
