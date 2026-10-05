"""Tiền xử lý ảnh (deskew, CLAHE, Otsu), dò lưới bảng kẻ viền và dò ô gộp."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# PHẦN 1: LẤY TỪ BASELINE (mục 4 'Xử lý cấu trúc bảng' của baseline.ipynb)
# =============================================================================

@dataclass(frozen=True)
class GridTable:
    """Một bảng dò được: khung bao và toạ độ các đường kẻ dọc (x_edges), ngang (y_edges)."""
    bbox: tuple[int, int, int, int]
    x_edges: tuple[int, ...]
    y_edges: tuple[int, ...]

    @property
    def rows(self) -> int:
        """Số hàng = số đường kẻ ngang − 1."""
        return max(0, len(self.y_edges) - 1)

    @property
    def cols(self) -> int:
        """Số cột = số đường kẻ dọc − 1."""
        return max(0, len(self.x_edges) - 1)


def _runs(mask: np.ndarray, minimum: int = 1, gap: int = 2) -> list[tuple[int, int]]:
    """Tìm các đoạn liên tiếp có giá trị True trong mảng 1 chiều (cho phép hở nhỏ `gap`)."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value - previous > gap:
            if previous - start + 1 >= minimum:
                result.append((start, previous))
            start = value
        previous = value
    if previous - start + 1 >= minimum:
        result.append((start, previous))
    return result


def _centres(runs: list[tuple[int, int]]) -> list[int]:
    """Lấy điểm giữa của mỗi đoạn."""
    return [round((left + right) / 2) for left, right in runs]


def load_gray(path: Path) -> np.ndarray:
    """Đọc ảnh ở dạng ảnh xám (1 kênh)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return gray


def deskew_gray(gray: np.ndarray, maximum_degrees: float = 4.0) -> tuple[np.ndarray, float]:
    """Chỉnh nghiêng: dò các đường thẳng dài bằng Hough, lấy trung vị góc rồi xoay ảnh."""
    edges = cv2.Canny(gray, 60, 180)
    height, width = gray.shape
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(70, width // 12),
        minLineLength=max(180, width // 5),
        maxLineGap=max(12, width // 80),
    )
    angles: list[float] = []
    if lines is not None:
        for raw in np.asarray(lines).reshape(-1, 4):   # OpenCV 4 trả (N,1,4), OpenCV 5 trả (N,4)
            x0, y0, x1, y1 = map(float, raw)
            angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180
            if abs(angle) <= maximum_degrees:
                angles.append(angle)
    if not angles:
        return gray, 0.0
    angle = float(np.median(angles))
    if abs(angle) < 0.08:
        return gray, 0.0
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated, angle


def binarize(gray: np.ndarray) -> np.ndarray:
    """Nhị phân hoá: CLAHE tăng tương phản rồi Otsu tách mực (trắng) khỏi nền (đen)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def line_masks(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tách riêng đường kẻ ngang và dọc bằng phép mở (opening) với kernel dẹt."""
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(45, width // 18), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, height // 45))),
    )
    return horizontal, vertical


def _line_positions(mask: np.ndarray, axis: int, extent: int) -> list[int]:
    """Chiếu mask lên một trục, vị trí nào có đủ nhiều pixel đường kẻ thì là một đường kẻ."""
    projection = np.count_nonzero(mask, axis=axis)
    threshold = max(8, round(extent * 0.42))
    positions: list[int] = []
    for start, end in _runs(projection >= threshold, gap=3):
        # A dark title band is seen by the line morphology as one thick run.
        # Its two sides are row boundaries; a normal 1–3 px rule contributes a
        # single centre coordinate.
        if end - start >= 9:
            positions.extend((start, end))
        else:
            positions.append(round((start + end) / 2))
    return positions


def _dedupe(values: list[int], tolerance: int = 6) -> list[int]:
    """Gộp các toạ độ đường kẻ quá gần nhau thành một."""
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group)) for group in groups]


def detect_grid_tables(gray: np.ndarray) -> list[GridTable]:
    """Tìm các bảng kẻ lưới: gom đường kẻ thành khối, rồi lấy toạ độ hàng/cột của từng khối."""
    binary = binarize(gray)
    horizontal, vertical = line_masks(binary)
    joined = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    candidates: list[GridTable] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.48 or h < 80 or w * h < width * height * 0.012:
            continue
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        local_h = horizontal[y0:y1, x0:x1]
        local_v = vertical[y0:y1, x0:x1]
        xs = _line_positions(local_v, axis=0, extent=y1 - y0)
        ys = _line_positions(local_h, axis=1, extent=x1 - x0)
        xs = _dedupe([x0 + value for value in xs])
        ys = _dedupe([y0 + value for value in ys])
        if len(xs) < 3 or len(ys) < 3:
            continue
        # Lines at the table boundary are occasionally one pixel shorter.  Use
        # the contour bounds only when they agree with the internal grid.
        if abs(xs[0] - x) > 10:
            xs.insert(0, x)
        if abs(xs[-1] - (x + w - 1)) > 10:
            xs.append(x + w - 1)
        if abs(ys[0] - y) > 10:
            ys.insert(0, y)
        if abs(ys[-1] - (y + h - 1)) > 10:
            ys.append(y + h - 1)
        if 2 <= len(xs) - 1 <= 18 and 2 <= len(ys) - 1 <= 60:
            candidates.append(GridTable((x, y, x + w, y + h), tuple(xs), tuple(ys)))

    # A merged grid may create nested contours; keep the largest region when
    # two boxes substantially overlap.
    candidates.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.rows * item.cols))
    kept: list[GridTable] = []
    for table in candidates:
        x0, y0, x1, y1 = table.bbox
        area = max(1, (x1 - x0) * (y1 - y0))
        duplicate = False
        for previous in kept:
            a0, b0, a1, b1 = previous.bbox
            intersection = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
            if intersection / min(area, max(1, (a1 - a0) * (b1 - b0))) > 0.72:
                duplicate = True
                break
        if not duplicate:
            kept.append(table)
    return kept


def blank_markdown(tables: list[GridTable]) -> str:
    """Tạo bảng Markdown rỗng có đúng số hàng/cột (baseline không dùng tới)."""
    blocks: list[str] = []
    for table in tables:
        rows = max(2, table.rows)
        cols = max(2, table.cols)
        lines = ["| " + " | ".join([""] * cols) + " |"]
        lines.append("| " + " | ".join(["---"] * cols) + " |")
        for _ in range(rows - 1):
            lines.append("| " + " | ".join([""] * cols) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip() + "\n"


# =============================================================================
# PHẦN 2: VIẾT MỚI
# =============================================================================

def deskew_gray_fast(gray: np.ndarray, maximum_degrees: float = 4.0, factor: float = 0.5) -> tuple[np.ndarray, float]:
    """Chỉnh nghiêng nhanh: đo góc trên ảnh thu nhỏ 1/2 rồi xoay ảnh gốc theo góc đó."""
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    _, angle = deskew_gray(small, maximum_degrees)
    if angle == 0.0:
        return gray, 0.0
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated, angle


def _coverage(mask: np.ndarray) -> float:
    """Tỉ lệ chiều dài đoạn biên có pixel đường kẻ (0 = không có, 1 = kẻ liền)."""
    if mask.size == 0:
        return 1.0
    return float(np.count_nonzero(mask.max(axis=1 if mask.shape[0] >= mask.shape[1] else 0))) / max(mask.shape)


def boundary_presence(table: GridTable, binary: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray,
                      band: int = 3, margin: int = 4, dense: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Đo xem từng đoạn biên giữa hai ô kề nhau có đường kẻ hay không."""
    xs, ys = table.x_edges, table.y_edges
    # Ô "đặc" = ô nền tối (dải tiêu đề): khoảng trống giữa hai chữ trông như đường kẻ -> bỏ qua biên ở đó.
    filled = np.zeros((table.rows, table.cols), dtype=bool)
    for r in range(table.rows):
        for c in range(table.cols):
            inner = binary[ys[r] + margin:ys[r + 1] - margin, xs[c] + margin:xs[c + 1] - margin]
            filled[r, c] = inner.size > 0 and np.count_nonzero(inner) > dense * inner.size
    v_line = np.ones((table.rows, max(0, table.cols - 1)))
    h_line = np.ones((max(0, table.rows - 1), table.cols))
    for r in range(table.rows):
        top, bottom = ys[r] + margin, ys[r + 1] - margin
        for c in range(table.cols - 1):
            x = xs[c + 1]
            v_line[r, c] = 0.0 if filled[r, c] and filled[r, c + 1] else                 _coverage(vertical[top:bottom, x - band:x + band + 1])
    for r in range(table.rows - 1):
        y = ys[r + 1]
        for c in range(table.cols):
            left, right = xs[c] + margin, xs[c + 1] - margin
            h_line[r, c] = 0.0 if filled[r, c] and filled[r + 1, c] else                 _coverage(horizontal[y - band:y + band + 1, left:right])
    return v_line, h_line


def merge_matrix(rows: int, cols: int, v_line: np.ndarray, h_line: np.ndarray,
                 threshold: float = 0.5) -> list[list[str | None]]:
    """Gom các ô không có biên ngăn cách thành ô gộp, đánh dấu [[H]]/[[V]] giống nhãn."""
    parent = list(range(rows * cols))

    def find(index: int) -> int:
        """Tìm ô "đại diện" của nhóm chứa ô này (union-find)."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        """Gộp nhóm của hai ô làm một."""
        parent[find(a)] = find(b)

    for r in range(rows):
        for c in range(cols - 1):
            if v_line[r, c] < threshold:
                union(r * cols + c, r * cols + c + 1)
    for r in range(rows - 1):
        for c in range(cols):
            if h_line[r, c] < threshold:
                union(r * cols + c, (r + 1) * cols + c)

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            groups.setdefault(find(r * cols + c), []).append((r, c))
    matrix: list[list[str | None]] = [[None] * cols for _ in range(rows)]
    for cells in groups.values():
        if len(cells) == 1:
            continue
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) == (r0, c0):
                    continue
                matrix[r][c] = "[[V]]" if c == c0 else "[[H]]"
    return matrix
