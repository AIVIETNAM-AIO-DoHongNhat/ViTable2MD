"""Danh sách thí nghiệm: một cấu hình gốc + phần thay đổi của từng thí nghiệm."""


# =============================================================================
# Cấu hình gốc = baseline của ban tổ chức. Chỉ sửa ở đây khi muốn đổi MỌI thí nghiệm.
# =============================================================================

BASE = dict(
    # --- OCR ---
    max_samples=1000,          # số ô dùng để học (0 = toàn bộ)
    epochs=8,
    batch_size=48,
    lr=8e-4,
    scheduler=False,           # OneCycleLR
    augment=False,             # làm bẩn ảnh khi train
    full_alphabet=False,       # bảng chữ cái lấy từ toàn bộ nhãn train
    max_width=384,             # bề rộng tối đa ảnh dòng chữ
    ocr_version=1,             # tăng lên mỗi khi sửa code ảnh hưởng tới mẫu OCR / mô hình

    # --- bố cục bảng ---
    fast_deskew=False,         # chỉnh nghiêng nhanh (đo góc trên ảnh thu nhỏ)
    size_filter=True,          # bộ lọc 7 cột x 15 hàng của baseline
    clean_crops=False,         # bỏ qua ô trống + đảo màu ô nền tối
    merge=False,               # phát hiện ô gộp [[H]]/[[V]]
    multiline=False,           # tách ô nhiều dòng -> <br>
    bold=False,                # phát hiện chữ in đậm
    bold_rows=False,           # (cần bold=True) hàng đầu + hàng cuối của bảng luôn đậm
    bold_header=False,         # (cần bold=True) hàng đầu có [[H]] -> 3 hàng tiêu đề đậm (bảng M2)
    stitch=False,              # nối bảng bị ngắt sang trang sau

    workers=12,                # số tiến trình CPU xử lý ảnh song song
)


# =============================================================================
# Mỗi thí nghiệm = thí nghiệm cha (`parent`) + các công tắc bị đổi.
# Không có `parent` -> đi từ BASE. `note` (tuỳ chọn) = một câu giải thích thêm.
# Thêm thí nghiệm mới: thêm một dòng ở CUỐI, không sửa các dòng đã chạy.
# =============================================================================

EXPERIMENTS = {
    "exp01_baseline": dict(note="baseline gốc"),
    "exp02_no_size_filter": dict(parent="exp01_baseline", size_filter=False,
                                 note="bỏ bộ lọc 7 cột x 15 hàng"),
    "exp03_merge": dict(parent="exp02_no_size_filter", merge=True),
    "exp02b_no_size_filter": dict(parent="exp02_no_size_filter",
                                  note="chạy lại exp02 với mô hình OCR ocr_version=1, làm mốc cho exp03"),
    "exp04_epochs30": dict(parent="exp03_merge", epochs=30,
                           note="OCR 8 epoch chưa hội tụ; 30 epoch hội tụ từ khoảng epoch 20"),
    "exp05_bold": dict(parent="exp04_epochs30", bold=True,
                       note="mọi tài liệu M1/M2 đều có chữ đậm, Bold-F1 đang = 0"),
    "exp06_bold_rows": dict(parent="exp05_bold", bold_rows=True,
                            note="nhãn M1/M2: hàng đầu và hàng cuối 100% in đậm"),
    "exp07_ocr_full": dict(parent="exp06_bold_rows", max_samples=0, epochs=8,
                           note="OCR học mọi mẫu khớp lưới thay vì 1000 mẫu (ô đậm chỉ đọc đúng 71,8%)"),
    "exp08_bold_header": dict(parent="exp07_ocr_full", bold_header=True,
                              note="nhãn M2: đúng 3 hàng đầu đậm khi hàng đầu có [[H]]; Bold-F1 M2 đang 57,2"),
    "exp09_stitch": dict(parent="exp08_bold_header", stitch=True,
                         note="4 tài liệu val tệ nhất đều là bảng vắt 2 trang (48% điểm mất M1+M2)"),
}


def load(name: str) -> dict:
    """Ghép cấu hình đầy đủ của thí nghiệm `name` (BASE -> ... -> cha -> chính nó)."""
    spec = dict(EXPERIMENTS[name])
    parent, note = spec.pop("parent", None), spec.pop("note", "")
    unknown = set(spec) - set(BASE)
    if unknown:
        raise KeyError(f"{name}: công tắc không có trong BASE: {sorted(unknown)}")
    config = load(parent) if parent else {"name": None, "note": None, **BASE}
    changes = ", ".join(f"{key}={value}" for key, value in spec.items() if config[key] != value)
    config.update(spec)
    config["name"] = name
    config["note"] = " | ".join(part for part in (changes, note) if part) or "cấu hình gốc"
    return config
