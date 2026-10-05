# ViTable2MD

Chuyển bảng trong ảnh scan tài liệu tiếng Việt sang Markdown. Đây là bài làm Tác vụ 2 (Trích xuất bảng) của vòng sơ loại OLP AI PTIT 2026.

**Kết quả:** private test 98,96 điểm (điểm chuẩn hoá 100) · tập val M1+M2 đạt 99,20.

## Bài toán

- **Đầu vào:** mỗi tài liệu gồm 1–2 ảnh trang A4, có một hoặc nhiều bảng.
- **Đầu ra:** mỗi tài liệu một tệp `{id}.md` chứa các bảng Markdown. Bảng phải giữ đúng ô gộp (`[[H]]`, `[[V]]`), xuống dòng trong ô (`<br>`), ký tự `\|` và chữ in đậm (`**...**`).
- **Điểm mỗi tài liệu:** `0,9 × TEDS + 0,1 × Bold-F1`. Tài liệu có bảng sai cú pháp hoặc ô gộp sai cấu trúc bị 0 điểm.
- Tập test chỉ có tài liệu mức M1 và M2, tức bảng kẻ viền đầy đủ, nên hệ thống chỉ xử lý bảng kẻ lưới.

Đề bài đầy đủ nằm trong [RULE.md](RULE.md).

## Cách hoạt động

```
ảnh trang ─► tiền xử lý ─► dò lưới bảng ─► dò ô gộp ─► cắt ô, tách dòng
                                                              │
  file ZIP ◄─ kiểm tra hợp lệ ◄─ nối bảng qua trang ◄─ in đậm ◄─ OCR
```

1. **Tiền xử lý** ([grid.py](src/vitable/grid.py)): chuyển ảnh sang thang xám rồi chỉnh nghiêng bằng Hough, lấy trung vị góc của các đường kẻ dài (tối đa 4°). Sau đó tăng tương phản bằng CLAHE và nhị phân hoá bằng Otsu.
2. **Dò lưới bảng** ([grid.py](src/vitable/grid.py)): phép mở hình thái học với kernel dẹt tách riêng đường kẻ ngang và đường kẻ dọc. Các đường kẻ được gom thành từng khối bảng. Toạ độ hàng và cột lấy được bằng cách chiếu mỗi khối lên hai trục.
3. **Dò ô gộp** ([grid.py](src/vitable/grid.py)): kiểm tra từng đoạn biên giữa hai ô kề nhau xem có đường kẻ không. Các ô không bị ngăn cách được gom bằng union-find thành một vùng chữ nhật, rồi đánh dấu `[[H]]`/`[[V]]` giống nhãn.
4. **Cắt ô, tách dòng** ([layout.py](src/vitable/layout.py)): cắt ảnh từng ô (ô gộp thì cắt cả vùng). Ô nhiều dòng được tách theo khoảng trắng ngang, có gộp dấu thanh vào dòng chữ của nó. Mỗi dòng được đưa về cao 40 px.
5. **OCR** ([ocr.py](src/vitable/ocr.py)): mô hình CRNN tự huấn luyện từ đầu, gồm 4 khối Conv-BN-ReLU-Pool, 2 lớp BiGRU và một lớp Linear, học bằng CTC loss. Dữ liệu học được tạo tự động từ tập train: với những trang có lưới dò được khớp đúng số hàng, số cột của nhãn, ảnh từng ô được ghép với chữ tương ứng trong nhãn. Kết quả là 82.814 mẫu, CER 2,6%.
6. **In đậm** ([bold.py](src/vitable/bold.py)): đo độ dày nét chữ bằng distance transform và so với một ngưỡng học từ tập train (chọn ngưỡng cho F1 cao nhất). Hàng có từ 60% ô đậm trở lên thì cả hàng được tô đậm. Thêm hai luật vị trí rút ra từ nhãn M1/M2: hàng đầu và hàng cuối luôn đậm; nếu hàng đầu có `[[H]]` thì 3 hàng đầu đều đậm.
7. **Nối bảng qua trang** ([data.py](src/vitable/data.py)): bảng đầu trang 2 được nối vào bảng cuối trang 1 khi hai bảng cùng số cột và tiêu đề giống nhau từ 80% trở lên (đo bằng Levenshtein). Khi nối, các hàng tiêu đề lặp lại bị bỏ và hàng cuối trang 1 được bỏ in đậm.
8. **Kiểm tra và đóng gói** ([data.py](src/vitable/data.py)): bảng có ô gộp sai cấu trúc thì bỏ hết ô gộp, chấp nhận mất ít điểm thay vì bị 0 điểm cả tài liệu. Tài liệu không dò được bảng nào thì ghi một bảng giả. Cuối cùng ghi các tệp `.md` (xuống dòng kiểu LF) và nén thành ZIP.

Điểm trên máy được tính đúng theo công thức của ban tổ chức trong [metric.py](src/vitable/metric.py), gồm TEDS (tính bằng APTED) và Bold-F1.

## Cấu trúc thư mục

```
ViTable2MD/
├── src/
│   ├── main.ipynb          # quy trình chính: trích mẫu → train OCR → dự đoán → chấm điểm → nộp bài
│   ├── eda.ipynb           # phân tích dữ liệu: quy mô các tập, độ khó M1–M4, ảnh trang, nhãn Markdown
│   ├── baseline.ipynb      # code mẫu của ban tổ chức
│   └── vitable/
│       ├── grid.py         # tiền xử lý, dò lưới, dò ô gộp
│       ├── layout.py       # cắt ô, tách dòng, đo độ dày nét, trích mẫu OCR
│       ├── ocr.py          # CRNN + CTC: huấn luyện và dự đoán
│       ├── bold.py         # luật in đậm
│       ├── data.py         # đọc/ghi Markdown, chia train/val, nối trang, đóng gói
│       ├── metric.py       # TEDS + Bold-F1
│       └── experiments.py  # cấu hình các thí nghiệm
├── study/                  # notebook học xử lý ảnh bảng bằng OpenCV
├── paper/                  # bài báo PubTables-1M
├── data/                   # dữ liệu của ban tổ chức (không đưa lên git)
├── runs/                   # model, cache, dự đoán của từng thí nghiệm (không đưa lên git)
└── report/                 # báo cáo, experiments.csv (không đưa lên git)
```

## Cách chạy

1. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```
   Muốn train OCR bằng GPU thì cài `torch` bản CUDA theo hướng dẫn trên [pytorch.org](https://pytorch.org/get-started/locally/) trước, rồi mới chạy lệnh trên.
2. Đặt dữ liệu vào `data/training_set`, `data/public_test` và `data/private_test`.
3. Mở [src/main.ipynb](src/main.ipynb) và sửa hai dòng:
   - Mục 1: `SPLIT = 'private_test'` (hoặc `'public_test'`).
   - Mục 4: `EXP = experiments.load('exp09_stitch')` (cấu hình tốt nhất).
4. Chạy lần lượt từng mục. File nộp được ghi vào `runs/<tên thí nghiệm>/submission_<SPLIT>.zip`.

Lưu ý:
- Mục 9 sẽ dừng nếu tên thí nghiệm đã có trong `report/experiments.csv`, để tránh ghi trùng. Nếu chỉ cần tạo file nộp, chạy mục 1–8 rồi chuyển thẳng sang mục 11.
- Mẫu OCR (`runs/cache/`) và model (`runs/models/`) được đặt tên theo mã băm của các khoá cấu hình OCR. Vì vậy thí nghiệm nào chỉ đổi phần bố cục sẽ dùng lại model cũ, không phải train lại.
- Muốn thêm thí nghiệm mới thì thêm một dòng vào cuối `EXPERIMENTS` trong [experiments.py](src/vitable/experiments.py), chỉ ghi những công tắc khác với thí nghiệm cha.

## Quá trình cải tiến

Điểm được đo trên 100 tài liệu val, tách từ tập train và giữ nguyên tỉ lệ các mức độ khó. Cột M1+M2 tính trên cùng loại tài liệu với tập test.

| Thí nghiệm | Thay đổi | Score M1+M2 |
| --- | --- | --- |
| exp01_baseline | Baseline của ban tổ chức | 38,12 |
| exp02_no_size_filter | Bỏ bộ lọc chỉ nhận bảng ≤ 7 cột × 15 hàng | 64,82 |
| exp03_merge | Dò ô gộp `[[H]]`/`[[V]]` (mốc chạy lại với cùng OCR: 61,52) | 65,38 |
| exp04_epochs30 | Train OCR 30 epoch (8 epoch chưa hội tụ) | 86,97 |
| exp05_bold | Phát hiện in đậm theo độ dày nét | 91,20 |
| exp06_bold_rows | Hàng đầu và hàng cuối luôn đậm | 92,65 |
| exp07_ocr_full | OCR học toàn bộ mẫu thay vì 1.000 mẫu | 97,05 |
| exp08_bold_header | Hàng đầu có `[[H]]` thì 3 hàng đầu đậm | 98,65 |
| exp09_stitch | Nối bảng vắt qua 2 trang | **99,20** |

## Hạn chế

- Hệ thống chỉ dò được bảng kẻ lưới đầy đủ. Bảng kẻ một phần hoặc không kẻ viền (M3, M4) gần như không có điểm: val M3 đạt 25,58, M4 đạt 1,31. Tập test chỉ có M1/M2 nên điểm thi không bị ảnh hưởng.
- Luật in đậm và luật nối trang được rút ra từ nhãn M1/M2, nên có thể không đúng với dạng tài liệu khác.
- Không dùng mô hình pretrained nào. OCR chỉ học từ chữ in trong tập train.
