# TRÍCH XUẤT BẢNG TỪ ẢNH TÀI LIỆU

## 1. Mô tả bài toán

Phần lớn thông tin có giá trị trong tài liệu hành chính nằm trong bảng: bảng lương, biểu thống kê, danh mục hàng hóa, báo cáo doanh thu theo khu vực. Khi tài liệu chỉ còn tồn tại dưới dạng ảnh chụp hoặc bản scan, cấu trúc đó biến mất — máy chỉ nhìn thấy những nét mực trên nền giấy. Muốn tra cứu, tổng hợp hay đưa số liệu vào hệ thống, trước hết phải dựng lại được đúng cái bảng ban đầu.

Khó khăn không nằm ở việc đọc chữ. Một bảng có thể không kẻ viền, tiêu đề gộp ngang nhiều cột, ô dữ liệu kéo dọc qua nhiều hàng, nội dung xuống dòng ngay bên trong một ô, hoặc chạy dài sang trang thứ hai. Chỉ cần nhận nhầm một ranh giới cột, mọi ô phía sau đều lệch theo và cả bảng trở nên vô nghĩa.

Trong tác vụ **Trích xuất bảng**, mỗi mẫu dữ liệu là một tài liệu gồm một đến hai ảnh trang A4 chứa một hoặc nhiều bảng. Các đội thi cần chuyển toàn bộ bảng trong tài liệu sang định dạng Markdown mở rộng, bảo toàn thứ tự bảng, nội dung văn bản, số hàng và số cột, ô gộp, ngắt dòng trong ô và các đoạn chữ in đậm. Kết quả của mỗi tài liệu được lưu thành một tệp Markdown riêng, tên tệp trùng với id của tài liệu trong `manifest.jsonl`.

## 2. Mô tả dữ liệu

Dữ liệu được chia thành ba tập:

| Tập dữ liệu | Số tài liệu | Số ảnh trang | Nhãn | Mục đích |
| --- | --- | --- | --- | --- |
| training_set | 1.100 | 1.375 | Có | Huấn luyện và kiểm tra chương trình |
| public_test | 50 | 52 | Không công bố | Đánh giá công khai |
| private_test | 80 | 86 | Không công bố | Đánh giá và xếp hạng cuối cùng |

Mỗi tài liệu gồm một hoặc hai trang. Ảnh trang là JPEG khổ dọc, độ phân giải thay đổi theo tài liệu, nằm trong khoảng từ 1600×2263 đến 2000×2828 pixel.

Tập huấn luyện được gán bốn mức độ khó tăng dần qua trường `difficulty`:

| Mức | Số tài liệu | Đặc điểm |
| --- | --- | --- |
| M1 | 325 | Một bảng kẻ viền đầy đủ, một trang, tối đa 7 cột, không có ô gộp |
| M2 | 330 | Kẻ viền đầy đủ, tối đa 9 cột, có ô gộp và ô nhiều dòng, một số tài liệu hai trang |
| M3 | 280 | Tối đa 12 cột, tối đa 3 bảng, xuất hiện bảng kẻ viền một phần hoặc không kẻ viền, có ký tự `\|` trong nội dung ô |
| M4 | 165 | Tối đa 14 cột, đa số hai trang, bảng dài vắt qua trang, không có bảng nào kẻ viền đầy đủ |

Hai tập kiểm tra chỉ gồm tài liệu mức **M1** và **M2**, và không kèm trường `difficulty` trong `manifest.jsonl`. Kết quả chấm được trả về kèm điểm tách riêng theo từng mức.

Dữ liệu nằm tại `/home/user/TACVU2`:

```text
TACVU2/
|-- baseline_TACVU2.ipynb           # Code mẫu tham khảo
|-- data/
    |-- training_set/
    |   |-- manifest.jsonl          # Thông tin 1.100 tài liệu
    |   |-- images/                 # 1.375 ảnh trang
    |   |-- labels/                 # Nhãn Markdown của cả tài liệu
    |   |-- page_labels/            # Nhãn Markdown tách theo từng trang
    |-- public_test/
    |   |-- manifest.jsonl          # Thông tin 50 tài liệu
    |   |-- images/                 # 52 ảnh trang
    |-- private_test.zip            # Được bảo vệ bằng mật khẩu
```

`private_test.zip` *chỉ được công bố mật khẩu khi bắt đầu Giai đoạn kiểm tra bí mật*. Sau khi giải nén, tập private test có cấu trúc giống hệt `public_test`.

### Tệp manifest.jsonl

Mỗi dòng là một JSON object mô tả một tài liệu:

```json
{"id": "public_test-00000", "image_paths": ["images/public_test-00000_p01.jpg"], "page_count": 1, "split": "public_test"}
```

- `id`: mã định danh duy nhất của tài liệu, đồng thời là tên tệp kết quả cần nộp.
- `image_paths`: danh sách đường dẫn ảnh trang *tương đối so với thư mục của tập dữ liệu*, xếp theo đúng thứ tự trang.
- `page_count`: số trang của tài liệu.
- `split`: tên tập dữ liệu.

Riêng `training_set` có thêm bốn trường:
- `label_path`: nhãn Markdown của toàn tài liệu, ví dụ `labels/train-00000.md`.
- `page_label_paths`: nhãn Markdown tách riêng theo từng trang.
- `difficulty`: mức độ khó, nhận giá trị `M1`, `M2`, `M3` hoặc `M4`.
- `attributes`: đặc điểm của tài liệu, gồm `border_mode`, `table_count`, `page_count`, `max_columns`, `max_data_rows`, `has_bold`, `has_merge`, `has_multiline`, `has_escaped_pipe`, `spans_two_pages` và `long_table_crosses_pages`.

### Quy ước nhãn Markdown

Nhãn sử dụng Markdown mở rộng với các quy ước bắt buộc sau:
- `[[H]]`: ô tiếp nối của một vùng gộp theo chiều ngang.
- `[[V]]`: ô tiếp nối của một vùng gộp theo chiều dọc.
- `<br>`: xuống dòng trong cùng một ô.
- `\|`: ký tự `|` xuất hiện trong nội dung ô.
- `**nội dung**`: nội dung in đậm.

Mỗi bảng phải có hàng phân cách Markdown gồm các ô `---`. Nếu một tài liệu có nhiều bảng, các bảng phải được phân cách bằng đúng một dòng trống.

Ví dụ một bảng ba cột có đủ các quy ước trên:

```markdown
| **Khu vực** | **Kế hoạch** | **Thực hiện** |
| --- | --- | --- |
| **Tổng hợp quý I** | [[H]] | [[H]] |
| Miền Bắc | 2.480 | 3.340 |
| [[V]] | 1.490 | 2.190 |
| Miền Nam | 4.680<br>(đã điều chỉnh) | 4.180 |
```

Hàng thứ ba là một tiêu đề gộp ngang cả ba cột. Ô `[[V]]` ở hàng thứ năm cho biết ô Miền Bắc phía trên kéo dài xuống hàng này. Ô cuối cùng của hàng thứ sáu chứa hai dòng văn bản trong cùng một ô.

## 3. Cấu trúc file nộp

Các đội thi cần nộp **01 file ZIP**. Trong ZIP, mỗi tài liệu của tập kiểm tra phải có đúng một tệp `{id}.md`:

```text
submission.zip
|-- public_test-00000.md
|-- public_test-00001.md
|-- ...
```

Đối với giai đoạn private test, tên tệp sử dụng id trong `private_test/manifest.jsonl`.

Trình chấm gom mọi tệp `{id}.md` ở bất kỳ vị trí nào trong ZIP và bỏ qua các tệp khác, nên nộp kèm notebook hay mã nguồn cũng không sao. Điều bắt buộc là **tên tệp phải đúng `{id}.md`**.

Mỗi tệp Markdown chỉ được chứa các bảng kết quả, không chứa code fence hoặc văn bản bên ngoài bảng. Tài liệu bị thiếu tệp, tệp rỗng, bảng sai cú pháp hoặc cấu trúc ô gộp không hợp lệ sẽ nhận 0 điểm cho tài liệu đó.

## 4. Thang đo đánh giá

Điểm của mỗi tài liệu được tính theo công thức:

$$\text{Document\_Score} = 0.90 \times \text{TEDS} + 0.10 \times \text{Bold-F1}$$

Trong đó:
- **TEDS** đánh giá mức độ tương đồng giữa cấu trúc và nội dung của bảng dự đoán với nhãn chuẩn.
- **Bold-F1** đánh giá độ chính xác của các nội dung được đánh dấu in đậm.

Điểm của bài nộp là trung bình điểm của toàn bộ tài liệu, quy đổi về thang 100:

$$\text{Score} = 100 \times \frac{1}{N} \sum_{i=1}^{N} \text{Document\_Score}_i$$

Điểm hiển thị trên bảng xếp hạng được chuẩn hoá từ $\text{Score}$ theo công thức:

$$\text{Final\_Score} = \begin{cases} 100 \times \frac{\text{Score} - \text{Min}}{\text{Max} - \text{Min}}, & \text{nếu } \text{Score} > \text{Min} \\ 0, & \text{nếu } \text{Score} \le \text{Min} \end{cases}$$

Trong đó:
- $\text{Min}$: ngưỡng tối thiểu do Ban tổ chức quy định cho tác vụ.
- $\text{Max}$: điểm cao nhất hiện có trên bảng xếp hạng của tác vụ.

Mỗi khi $\text{Max}$ thay đổi, điểm chuẩn hoá của toàn bộ đội thi được tính lại theo công thức trên.

Tệp thiếu, rỗng hoặc không đúng cú pháp nhận điểm 0 cho tài liệu tương ứng. Kết quả trên public_test được dùng để phản hồi trong thời gian thi; thứ hạng cuối cùng được xác định bằng private_test.




------------------------------------------------------

# [REOPEN] Vòng sơ loại OLP AI PTIT 2026 - Ca 1 — Rules

> Nguồn: https://aichallenge.ptit.edu.vn/competitions/126/ (tab **Rules**)

Vòng hiện tại chỉ thi trên dữ liệu private test. Link tải [Google Drive](https://drive.google.com/file/d/1eDaUwV8IX8Rw7Oh2ryAAjA8_FYZcWZ-B/view). Pass giải nén `private_test.zip`: `225554`.

Danh sách pretrained models được phép:

| Nhóm | Model |
| --- | --- |
| Hugging Face | microsoft/trocr-base-printed |
| Hugging Face | microsoft/table-transformer-detection |
| Hugging Face | microsoft/table-transformer-structure-recognition-v1.1-all |
| Hugging Face | stepfun-ai/GOT-OCR-2.0-hf |
| EasyOCR | craft |
| EasyOCR | latin_g2 |
| EasyOCR | english_g2 |
| VietOCR | vgg_transformer |
| VietOCR | vgg_seq2seq |
| Torchvision | densenet121 |
| Torchvision | efficientnet_b0 |
| Torchvision | efficientnet_b1 |
| Torchvision | efficientnet_b2 |
| Torchvision | googlenet |
| Torchvision | mnasnet0_5 |
| Torchvision | mnasnet0_75 |
| Torchvision | mnasnet1_0 |
| Torchvision | mobilenet_v2 |
| Torchvision | mobilenet_v3_large |
| Torchvision | mobilenet_v3_small |
| Torchvision | regnet_x_400mf |
| Torchvision | regnet_x_800mf |
| Torchvision | regnet_y_400mf |
| Torchvision | regnet_y_800mf |
| Torchvision | resnet18 |
| Torchvision | resnet34 |
| Torchvision | shufflenet_v2_x0_5 |
| Torchvision | shufflenet_v2_x1_0 |
| Torchvision | squeezenet1_0 |
| Torchvision | squeezenet1_1 |
| Torchvision | vgg19_bn |

## HƯỚNG DẪN CHUNG

### 1. Dữ liệu và mã nguồn mẫu

Đề thi gồm có 2 tác vụ về xử lý ngôn ngữ tự nhiên và xử lý ảnh.

- Mỗi tác vụ đều đi kèm với **một notebook `.ipynb` mẫu**, trong đó đã bao gồm:
  - Mô tả tác vụ.
  - Mã nguồn của **mô hình mẫu tham khảo**.
- Bộ dữ liệu **tập huấn luyện (`training_set`)** và **tập kiểm tra công khai (`public_test`)** đã được lưu sẵn trong môi trường notebook và có thể truy cập bất kỳ lúc nào.
- Tập dữ liệu **kiểm tra riêng (`private_test`)** sẽ chỉ được mở khóa trong **giờ cuối của cuộc thi**. Thí sinh cần mật khẩu (do BTC cung cấp) để truy cập vào tệp này. Việc truy cập trước giờ quy định hoặc sửa nội dung test đều bị coi là vi phạm quy chế.

### 2. Sử dụng mô hình ngôn ngữ lớn (LLM)

**Truy cập:** Thí sinh có thể truy cập LLM thông qua hệ thống thi. Mô hình sử dụng là Gemini 3.5 Flash. Tổng giới hạn Token cho cả input và output trong 1 phiên trò chuyện là 2000. Không giới hạn số phiên trò chuyện.

**Thời gian sử dụng:** LLM chỉ được sử dụng trong **3 tiếng đầu** của kỳ thi (trong tổng thời gian thi 4 tiếng).

**Hạn chế:**

- Sau 3 tiếng đầu, quyền truy cập LLM sẽ tự động bị khóa.
- Thí sinh chỉ được dùng LLM để hỗ trợ tư duy và phát triển mô hình.

### 3. Nộp bài và đánh giá kết quả

**Quy trình thi:**

- **Giờ 0–3:** Làm bài, được phép dùng LLM và 2 tập dữ liệu `training_set` và `public_test`.
- **Giờ thứ 4:** Không còn quyền truy cập LLM. BTC công bố password cho tập dữ liệu test (`private_test`). Thí sinh sử dụng mô hình của mình để tạo ra file kết quả.

**Yêu cầu khi tạo file kết quả:**

- Không được chỉnh sửa thủ công dữ liệu đầu vào/đầu ra.
- Quá trình chạy sẽ được ghi log toàn bộ để đảm bảo minh bạch.

**Lưu ý:** Trước khi hết giờ, lưu **kết quả tốt nhất của từng bài** vào thư mục **`/home/user/FINAL`**. Trong thư mục này gồm **2 thư mục con: `TACVU1` và `TACVU2`**, mỗi thư mục chứa:

- **Model tốt nhất**: ví dụ `best_model.pt`
- **File submission kết quả tốt nhất trong Giai đoạn 2 - Private Test**: ví dụ `private_submission.zip`
- **Notebook chứa toàn bộ quy trình huấn luyện, fine-tune và sinh kết quả**: ví dụ `generate_result.ipynb`
- **Các file phụ trợ liên quan**: ví dụ `utils.py`, `config.yaml`

**Mã nguồn khi chạy phải sinh ra đúng kết quả dự đoán cho tập dữ liệu `private_test`.**

Cấu trúc thư mục minh hoạ:

```
/home/user/FINAL/
|-- TACVU1/
|   |-- best_model.pt
|   |-- private_submission.zip
|   |-- generate_result.ipynb
|   |-- utils.py
|   |-- config.yaml
|
|-- TACVU2/
    |-- best_model.pt
    |-- private_submission.zip
    |-- generate_result.ipynb
    |-- utils.py
    |-- config.yaml
```

Chi tiết yêu cầu nộp bài vui lòng đọc hướng dẫn trong từng tác vụ.

**Giới hạn số lượt nộp:**

- Giai đoạn 1 - Public Test: tối đa **20 lượt nộp**, tính chung cho cả hai tác vụ.
- Giai đoạn 2 - Private Test: tối đa **10 lượt nộp**, tính chung cho cả hai tác vụ.

**Đánh giá:**

- Kết quả chung cuộc sẽ được dựa trên kết quả của vòng kiểm tra bí mật trên tập `private_test` của Giai đoạn 2 - Private Test.
- Các thí sinh phạm quy sẽ không được chấm điểm và bị loại khỏi việc xét kết quả.
- Với mỗi tác vụ, mỗi lượt nộp bài sẽ được chấm điểm theo công thức trong tác vụ, sau đó chuẩn hoá dựa trên điểm trên scoreboard của tác vụ đó như sau:
  - Gọi $Min$ là ngưỡng tối thiểu do Ban tổ chức quy định cho tác vụ.
  - Gọi $Max$ là điểm cao nhất trong tất cả các điểm hiện tại trên scoreboard của tác vụ.
  - Gọi $Score$ là điểm của bài vừa được chấm theo công thức trong tác vụ.
  - Điểm hiện lên scoreboard của thí sinh là:

    $$\frac{Score - Min}{Max - Min} \times 100\%$$

  - Bài nộp có $Score \le Min$ nhận 0 điểm trên scoreboard.
  - Mỗi khi điểm $Max$ thay đổi, điểm của toàn bộ thí sinh trên scoreboard cũng sẽ được cập nhật theo công thức trên.
- Điểm tổng cộng của thí sinh là tổng điểm hai bài sau khi đã được chuẩn hoá trên scoreboard.

#### Quy định về Mô hình tiền huấn luyện (Pretrained Models)

Ngoài mô hình cơ sở do Ban tổ chức cung cấp, các đội thi **được phép** sử dụng các kiến trúc và trọng số tiền huấn luyện nằm trong thư mục **`/cache_models`**.

#### Môi trường và thư viện

Mỗi đội thi được cấp một JupyterLab riêng với 1 GPU NVIDIA H100 MIG (10 GB VRAM), 32 GB RAM và 200 GB dung lượng lưu trữ. Môi trường **không có kết nối Internet** trong suốt thời gian thi. Các thư viện cho phép đã được cài sẵn.