"""Nhận dạng chữ trong ô bằng CRNN + CTC: dữ liệu, mô hình, huấn luyện, dự đoán."""

from dataclasses import dataclass
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .layout import CROP_HEIGHT, MAX_CROP_WIDTH, decode_png


# =============================================================================
# PHẦN 1: LẤY TỪ BASELINE (mục 5 'Mô hình nhận dạng ô' của baseline.ipynb)
# =============================================================================

@dataclass
class CellSample:
    """Một mẫu huấn luyện OCR: ảnh dòng chữ (PNG) + chữ đúng."""
    image: bytes      # crop đã normalize_crop, mã hoá PNG cho nhẹ RAM
    text: str


# [baseline, đã sửa: fit() bỏ ký tự xuống dòng]
class Alphabet:
    """Bảng chữ cái: đổi ký tự ↔ số (số 0 dành cho ký tự "trống" của CTC)."""
    def __init__(self, characters: list[str]):
        """Gán mỗi ký tự một số, bắt đầu từ 1."""
        self.characters = characters
        self.to_id = {character: index + 1 for index, character in enumerate(characters)}

    @classmethod
    def fit(cls, texts: list[str]) -> "Alphabet":
        """Tạo bảng chữ cái từ mọi ký tự xuất hiện trong danh sách chữ."""
        return cls(sorted({character for text in texts for character in text if character != "\n"}))

    def encode(self, text: str) -> list[int]:
        """Đổi chuỗi chữ thành dãy số (bỏ ký tự không có trong bảng)."""
        return [self.to_id[character] for character in text if character in self.to_id]

    def decode(self, indices: list[int]) -> str:
        """Giải mã CTC: bỏ ký tự trống, gộp ký tự lặp liền nhau, rồi đổi số thành chữ."""
        result: list[str] = []
        previous = -1
        for index in indices:
            if index and index != previous and index <= len(self.characters):
                result.append(self.characters[index - 1])
            previous = index
        return "".join(result)


# [baseline, đã sửa: thêm tuỳ chọn augment]
class CellDataset(Dataset):
    """Dataset PyTorch: trả về (ảnh tensor, dãy số của chữ) cho từng mẫu."""
    def __init__(self, samples: list[CellSample], alphabet: Alphabet, use_augment: bool = False, seed: int = 0):
        """Lưu danh sách mẫu, bảng chữ cái và tuỳ chọn augment."""
        self.samples = samples
        self.alphabet = alphabet
        self.use_augment = use_augment
        self.rng = random.Random(seed)

    def __len__(self):
        """Số mẫu trong dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Lấy mẫu thứ index: giải nén ảnh, (augment), đảo màu về 0..1, mã hoá chữ."""
        sample = self.samples[index]
        image = decode_png(sample.image)
        if self.use_augment:
            image = augment(image, self.rng)
        image = torch.from_numpy(1.0 - image.astype(np.float32) / 255.0).unsqueeze(0)
        target = torch.tensor(self.alphabet.encode(sample.text), dtype=torch.long)
        return image, target


# [baseline, đã sửa: dùng hằng CROP_HEIGHT]
def collate_cells(batch):
    """Ghép nhiều mẫu thành một batch: đệm các ảnh về cùng bề rộng."""
    max_width = max(image.shape[-1] for image, _ in batch)
    images = torch.zeros((len(batch), 1, CROP_HEIGHT, max_width), dtype=torch.float32)
    targets, lengths, widths = [], [], []
    for row, (image, target) in enumerate(batch):
        images[row, :, :, : image.shape[-1]] = image
        targets.append(target)
        lengths.append(len(target))
        widths.append(image.shape[-1])
    return images, torch.cat(targets), torch.tensor(lengths), torch.tensor(widths)


class GridCRNN(nn.Module):
    """Mô hình CRNN: CNN trích đặc trưng → GRU hai chiều đọc theo chiều ngang → phân lớp ký tự."""

    def __init__(self, classes: int):
        """Khai báo 4 khối Conv-BN-ReLU-Pool, 2 lớp GRU hai chiều và lớp Linear cuối."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(192, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.sequence = nn.GRU(256, 192, num_layers=2, bidirectional=True, batch_first=True, dropout=0.15)
        self.classifier = nn.Linear(384, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Ảnh → xác suất (log) của từng ký tự tại mỗi vị trí theo chiều ngang."""
        features = self.features(images).mean(dim=2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence).log_softmax(dim=-1)


# [baseline, đã sửa: chia lô, sắp ảnh theo bề rộng để ít padding]
@torch.inference_mode()
def recognize_images(model, alphabet: Alphabet, images: list[np.ndarray], device: torch.device,
                     batch_size: int = 256) -> list[str]:
    """Đọc chữ cho nhiều ảnh dòng cùng lúc (theo lô trên GPU), giải mã tham lam."""
    results: list[str] = []
    # Gom các ảnh có bề rộng gần nhau để ít phải padding.
    order = sorted(range(len(images)), key=lambda index: images[index].shape[1])
    texts: dict[int, str] = {}
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        width = max(images[index].shape[1] for index in chunk)
        batch = torch.zeros((len(chunk), 1, CROP_HEIGHT, width), dtype=torch.float32)
        for row, index in enumerate(chunk):
            image = images[index]
            batch[row, 0, :, : image.shape[1]] = torch.from_numpy(1.0 - image.astype(np.float32) / 255.0)
        best_ids = model(batch.to(device)).argmax(dim=-1).cpu().tolist()
        for index, ids in zip(chunk, best_ids):
            texts[index] = alphabet.decode(ids)
    results = [texts[index] for index in range(len(images))]
    return results


# [baseline, đã sửa: nhận đường dẫn file thay vì thư mục]
def load_recognizer(path, device: torch.device):
    """Nạp mô hình OCR và bảng chữ cái từ file .pt đã lưu."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    alphabet = Alphabet(payload["characters"])
    model = GridCRNN(len(alphabet.characters) + 1)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), alphabet


# =============================================================================
# =============================================================================

def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Làm bẩn ảnh ngẫu nhiên (co giãn, mờ, nhiễu...) khi train để mô hình chịu nhiễu tốt hơn."""
    height, width = image.shape
    if rng.random() < 0.5:   # co/giãn ngang
        new_width = max(4, min(MAX_CROP_WIDTH, round(width * rng.uniform(0.85, 1.15))))
        image = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:   # nét mảnh/dày hơn
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel) if rng.random() < 0.5 else cv2.dilate(image, kernel)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:   # đổi độ tương phản / độ sáng
        alpha, beta = rng.uniform(0.7, 1.1), rng.uniform(-20, 30)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 10), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def character_error_rate(predictions: list[str], targets: list[str]) -> float:
    """CER = tổng số ký tự sai / tổng số ký tự đúng (càng thấp càng tốt)."""
    import Levenshtein
    errors = sum(Levenshtein.distance(pred, true) for pred, true in zip(predictions, targets))
    return errors / max(1, sum(len(true) for true in targets))


# [mới, viết lại từ vòng lặp huấn luyện ở mục 6 của baseline; thêm CER, scheduler, giữ checkpoint tốt nhất]
def train_recognizer(train_samples: list[CellSample], val_samples: list[CellSample], alphabet: Alphabet,
                     device: torch.device, epochs: int, batch_size: int, lr: float, seed: int,
                     use_augment: bool = False, use_scheduler: bool = False, log=print):
    """Huấn luyện CRNN bằng CTC loss; sau mỗi epoch đo CER và giữ bản tốt nhất."""
    torch.manual_seed(seed)
    dataset = CellDataset(train_samples, alphabet, use_augment=use_augment, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        collate_fn=collate_cells, generator=torch.Generator().manual_seed(seed))
    model = GridCRNN(len(alphabet.characters) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(loader),
                                                     pct_start=0.15)
                 if use_scheduler else None)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    val_images = [decode_png(sample.image) for sample in val_samples]
    val_texts = [sample.text for sample in val_samples]
    best_cer, best_state, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count, started = 0.0, 0, time.time()
        for images, targets, target_lengths, widths in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            input_lengths = torch.clamp((widths + 3) // 4, max=logits.shape[1]).to(dtype=torch.long)
            loss = criterion(logits.transpose(0, 1), targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach().cpu()) * len(images)
            count += len(images)
        model.eval()
        # Xả bộ nhớ đệm GPU trước/sau khi đo CER (không đổi kết quả). Trên Windows, khi VRAM đầy driver
        # mượn RAM hệ thống thay vì báo lỗi, nên PyTorch không tự xả đệm -> huấn luyện chậm đi hàng chục lần.
        torch.cuda.empty_cache()
        cer = character_error_rate(recognize_images(model, alphabet, val_images, device), val_texts) \
            if val_images else float("nan")
        torch.cuda.empty_cache()
        history.append({"epoch": epoch, "loss": total / max(1, count), "val_cer": cer})
        log(f"[ocr] epoch {epoch}/{epochs} | loss={total / max(1, count):.4f} | val CER={cer:.4f} "
            f"| {time.time() - started:.1f}s")
        if not val_images or cer < best_cer:
            best_cer = cer
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.cuda.empty_cache()   # trả VRAM cho bước đọc chữ (mục 8) sau khi huấn luyện
    return model.eval(), history


def save_recognizer(path, model, alphabet: Alphabet, extra: dict) -> None:
    """Lưu trọng số mô hình + bảng chữ cái + thông tin huấn luyện ra file .pt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "characters": alphabet.characters, **extra}, path)
