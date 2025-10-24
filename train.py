import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
import os
import multiprocessing
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler
import torch.amp
import traceback


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout2d(p=dropout_prob))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels, dropout_prob=0.0)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        dp_low = 0.1
        dp_med = 0.2
        dp_high = 0.3

        self.inc = DoubleConv(n_channels, 64, dropout_prob=dp_low)
        self.down1 = Down(64, 128, dropout_prob=dp_low)
        self.down2 = Down(128, 256, dropout_prob=dp_med)
        self.down3 = Down(256, 512, dropout_prob=dp_med)
        self.down4 = Down(512, 1024, dropout_prob=dp_high)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class BratsDataset(Dataset):
    def __init__(self, patient_dirs, transform=None, cache_data=True):
        super().__init__()
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.cache_data = cache_data
        self.image_paths = []
        self.mask_paths = []
        self.cached_images = []
        self.cached_masks = []
        self.num_slices_per_scan = 155
        valid_patients = 0

        print("Veri seti taraniyor ve dosyalar dogrulaniyor...")
        original_indices = []
        for i, patient_dir in enumerate(
            tqdm(self.patient_dirs, desc="Hastalar Taraniyor")
        ):
            flair = list(patient_dir.glob("*_flair.nii"))
            seg = list(patient_dir.glob("*_seg.nii")) + list(
                patient_dir.glob("*Segm.nii")
            )
            if len(flair) == 1 and len(seg) == 1:
                self.image_paths.append(flair[0])
                self.mask_paths.append(seg[0])
                original_indices.append(i)
                valid_patients += 1

        print(
            f"Dogrulama tamamlandi. Toplam {valid_patients} adet gecerli hasta bulundu."
        )

        if self.cache_data:
            print("Veriler bellege yukleniyor (bu biraz surebilir)...")

            temp_cached_images = []
            temp_cached_masks = []
            temp_valid_image_paths = []
            temp_valid_mask_paths = []

            for img_p, msk_p in tqdm(
                zip(self.image_paths, self.mask_paths),
                total=len(self.image_paths),
                desc="Caching Data",
            ):
                try:
                    img_3d = nib.load(img_p).get_fdata(dtype=np.float32)

                    msk_3d_raw = nib.load(msk_p).get_fdata()
                    msk_3d = msk_3d_raw.astype(np.uint8)

                    temp_cached_images.append(img_3d)
                    temp_cached_masks.append(msk_3d)
                    temp_valid_image_paths.append(img_p)
                    temp_valid_mask_paths.append(msk_p)
                except Exception as e:
                    print(
                        f"Uyari: {img_p} veya {msk_p} yuklenirken hata: {e}. Bu hasta cache'lenemedi ve atlanacak."
                    )

            self.cached_images = temp_cached_images
            self.cached_masks = temp_cached_masks
            self.image_paths = temp_valid_image_paths
            self.mask_paths = temp_valid_mask_paths
            print(
                f"Veriler bellege yuklendi. Toplam {len(self.cached_images)} hasta cache'lendi."
            )

        self.num_valid_patients = len(self.image_paths)

    def __len__(self):

        return self.num_valid_patients * self.num_slices_per_scan

    def __getitem__(self, idx):

        if self.num_valid_patients == 0:
            raise IndexError("Dataset'te gecerli hasta bulunamadi.")
        p_idx = (idx // self.num_slices_per_scan) % self.num_valid_patients
        s_idx = idx % self.num_slices_per_scan

        try:
            if self.cache_data:

                img_3d = self.cached_images[p_idx]
                msk_3d = self.cached_masks[p_idx]
            else:
                img_p = self.image_paths[p_idx]
                msk_p = self.mask_paths[p_idx]
                img_3d = nib.load(img_p).get_fdata(dtype=np.float32)

                msk_3d_raw = nib.load(msk_p).get_fdata()
                msk_3d = msk_3d_raw.astype(np.uint8)

            img_s = img_3d[:, :, s_idx]

            msk_s = msk_3d[:, :, s_idx].astype(np.uint8)
            msk_s[msk_s == 4] = 3

            p1, p99 = (
                np.percentile(img_s[img_s > 0], [1, 99])
                if np.any(img_s > 0)
                else (0, 1)
            )
            img_s = np.clip(img_s, p1, p99)

            img_s = img_s.astype(np.float32)

            if self.transform:

                aug = self.transform(image=img_s, mask=msk_s)
                img_s = aug["image"]
                msk_s = aug["mask"]

            if not isinstance(img_s, torch.Tensor):
                img_t = torch.from_numpy(img_s.copy()).float().unsqueeze(0)
            else:
                img_t = img_s

            if not isinstance(msk_s, torch.Tensor):
                msk_t = torch.from_numpy(msk_s.copy()).long()
            else:
                msk_t = msk_s.long()

            return {"image": img_t, "mask": msk_t}

        except Exception as e:
            print(
                f"Hata olustu __getitem__ index {idx} (Hasta {p_idx}, Slice {s_idx}): {e}"
            )

            img_shape = (1, 240, 240)
            mask_shape = (240, 240)
            return {
                "image": torch.zeros(img_shape, dtype=torch.float32),
                "mask": torch.zeros(mask_shape, dtype=torch.long),
            }


class DiceLoss(nn.Module):
    def __init__(self, n_classes, epsilon=1e-6, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.nc = n_classes
        self.eps = epsilon
        self.smooth = smooth

    def forward(self, pred_l, targ):
        pred_s = torch.softmax(pred_l, dim=1)
        targ_oh = (
            torch.nn.functional.one_hot(targ, num_classes=self.nc)
            .permute(0, 3, 1, 2)
            .float()
        )
        dims = (0, 2, 3)
        inter = torch.sum(pred_s * targ_oh, dims)
        card_p = torch.sum(pred_s, dims)
        card_t = torch.sum(targ_oh, dims)
        dice = (2.0 * inter + self.smooth) / (card_p + card_t + self.smooth)
        dice_tumor = dice[1:]
        mean_dice = dice_tumor.mean()
        return 1.0 - mean_dice


class DiceCELoss(nn.Module):
    def __init__(self, n_classes, ce_weight=0.5, dice_weight=0.5):
        super(DiceCELoss, self).__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=n_classes, smooth=1.0)

    def forward(self, preds_logits, targets):
        loss_ce = self.ce_loss(preds_logits, targets)
        loss_dice = self.dice_loss(preds_logits, targets)
        combined_loss = (self.ce_weight * loss_ce) + (self.dice_weight * loss_dice)
        return combined_loss


def calculate_dice_score(preds_logits, targets, n_classes=4, epsilon=1e-6):
    preds = torch.argmax(preds_logits, dim=1)
    preds_one_hot = (
        torch.nn.functional.one_hot(preds, num_classes=n_classes)
        .permute(0, 3, 1, 2)
        .float()
    )
    targets_one_hot = (
        torch.nn.functional.one_hot(targets, num_classes=n_classes)
        .permute(0, 3, 1, 2)
        .float()
    )
    dims = (0, 2, 3)
    intersection = torch.sum(preds_one_hot * targets_one_hot, dims)
    cardinality_preds = torch.sum(preds_one_hot, dims)
    cardinality_targets = torch.sum(targets_one_hot, dims)
    dice_score = (2.0 * intersection + epsilon) / (
        cardinality_preds + cardinality_targets + epsilon
    )
    return dice_score


import csv
from datetime import datetime


def main():
    try:

        data_folder = Path("./data/MICCAI_BraTS2020_TrainingData/")
        all_patient_dirs = [d for d in data_folder.iterdir() if d.is_dir()]
        train_dirs, val_dirs = train_test_split(
            all_patient_dirs, test_size=0.2, random_state=42
        )
        print(f"Toplam {len(all_patient_dirs)} hasta bulundu.")
        print(f"{len(train_dirs)} egitim, {len(val_dirs)} validasyon.")

        train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.ElasticTransform(
                    p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                # A.Normalize(mean=(0.0,), std=(1.0,)), # Dataset icinde yapiliyor
                ToTensorV2(),
            ]
        )
        val_transform = A.Compose(
            [
                ToTensorV2(),
            ]
        )

        print("\n--- Veri Setleri Yukleniyor (Caching Aktif) ---")
        train_dataset = BratsDataset(
            patient_dirs=train_dirs, transform=train_transform, cache_data=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True if os.name != "nt" else False,
        )
        validation_dataset = BratsDataset(
            patient_dirs=val_dirs, transform=val_transform, cache_data=False
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True if os.name != "nt" else False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        model = UNet(n_channels=1, n_classes=4).to(device)
        criterion = DiceCELoss(n_classes=4, ce_weight=0.5, dice_weight=0.5)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        use_amp = True
        scaler = torch.amp.GradScaler(enabled=use_amp)
        max_grad_norm = 1.0

        num_epochs = 50
        CHECKPOINT_PATH = "best_model_bn_do_dice.pth"
        start_epoch = 0
        best_val_dice = 0.0
        epochs_no_improve = 0
        patience = 10

        log_file_path = "training_log_bn_do.csv"
        if start_epoch == 0 and not Path(log_file_path).exists():
            with open(log_file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["Epoch", "LearningRate", "TrainLoss", "ValLoss", "ValDice"]
                )

        if Path(CHECKPOINT_PATH).exists():
            print(f"\nCheckpoint dosyasi {CHECKPOINT_PATH} bulundu. Yukleniyor...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model_state_dict"])

            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                start_epoch = checkpoint["epoch"] + 1
                best_val_dice = checkpoint.get("best_val_dice", 0.0)
                print(
                    f"Model ve optimizer durumu yuklendi. Egitime Epoch {start_epoch + 1}'den devam ediliyor..."
                )
                print(f"Onceki en iyi Val Dice: {best_val_dice:.4f}")
            except KeyError:
                print(
                    "Uyari: Checkpoint dosyasinda optimizer durumu bulunamadi. Optimizer sifirdan baslatiliyor."
                )
                start_epoch = checkpoint.get("epoch", -1) + 1
                best_val_dice = checkpoint.get("best_val_dice", 0.0)
                print(
                    f"Model durumu yuklendi. Egitime Epoch {start_epoch + 1}'den devam ediliyor..."
                )

        else:
            print(f"\nCheckpoint dosyasi bulunamadi. Egitime Epoch 1'den baslaniyor.")
            start_epoch = 0
            best_val_dice = 0.0

        print(f"\nEgitim {start_epoch + 1}. epoch'tan basliyor...")
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_train_loss = 0
            train_loop = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Egitim]",
                leave=False,
            )
            for batch in train_loop:
                if batch is None:
                    continue
                images = batch["image"].to(device=device)
                true_masks = batch["mask"].to(device=device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    predicted_masks = model(images)
                    loss = criterion(predicted_masks, true_masks)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"Uyari: NaN/Inf Loss tespit edildi, batch atlaniliyor (Epoch {epoch+1}). Loss: {loss.item()}"
                    )
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                epoch_train_loss += loss.item()

            avg_train_loss = (
                epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            )

            model.eval()
            epoch_val_loss = 0
            all_dice_scores_per_class = []
            val_loop = tqdm(
                validation_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Dogrulama]",
                leave=False,
            )
            with torch.no_grad():
                for batch in val_loop:
                    if batch is None:
                        continue
                    images = batch["image"].to(device=device)
                    true_masks = batch["mask"].to(device=device)
                    with torch.amp.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=use_amp
                    ):
                        predicted_masks = model(images)
                        loss = criterion(predicted_masks, true_masks)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(
                            f"Uyari: NaN/Inf Val Loss tespit edildi, batch atlaniliyor (Epoch {epoch+1}). Loss: {loss.item()}"
                        )
                        continue

                    epoch_val_loss += loss.item()
                    dice_scores = calculate_dice_score(
                        predicted_masks, true_masks, n_classes=4
                    )
                    all_dice_scores_per_class.append(dice_scores.cpu().numpy())

            avg_val_loss = (
                epoch_val_loss / len(validation_loader)
                if len(validation_loader) > 0
                else 0
            )
            if len(all_dice_scores_per_class) > 0:
                mean_dice_per_class = np.mean(all_dice_scores_per_class, axis=0)
                avg_val_dice_score = np.mean(mean_dice_per_class[1:])
            else:
                avg_val_dice_score = 0.0

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{num_epochs} -> "
                f"LR: {current_lr:.6f}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Dice: {avg_val_dice_score:.4f}"
            )

            try:
                with open(log_file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [
                            epoch + 1,
                            f"{current_lr:.6f}",
                            f"{avg_train_loss:.4f}",
                            f"{avg_val_loss:.4f}",
                            f"{avg_val_dice_score:.4f}",
                        ]
                    )
            except Exception as log_e:
                print(f"Uyari: Log dosyasina yazarken hata olustu: {log_e}")

            if avg_val_dice_score > best_val_dice:
                best_val_dice = avg_val_dice_score
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_val_dice,
                    },
                    CHECKPOINT_PATH,
                )
                print(
                    f"** Yeni en iyi model (BN+DO) kaydedildi! Val Dice: {best_val_dice:.4f} **"
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Iyilesme yok, sayac: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"{patience} epoch boyunca iyilesme olmadi. Egitim durduruluyor.")
                break

            scheduler.step(avg_val_dice_score)

        print("Training finished!")
        print(
            f"En iyi Validation Dice Skoru (BatchNorm+Dropout ile): {best_val_dice:.4f}"
        )

    except Exception as e:
        print("\nEgitim sirasinda bir hata olustu:")
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
