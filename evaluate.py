import torch
import torch.nn as nn
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


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
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
    def __init__(self, n_channels, n_classes, dropout_prob=0.2):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64, dropout_prob=dropout_prob)
        self.down1 = Down(64, 128, dropout_prob=dropout_prob)
        self.down2 = Down(128, 256, dropout_prob=dropout_prob)
        self.down3 = Down(256, 512, dropout_prob=dropout_prob)
        self.down4 = Down(512, 1024, dropout_prob=dropout_prob)
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

    def __init__(self, patient_dirs, transform=None):
        super().__init__()
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        self.num_slices_per_scan = 155
        print("Veri seti taraniyor ve dosyalar dogrulaniyor...")
        for patient_dir in tqdm(self.patient_dirs, desc="Hastalar Taraniyor"):
            flair_files = list(patient_dir.glob("*_flair.nii"))
            seg_files = list(patient_dir.glob("*_seg.nii")) + list(
                patient_dir.glob("*Segm.nii")
            )
            if len(flair_files) == 1 and len(seg_files) == 1:
                self.image_paths.append(flair_files[0])
                self.mask_paths.append(seg_files[0])
        print(
            f"Dogrulama tamamlandi. Toplam {len(self.image_paths)} adet gecerli hasta bulundu."
        )

    def __len__(self):
        return len(self.image_paths) * self.num_slices_per_scan

    def __getitem__(self, idx):
        patient_idx = idx // self.num_slices_per_scan
        slice_idx = idx % self.num_slices_per_scan
        image_path = self.image_paths[patient_idx]
        mask_path = self.mask_paths[patient_idx]
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        image_data_3d = image_nii.get_fdata()
        mask_data_3d = mask_nii.get_fdata()
        image_slice = image_data_3d[:, :, slice_idx]
        mask_slice = mask_data_3d[:, :, slice_idx]
        mask_slice[mask_slice == 4] = 3
        if self.transform:
            augmented = self.transform(image=image_slice, mask=mask_slice)
            image_slice = augmented["image"]
            mask_slice = augmented["mask"]
        if not self.transform:
            image_tensor = torch.from_numpy(image_slice.copy()).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_slice.copy()).long()
        else:
            image_tensor = image_slice
            mask_tensor = mask_slice.long()
        return {"image": image_tensor, "mask": mask_tensor}


def calculate_dice_components(preds_logits, targets, n_classes=4, epsilon=1e-6):
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


def main():
    try:
        # --- Veri Yukleyiciyi Hazirla (Ayni) ---
        data_folder = Path("./data/MICCAI_BraTS2020_TrainingData/")
        all_patient_dirs = [d for d in data_folder.iterdir() if d.is_dir()]
        train_dirs, val_dirs = train_test_split(
            all_patient_dirs, test_size=0.2, random_state=42
        )
        val_transform = A.Compose(
            [
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        )
        print("--- Validasyon Veri Seti Yukleniyor ---")
        validation_dataset = BratsDataset(
            patient_dirs=val_dirs, transform=val_transform
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}\n")

        model_paths = [
            "model_reg_aug_epoch_1.pth",
            "model_reg_aug_epoch_2.pth",
            "model_reg_aug_epoch_3.pth",
            "model_reg_aug_epoch_4.pth",
            "model_reg_aug_epoch_5.pth",
        ]

        best_dice = 0.0
        best_model_path = ""

        criterion = nn.CrossEntropyLoss()

        for model_path in model_paths:
            if not Path(model_path).exists():
                print(f"Uyari: {model_path} bulunamadi, atlaniliyor.")
                continue

            model = UNet(n_channels=1, n_classes=4, dropout_prob=0.2).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            epoch_val_loss = 0

            all_dice_scores_per_class = []

            with torch.no_grad():
                for batch in tqdm(
                    validation_loader, desc=f"Test ediliyor: {model_path}", leave=False
                ):
                    images = batch["image"].to(device=device)
                    true_masks = batch["mask"].to(device=device)
                    predicted_masks = model(images)

                    loss = criterion(predicted_masks, true_masks)
                    epoch_val_loss += loss.item()

                    dice_scores = calculate_dice_components(
                        predicted_masks, true_masks, n_classes=4
                    )

                    all_dice_scores_per_class.append(dice_scores.cpu().numpy())

            avg_val_loss = epoch_val_loss / len(validation_loader)

            mean_dice_per_class = np.mean(all_dice_scores_per_class, axis=0)

            avg_dice_score = np.mean(mean_dice_per_class[1:])

            print(
                f"Model: {model_path} -> Ortalama Kayip (CE Loss): {avg_val_loss:.4f}, Ortalama Dice Skoru: {avg_dice_score:.4f}"
            )

            if avg_dice_score > best_dice:
                best_dice = avg_dice_score
                best_model_path = model_path

        print("\n--- Degerlendirme Tamamlandi ---")
        print(f"En yuksek Ortalama Dice Skoru: {best_dice:.4f}")
        print(f"En iyi model dosyasi: {best_model_path}")

    except Exception as e:
        print("\nDegerlendirme sirasinda bir hata olustu:")
        print(e)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
