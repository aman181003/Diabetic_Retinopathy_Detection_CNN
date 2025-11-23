# src/training/augment.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])

def get_valid_transforms(size=224):
    return A.Compose([
        A.Resize(size, size)
    ])
