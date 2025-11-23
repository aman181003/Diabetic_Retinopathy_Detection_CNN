# src/training/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FundusDataset(Dataset):
    def __init__(self, df, img_dir, mode='train', size=224, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mode = mode
        self.size = size
        if transforms is None:
            if mode=='train':
                self.transforms = T.Compose([
                    T.Resize((size,size)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.15),
                    T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize((size,size)),
                    T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        label = int(row['label'])
        return img, label
