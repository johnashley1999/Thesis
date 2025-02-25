import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATA_DIR = "/home/john/Documents/Thesis/GRTI Automation/UNET_Bird_Segmentation/Data"

class BirdSegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image files
        self.images = [f for f in os.listdir(image_dir) if not f.endswith('_mask.png')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        print(f"image path: {img_path}")

        # Construct mask path by adding '_mask.png'
        mask_name = img_name.rsplit('.', 1)[0] + '_mask.png'
        mask_path = os.path.join(self.image_dir, mask_name)
        print(f"mask path: {mask_path}")

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

        image = image.resize((160, 160))
        mask = mask.resize((160, 160))
        
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Normalize mask to binary (0 and 1)
        mask = (mask > 0).astype(np.float32)
        
        # Apply transformations if any
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensor
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        mask = torch.FloatTensor(mask)
        
        return image, mask

# Create transforms for training
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = A.Compose([
    A.Resize(160, 160),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Datasets
train_dataset = BirdSegmentationDataset(
    image_dir=os.path.join(DATA_DIR, 'train'),
    transform=train_transform
)

val_dataset = BirdSegmentationDataset(
    image_dir=os.path.join(DATA_DIR, 'valid'),
    transform=val_transform
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)