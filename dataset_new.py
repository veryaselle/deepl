import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 从 config.py 导入自动化路径
try:
    from config import RGB_DATASET_ROOT, SPLITS_ROOT
except ImportError:
    RGB_DATASET_ROOT = "EuroSAT_RGB"
    SPLITS_ROOT = "splits"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 1. Augmentation
def get_transforms(mode: str='train', aug_strength: str="mild", input_size: int=64):
    if mode == 'train':
        if aug_strength == "mild":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        elif aug_strength == "strong":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(45), 
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), 
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2), 
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

# 2. Dataset
class EuroSATDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.transform = transform
        self.root_dir = Path(root_dir)
        
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Split file not found: {txt_file}. Please run split_data.py first!")

        self.img_list = []
        with open(txt_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                path_str = line.split(' ')[0]
                self.img_list.append(path_str)
            
        self.classes = sorted(list(set([Path(p).parts[0] for p in self.img_list])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __getitem__(self, idx):
        rel_path = self.img_list[idx]
        img_path = self.root_dir / rel_path 
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error loading image at absolute path: {img_path.resolve()}")
            raise

        cls_name = Path(rel_path).parts[0]
        label = self.class_to_idx[cls_name]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_list)

# 3. Loader
def get_dataloaders(
        data_root=RGB_DATASET_ROOT, 
        split_dir=SPLITS_ROOT,      
        batch_size: int=64, 
        num_workers: int=2,
        aug_strength: str="mild"
        ) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    data_root = Path(data_root)
    split_path = Path(split_dir)

    train_dataset = EuroSATDataset(
        txt_file=split_path / "train.txt", 
        root_dir=data_root, 
        transform=get_transforms(mode='train', aug_strength=aug_strength)
    )

    val_dataset = EuroSATDataset(
        txt_file=split_path / "val.txt", 
        root_dir=data_root, 
        transform=get_transforms(mode='eval')
    )

    test_dataset = EuroSATDataset(
        txt_file=split_path / "test.txt", 
        root_dir=data_root, 
        transform=get_transforms(mode='eval')
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader