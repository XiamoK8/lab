import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class DataFolderOpen(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.imgs = []

        for dir_name in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, dir_name)
            if not os.path.isdir(class_path):
                continue
            try:
                label = int(dir_name)
            except ValueError:
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isdir(img_path):
                    continue
                self.imgs.append((img_path, label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label


class CIFAR10_c2ae:
    def __init__(
        self,
        known_classes,
        batch_size=64,
        train_root=os.path.join("..", "cifar10", "train"),
        test_root=os.path.join("..", "cifar10", "test"),
    ):
        self.known_classes = list(known_classes)
        self.unknown_classes = [i for i in range(10) if i not in self.known_classes]
        self.batch_size = batch_size
        self.train_root = train_root
        self.test_root = test_root

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self._load_dataset()

    def _load_dataset(self):
        full_train_dataset = DataFolderOpen(
            self.train_root,
            transform=self.transform_train,
        )

        full_test_dataset = DataFolderOpen(
            self.test_root,
            transform=self.transform_test,
        )

        self._split_dataset(full_train_dataset, full_test_dataset)

    def _split_dataset(self, full_train_dataset, full_test_dataset):
        train_indices = [i for i in range(len(full_train_dataset))
            if full_train_dataset.imgs[i][1] in self.known_classes]
        train_dataset = Subset(full_train_dataset, train_indices)

        known_test_indices = [i for i in range(len(full_test_dataset))
            if full_test_dataset.imgs[i][1] in self.known_classes]
        known_test_dataset = Subset(full_test_dataset, known_test_indices)
        
        unknown_test_indices = [i for i in range(len(full_test_dataset))
            if full_test_dataset.imgs[i][1] in self.unknown_classes]
        unknown_test_dataset = Subset(full_test_dataset, unknown_test_indices)

        common_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs,
        )
        self.known_test_loader = DataLoader(
            known_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **common_kwargs,
        )
        self.unknown_test_loader = DataLoader(
            unknown_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **common_kwargs,
        )
