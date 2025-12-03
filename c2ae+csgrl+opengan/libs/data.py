import os
from typing import Dict, Sequence

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as transforms

from libs.opt import CFG


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


class _RemappedSubset(Dataset):
    def __init__(self, base: Dataset, indices, label_map: Dict[int, int], unknown_label: int):
        self.base = base
        self.indices = list(indices)
        self.label_map = label_map
        self.unknown_label = unknown_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, label = self.base[base_idx]
        label = self.label_map.get(label, self.unknown_label)
        return img, label


class Data:
    def __init__(
        self,
        cfgs=None,
        known_classes: Sequence[int] | None = None,
        train_root: str | None = None,
        test_root: str | None = None,
    ):
        self.cfgs = cfgs if cfgs else CFG()
        self.model_name = getattr(self.cfgs, "model_name", "csgrl").lower()
        self.batch_size = self.cfgs.batch_size
        self.num_known_classes = self.cfgs.num_known_classes
        self.total_num_classes = self.cfgs.total_num_classes

        self.known_classes = list(known_classes) if known_classes is not None else list(
            range(self.num_known_classes)
        )
        self.unknown_classes = [i for i in range(self.total_num_classes) if i not in self.known_classes]
        self.unknown_label = len(self.known_classes)

        base_dir = getattr(self.cfgs, "data_dir", "./data")
        self.train_root = train_root or os.path.join(base_dir, "train")
        self.test_root = test_root or os.path.join(base_dir, "test")

        self._build_transforms()
        self._load_dataset()

    def _build_transforms(self):
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def _load_dataset(self):
        self.full_train_dataset = DataFolderOpen(self.train_root, transform=self.transform_train)
        self.full_test_dataset = DataFolderOpen(self.test_root, transform=self.transform_test)
        self._split_dataset()

    def _split_dataset(self):
        label_map = {k: i for i, k in enumerate(self.known_classes)}
        self.label_map = label_map

        # Train splits
        train_indices = [i for i, (_, lbl) in enumerate(self.full_train_dataset.imgs) if lbl in self.known_classes]
        open_train_indices = [i for i, (_, lbl) in enumerate(self.full_train_dataset.imgs) if lbl in self.unknown_classes]

        self.train_dataset = _RemappedSubset(
            self.full_train_dataset, train_indices, label_map, self.unknown_label
        )
        self.open_train_dataset = _RemappedSubset(
            self.full_train_dataset, open_train_indices, label_map, self.unknown_label
        )

        # Test splits
        known_test_indices = [i for i, (_, lbl) in enumerate(self.full_test_dataset.imgs) if lbl in self.known_classes]
        unknown_test_indices = [i for i, (_, lbl) in enumerate(self.full_test_dataset.imgs) if lbl in self.unknown_classes]

        self.known_test_dataset = _RemappedSubset(
            self.full_test_dataset, known_test_indices, label_map, self.unknown_label
        )
        self.unknown_test_dataset = _RemappedSubset(
            self.full_test_dataset, unknown_test_indices, label_map, self.unknown_label
        )
        self.all_test_dataset = ConcatDataset([self.known_test_dataset, self.unknown_test_dataset])

    def _make_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool = False):
        common_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **common_kwargs,
        )

    def get_dataloader(self):
        train_loader = self._make_loader(self.train_dataset, shuffle=True, drop_last=True)
        test_known_loader = self._make_loader(self.known_test_dataset, shuffle=False)
        test_unknown_loader = self._make_loader(self.unknown_test_dataset, shuffle=False)

        if self.model_name in ("csgrl", "opengan"):
            open_source = self.open_train_dataset if len(self.open_train_dataset) > 0 else self.unknown_test_dataset
            open_loader = self._make_loader(open_source, shuffle=True)
            test_loader = self._make_loader(self.all_test_dataset, shuffle=False)
            return (train_loader, open_loader), test_loader

        return train_loader, (test_known_loader, test_unknown_loader)