import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class OpenDataset(Dataset):
    def __init__(self, base, known_classes, unknown_label):
        self.base = base
        self.label_map = {k: i for i, k in enumerate(sorted(known_classes))}
        self.unknown_label = unknown_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, lbl = self.base[idx]
        return img, self.label_map.get(lbl, self.unknown_label)


class Data:
    def __init__(self, args):
        self.args = args
        t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        t_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        root = getattr(args, "data_dir", "./data")
        self.train_set = datasets.ImageFolder(os.path.join(root, "train"), transform=t_train)
        self.test_set = datasets.ImageFolder(os.path.join(root, "test"), transform=t_test)
        self.known = list(range(getattr(args, "num_known_classes", 0)))
        self.unknown = [c for c in set(self.train_set.class_to_idx.values()) if c not in self.known]
        self.unknown_label = len(self.known)

    def get_dataloader(self):
        train_idx = [i for i, (_, lbl) in enumerate(self.train_set.samples) if lbl in self.known]
        known_test_idx = [i for i, (_, lbl) in enumerate(self.test_set.samples) if lbl in self.known]
        unknown_test_idx = [i for i, (_, lbl) in enumerate(self.test_set.samples) if lbl not in self.known]
        batch = getattr(self.args, "batch_size", 128)
        train_loader = DataLoader(OpenDataset(Subset(self.train_set, train_idx), self.known, self.unknown_label), batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        test_loader = DataLoader(OpenDataset(self.test_set, self.known, self.unknown_label), batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, test_loader
        # test_known_loader = DataLoader(OpenDataset(Subset(self.test_set, known_test_idx), self.known, self.unknown_label), batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
        # test_unknown_loader = DataLoader(OpenDataset(Subset(self.test_set, unknown_test_idx), self.known, self.unknown_label), batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
        # return train_loader, (test_known_loader, test_unknown_loader)
