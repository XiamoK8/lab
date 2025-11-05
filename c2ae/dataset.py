import os
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class OpenSetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True, transform=None, known_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.known_classes = known_classes if known_classes else []

        split = "train" if train else "test"
        self.data_dir = os.path.join(root_dir, split)

        self.base_dataset = ImageFolder(
            root=self.data_dir,
            transform=transform
        )

        self.class_to_idx = self.base_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.indices = self._filter_known_classes()

    def _filter_known_classes(self):
        if not self.known_classes:
            return list(range(len(self.base_dataset)))

        known_cls_str = [str(cls) for cls in self.known_classes]

        known_labels = [self.class_to_idx[cls_str] for cls_str in known_cls_str
                       if cls_str in self.class_to_idx]

        indices = []
        for i, (_, label) in enumerate(self.base_dataset.samples):
            if label in known_labels:
                indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.base_dataset[original_idx]
        return image, label


def get_dataloaders(known_classes, batch_size, data_dir):
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
        #                      std=[0.2470, 0.2435, 0.2616])
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = OpenSetDataset(data_dir, train=True, transform=train_transform,
                                   known_classes=known_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, drop_last=True)

    test_known_dataset = OpenSetDataset(data_dir, train=False, transform=test_transform,
                                        known_classes=known_classes)
    test_known_loader = DataLoader(test_known_dataset, batch_size=batch_size, shuffle=False, 
                                   num_workers=0)

    total_num_classes = 10
    unknown_classes = [i for i in range(total_num_classes) if i not in known_classes]
    
    test_unknown_dataset = OpenSetDataset(data_dir, train=False, transform=test_transform,
                                          known_classes=unknown_classes)
    test_unknown_loader = DataLoader(test_unknown_dataset, batch_size=batch_size, shuffle=False, 
                                     num_workers=0)

    return train_loader, test_known_loader, test_unknown_loader