import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        z = F.relu(self.fc(x))
        return z


class Classifier(nn.Module):
    def __init__(self, latent_dim: int = 128, num_classes: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z))
        logits = self.fc2(x)
        return logits


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 3, 3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 128, 4, 4)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x


class BaselineNet(nn.Module):
    """
    Simple baseline: shared encoder for classification and reconstruction.
    Stage 1 trains encoder+classifier with cross-entropy.
    Stage 2 trains encoder+decoder to reconstruct inputs.
    Open-set decision is based on reconstruction error.
    """

    def __init__(self, latent_dim: int = 128, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(latent_dim=latent_dim)
        self.classifier = Classifier(latent_dim=latent_dim, num_classes=num_classes)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor, mode: str = "classify") -> torch.Tensor:
        z = self.encoder(x)
        if mode == "classify":
            logits = self.classifier(z)
            return logits
        elif mode == "reconstruct":
            x_recon = self.decoder(z)
            return x_recon
        else:
            raise ValueError(f"Unsupported mode '{mode}' for BaselineNet.")

