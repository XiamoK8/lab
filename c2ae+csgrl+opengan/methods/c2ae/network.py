import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        z = F.relu(self.fc(x))
        return z

class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = self.fc2(x)
        return x

class FiLMLayer(nn.Module):
    def __init__(self, latent_dim=128, num_classes=6):
        super().__init__()
        self.fc_gamma = nn.Linear(num_classes, latent_dim)
        self.fc_beta = nn.Linear(num_classes, latent_dim)
        nn.init.constant_(self.fc_gamma.weight, 0)
        nn.init.constant_(self.fc_gamma.bias, 1)
        nn.init.constant_(self.fc_beta.weight, 0)
        nn.init.constant_(self.fc_beta.bias, 0)

    def forward(self, z, condition_vector):
        gamma = self.fc_gamma(condition_vector)
        beta = self.fc_beta(condition_vector)
        return gamma * z + beta

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        
    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 128, 4, 4)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        return x

class C2AE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)
        self.film = FiLMLayer(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim)
        
    def create_condition_vector(self, labels):
        batch_size = labels.size(0)
        condition = torch.ones(batch_size, self.num_classes, device=labels.device) * (-1)
        condition[torch.arange(batch_size), labels] = 1
        return condition
    
    def forward(self, x, labels=None, mode='stage1'):
        if mode == 'stage1':
            z = self.encoder(x)
            logits = self.classifier(z)
            return logits
            
        elif mode == 'stage2':
            with torch.no_grad():
                z = self.encoder(x)

            condition_match = self.create_condition_vector(labels)
            z_match = self.film(z, condition_match)
            x_recon_match = self.decoder(z_match)
            x_recon_match = torch.clamp(x_recon_match, 0, 1)
            err_match = torch.mean(torch.abs(x - x_recon_match), dim=[1, 2, 3])

            batch_size = x.size(0)
            errs = []
            for k in range(self.num_classes):
                labels_k = torch.full((batch_size,), k, dtype=torch.long, device=x.device)
                condition_k = self.create_condition_vector(labels_k)
                z_k = self.film(z, condition_k)
                x_recon_k = self.decoder(z_k)
                x_recon_k = torch.clamp(x_recon_k, 0, 1)
                err_k = torch.mean(torch.abs(x - x_recon_k), dim=[1, 2, 3])
                errs.append(err_k)
            errs = torch.stack(errs, dim=1)

            mask_true = torch.zeros_like(errs, dtype=torch.bool)
            mask_true[torch.arange(batch_size), labels] = True
            big = torch.finfo(errs.dtype).max
            errs_masked = errs.masked_fill(mask_true, big)
            err_nonmatch_min, _ = torch.min(errs_masked, dim=1)

            return err_match, err_nonmatch_min
            
        elif mode == 'stage3':
            with torch.no_grad():
                z = self.encoder(x)
                logits = self.classifier(z)
                pred = torch.argmax(logits, dim=1)
                recon_errors = []

                for k in range(self.num_classes):
                    labels_k = torch.full((x.size(0),), k, dtype=torch.long, device=x.device)
                    condition_k = self.create_condition_vector(labels_k)
                    z_k = self.film(z, condition_k)
                    x_recon_k = self.decoder(z_k)
                    x_recon_k = torch.clamp(x_recon_k, 0, 1)
                    error_k = torch.mean(torch.abs(x - x_recon_k), dim=[1,2,3])
                    recon_errors.append(error_k)
                
                recon_errors = torch.stack(recon_errors, dim=1)
                min_recon_error, _ = torch.min(recon_errors, dim=1)
                
            return pred, min_recon_error
