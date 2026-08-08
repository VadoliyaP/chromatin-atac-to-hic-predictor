import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DilatedGenomicPredictor(nn.Module):
    def __init__(self, bins=100):
        super(DilatedGenomicPredictor, self).__init__()
        self.bins = bins
        
        self.conv1d_stage = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.conv2d_stage = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        coords = torch.arange(bins, dtype=torch.float32)
        dist = torch.abs(coords.unsqueeze(0) - coords.unsqueeze(1)) / float(bins)
        self.register_buffer('dist_matrix', dist.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        batch_size, _, bins = x.shape
        f1d = self.conv1d_stage(x)
        
        f2d_x = f1d.unsqueeze(3).repeat(1, 1, 1, bins)
        f2d_y = f1d.unsqueeze(2).repeat(1, 1, bins, 1)
        dist_batch = self.dist_matrix.repeat(batch_size, 1, 1, 1)
        
        f2d = torch.cat([f2d_x, f2d_y, dist_batch], dim=1)
        out = self.conv2d_stage(f2d).squeeze(1)
        return out

class StructuralChromatinLoss(nn.Module):
    def __init__(self):
        super(StructuralChromatinLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)
        batch_size, bins, _ = pred.shape
        weight_matrix = torch.eye(bins, device=pred.device) * 3.0 + 1.0
        weighted_loss = torch.mean(weight_matrix * (pred - target) ** 2)
        return base_mse + 0.5 * weighted_loss

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dict = torch.load("train_data.pt")
    val_dict = torch.load("val_data.pt")

    train_ds = TensorDataset(train_dict['x'], train_dict['y'])
    val_ds = TensorDataset(val_dict['x'], val_dict['y'])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = DilatedGenomicPredictor(bins=100).to(device)
    print(f"Dilated Architecture initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    criterion = StructuralChromatinLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 25
    print("\nStarting execution run (25 Epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                predictions = model(x)
                loss = criterion(predictions, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "chromatin_predictor_checkpoint.pt")
    print("\nRun complete! Model saved to chromatin_predictor_checkpoint.pt")

if __name__ == "__main__":
    run_training()
