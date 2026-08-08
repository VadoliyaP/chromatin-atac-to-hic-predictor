import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from production_dataset import ProductionGenomicDataset
import os

class GenomicPredictor(nn.Module):
    def __init__(self, bins=100):
        super(GenomicPredictor, self).__init__()
        self.bins = bins
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * bins, bins * bins),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.conv1d(x)
        features = features.view(batch_size, -1) 
        out = self.fc(features) 
        return out.view(batch_size, self.bins, self.bins) 

class StructuralChromatinLoss(nn.Module):
    def __init__(self):
        super(StructuralChromatinLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)
        
        batch_size, bins, _ = pred.shape
        weight_matrix = torch.eye(bins, device=pred.device) * 2.0 + 1.0
        weighted_loss = torch.mean(weight_matrix * (pred - target) ** 2)
        
        return base_mse + 0.5 * weighted_loss

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    HIC_PATH = "data/ENCFF216ZNY.hic"
    BIGWIG_PATH = "data/ENCFF667MDI.bigWig"
    
    train_dataset = ProductionGenomicDataset(HIC_PATH, BIGWIG_PATH, chromosomes=['chr1', 'chr2'])
    val_dataset = ProductionGenomicDataset(HIC_PATH, BIGWIG_PATH, chromosomes=['chr21'])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model = GenomicPredictor(bins=100).to(device)
    criterion = StructuralChromatinLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    epochs = 3
    print("\nStarting scale-up execution run...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            predictions = model.forward(x)
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
                predictions = model(x)
                loss = criterion(predictions, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    torch.save(model.state_dict(), "chromatin_predictor_checkpoint.pt")
    print("\nProduction training test pass complete! Model saved.")

if __name__ == "__main__":
    train_model()
