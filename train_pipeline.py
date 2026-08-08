import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time

from dataset_pipeline import GenomicWindowDataset
from hic_predictor_model import DilatedHiCPredictor

def train_model():
    HIC_PATH = "data/ENCFF216ZNY.hic"
    BIGWIG_PATH = "data/ENCFF667MDI.bigWig"
    
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    EPOCHS = 5
    
    print("[1/4] Loading genomic dataset profiles...")
    full_dataset = GenomicWindowDataset(
        HIC_PATH, BIGWIG_PATH, chromosomes=['chr1'], 
        window_size=1000000, stride=500000
    )
    
    print("[2/4] Slicing out the first 15 windows for fast CPU test...")
    test_subset = Subset(full_dataset, range(min(15, len(full_dataset))))
    dataloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cpu")
    print(f"[3/4] Initializing network architecture on device: {device}")
    model = DilatedHiCPredictor(num_bins=100).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n[4/4] Starting model training execution loop...")
    model.train()
    
    total_start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        running_loss = 0.0
        batch_count = 0
        
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_start = time.time()
            print(f"  -> Processing Batch {i+1}/{len(dataloader)} (Contains {len(batch_x)} windows)...", end="", flush=True)
            
            optimizer.zero_grad()
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            batch_duration = time.time() - batch_start
            print(f" Done! (Loss: {loss.item():.4f} | Time: {batch_duration:.2f}s)")
            
        epoch_loss = running_loss / batch_count if batch_count > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        print(f"=== Epoch {epoch} finished in {epoch_duration:.2f} seconds | Avg Loss: {epoch_loss:.4f} ===")
        
    total_duration = time.time() - total_start_time
    torch.save(model.state_dict(), "hic_predictor_weights.pt")
    print(f"\nSuccess! Total training time: {total_duration:.2f} seconds.")
    print("Light weights saved cleanly to 'hic_predictor_weights.pt'")

if __name__ == "__main__":
    train_model()
