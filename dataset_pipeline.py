import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pyBigWig
import hicstraw

class GenomicWindowDataset(Dataset):
    def __init__(self, hic_file, bw_file, chromosomes, resolution=10000, window_size=10000000, stride=2000000):
        """
        chromosomes: List of strings, e.g., ['chr1', 'chr2']
        window_size: 10 Mb (10,000,000 bp)
        stride: 2 Mb sliding step between windows
        """
        self.hic_file = hic_file
        self.bw_file = bw_file
        self.resolution = resolution
        self.window_size = window_size
        self.bins_per_window = window_size // resolution # 1000 bins
        
        self.windows = []
        
        # Open BigWig briefly to check chromosome lengths
        bw = pyBigWig.open(bw_file)
        chrom_lengths = bw.chroms()
        bw.close()
        
        # Generate target window coordinates across specified chromosomes
        for chrom in chromosomes:
            if chrom not in chrom_lengths:
                continue
            chr_len = chrom_lengths[chrom]
            for start in range(0, chr_len - window_size, stride):
                end = start + window_size
                self.windows.append((chrom, start, end))
                
        print(f"Initialized Dataset with {len(self.windows)} windows across {chromosomes}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chrom, start, end = self.windows[idx]
        
        # 1. Extract 1D ATAC-seq signal (Input X)
        bw = pyBigWig.open(self.bw_file)
        # Fetch individual base-pair metrics binned to resolution steps
        try:
            signals = bw.stats(chrom, start, end, type="mean", nBins=self.bins_per_window)
            # Replace any NaNs with 0.0
            x = np.array([s if s is not None else 0.0 for s in signals], dtype=np.float32)
        except Exception:
            x = np.zeros(self.bins_per_window, dtype=np.float32)
        bw.close()
        
        # Add channel dimension: [1, 1000]
        x_tensor = torch.tensor(x).unsqueeze(0)
        
        # 2. Extract 2D Hi-C Contact Matrix (Target Y)
        y_matrix = np.zeros((self.bins_per_window, self.bins_per_window), dtype=np.float32)
        try:
            result = hicstraw.straw('observed', 'NONE', self.hic_file, f"{chrom}:{start}:{end}", f"{chrom}:{start}:{end}", 'BP', self.resolution)
            for i in range(len(result)):
                # Calculate relative index within this specific window
                idx_x = (result[i].binX - start) // self.resolution
                idx_y = (result[i].binY - start) // self.resolution
                
                if 0 <= idx_x < self.bins_per_window and 0 <= idx_y < self.bins_per_window:
                    val = np.log10(result[i].counts + 1) # Normalization applied here
                    y_matrix[idx_x, idx_y] = val
                    y_matrix[idx_y, idx_x] = val # Maintain symmetry
        except Exception:
            pass # Keep matrix as zeros if an empty region is queried
            
        y_tensor = torch.tensor(y_matrix)
        
        return x_tensor, y_tensor

if __name__ == "__main__":
    # Quick trial run using a subset of chromosomes
    HIC_PATH = "data/ENCFF216ZNY.hic"
    BIGWIG_PATH = "data/ENCFF667MDI.bigWig"
    
    # Train on Chromosome 1, save others for validation/testing later
    train_dataset = GenomicWindowDataset(HIC_PATH, BIGWIG_PATH, chromosomes=['chr1'])
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print("Testing data loader extraction shape...")
    for batch_x, batch_y in train_loader:
        print("Input X shape (ATAC):", batch_x.shape)   # Expected: [batch, 1, 1000]
        print("Target Y shape (Hi-C):", batch_y.shape) # Expected: [batch, 1000, 1000]
        break
