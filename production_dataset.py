import numpy as np
import torch
from torch.utils.data import Dataset
import pyBigWig
import hicstraw

class ProductionGenomicDataset(Dataset):
    def __init__(self, hic_file, bw_file, chromosomes, resolution=10000, window_size=1000000, stride=500000):
        """
        Production dataset handling full-chromosome window extraction.
        window_size: 1 Mb (1,000,000 bp) for manageable CPU/GPU memory footprint
        stride: 500 kb step for overlapping window coverage
        """
        self.hic_file = hic_file
        self.bw_file = bw_file
        self.resolution = resolution
        self.window_size = window_size
        self.bins_per_window = window_size // resolution # 100 bins
        
        self.windows = []
        
        # Open BigWig to determine precise chromosome lengths
        bw = pyBigWig.open(bw_file)
        chrom_lengths = bw.chroms()
        bw.close()
        
        # Scan entire target chromosomes to build the coordinate index map
        for chrom in chromosomes:
            if chrom not in chrom_lengths:
                print(f"Warning: {chrom} not found in BigWig file mapping.")
                continue
            chr_len = chrom_lengths[chrom]
            
            # Slide across the entire length of the chromosome
            for start in range(0, chr_len - window_size, stride):
                end = start + window_size
                self.windows.append((chrom, start, end))
                
        print(f"--> Dataset mapped: {len(self.windows)} windows ready across {chromosomes}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chrom, start, end = self.windows[idx]
        
        # 1. Fetch 1D ATAC-seq accessibility signals
        bw = pyBigWig.open(self.bw_file)
        try:
            signals = bw.stats(chrom, start, end, type="mean", nBins=self.bins_per_window)
            x = np.array([s if s is not None else 0.0 for s in signals], dtype=np.float32)
        except Exception:
            x = np.zeros(self.bins_per_window, dtype=np.float32)
        bw.close()
        
        x_tensor = torch.tensor(x).unsqueeze(0) # Shape: [1, 100]
        
        # 2. Fetch 2D Hi-C spatial interactions
        y_matrix = np.zeros((self.bins_per_window, self.bins_per_window), dtype=np.float32)
        try:
            result = hicstraw.straw('observed', 'NONE', self.hic_file, f"{chrom}:{start}:{end}", f"{chrom}:{start}:{end}", 'BP', self.resolution)
            for i in range(len(result)):
                idx_x = (result[i].binX - start) // self.resolution
                idx_y = (result[i].binY - start) // self.resolution
                
                if 0 <= idx_x < self.bins_per_window and 0 <= idx_y < self.bins_per_window:
                    val = np.log10(result[i].counts + 1)
                    y_matrix[idx_x, idx_y] = val
                    y_matrix[idx_y, idx_x] = val # Enforce matrix symmetry
        except Exception:
            pass
            
        y_tensor = torch.tensor(y_matrix) # Shape: [100, 100]
        
        return x_tensor, y_tensor

if __name__ == "__main__":
    HIC_PATH = "data/ENCFF216ZNY.hic"
    BIGWIG_PATH = "data/ENCFF667MDI.bigWig"
    
    # Define our split strategy
    train_chroms = ['chr1', 'chr2'] # Expand this list as needed
    test_chroms = ['chr21']          # Strictly held out for evaluation
    
    print("Testing production training dataset initialization...")
    train_ds = ProductionGenomicDataset(HIC_PATH, BIGWIG_PATH, chromosomes=train_chroms)
    print(f"Total training windows available: {len(train_ds)}")
    
    print("\nTesting production evaluation dataset initialization...")
    test_ds = ProductionGenomicDataset(HIC_PATH, BIGWIG_PATH, chromosomes=test_chroms)
    print(f"Total validation/test windows available: {len(test_ds)}")
