import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from dataset_pipeline import GenomicWindowDataset
from hic_predictor_model import DilatedHiCPredictor

def calculate_hicrep_simplified(matrix_true, matrix_pred, max_distance=30):
    """
    Calculates a simplified Stratified Correlation Coefficient (HiC-Rep style)
    by breaking the 2D matrix into distance strata (diagonals) and averaging correlations.
    """
    N = matrix_true.shape[0]
    strata_corrs = []
    
        for d in range(1, min(max_distance, N)):
        diag_true = np.diagonal(matrix_true, offset=d)
        diag_pred = np.diagonal(matrix_pred, offset=d)
        
        
        if np.std(diag_true) > 1e-5 and np.std(diag_pred) > 1e-5:
            corr, _ = pearsonr(diag_true, diag_pred)
            if not np.isnan(corr):
                strata_corrs.append(corr)
                
    return np.mean(strata_corrs) if strata_corrs else 0.0

def evaluate():
    HIC_PATH = "data/ENCFF216ZNY.hic"
    BIGWIG_PATH = "data/ENCFF667MDI.bigWig"
    
    print("[1/3] Loading held-out test data window...")
    
    test_dataset = GenomicWindowDataset(
        HIC_PATH, BIGWIG_PATH, chromosomes=['chr1'], 
        window_size=1000000, stride=500000
    )
    
    
    X_test, Y_true = test_dataset[30]
    X_test = X_test.unsqueeze(0) 
    
    print("[2/3] Loading trained model weights...")
    model = DilatedHiCPredictor(num_bins=100)
    model.load_state_dict(torch.load("hic_predictor_weights.pt"))
    model.eval()
    
    print("[3/3] Generating model predictions...")
    with torch.no_grad():
        Y_pred = model(X_test).squeeze(0).numpy() 
        Y_true = Y_true.numpy()
        
    Y_pred = (Y_pred + Y_pred.T) / 2
    
    hicrep_score = calculate_hicrep_simplified(Y_true, Y_pred)
    print(f"\n=============================================")
    print(f"Evaluation Simplified HiC-Rep Score: {hicrep_score:.4f}")
    print(f"=============================================")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = ax1.imshow(Y_true, cmap="YlOrRd", aspect='auto')
    ax1.set_title("Experimental Hi-C (Ground Truth)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(Y_pred, cmap="YlOrRd", aspect='auto')
    ax2.set_title("Predicted Hi-C (From ATAC-seq)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("hic_prediction_comparison.png", dpi=300)
    print("Comparison plot successfully saved as 'hic_prediction_comparison.png'!")

if __name__ == "__main__":
    evaluate()
