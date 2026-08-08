import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- Matching Network Architecture with Distance Prior ---
class FastGenomicPredictorWithDistance(nn.Module):
    def __init__(self, bins=100):
        super(FastGenomicPredictorWithDistance, self).__init__()
        self.bins = bins
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        coords = torch.arange(bins, dtype=torch.float32)
        dist = torch.abs(coords.unsqueeze(0) - coords.unsqueeze(1)) / float(bins)
        self.register_buffer('dist_matrix', dist.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        batch_size, _, bins = x.shape
        f1d = self.conv1d(x)
        f2d_x = f1d.unsqueeze(3).repeat(1, 1, 1, bins)
        f2d_y = f1d.unsqueeze(2).repeat(1, 1, bins, 1)
        dist_batch = self.dist_matrix.repeat(batch_size, 1, 1, 1)
        f2d = torch.cat([f2d_x, f2d_y, dist_batch], dim=1)
        out = self.conv2d(f2d).squeeze(1)
        return out

def evaluate_and_plot():
    device = torch.device("cpu")
    print("Loading updated model checkpoint...")
    model = FastGenomicPredictorWithDistance(bins=100)
    model.load_state_dict(torch.load("chromatin_predictor_checkpoint.pt", map_location=device))
    model.eval()

    print("Loading clean validation dataset (chr21)...")
    val_dict = torch.load("val_data.pt")
    val_x, val_y = val_dict['x'], val_dict['y']

    # Select window index (sample 10 from non-sparse validation set)
    sample_idx = 10
    x_sample = val_x[sample_idx:sample_idx+1]
    y_true = val_y[sample_idx].numpy()

    if x_sample.dim() == 2:
        x_sample = x_sample.unsqueeze(1)

    with torch.no_grad():
        y_pred = model(x_sample).squeeze(0).numpy()

    error_map = np.abs(y_true - y_pred)

    # Calculate Pearson Correlation for this window
    pearson_corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    print(f"\nUpdated Window Pearson Correlation (r): {pearson_corr:.4f}")

    # Plot Side-by-Side Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(y_true, cmap='Reds', origin='lower')
    axes[0].set_title("Ground Truth Log(Hi-C) (chr21)")
    axes[0].set_xlabel("Genomic Bins (10kb)")
    axes[0].set_ylabel("Genomic Bins (10kb)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, cmap='Reds', origin='lower')
    axes[1].set_title(f"Model Prediction (r = {pearson_corr:.3f})")
    axes[1].set_xlabel("Genomic Bins (10kb)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(error_map, cmap='viridis', origin='lower')
    axes[2].set_title("Absolute Residual Error")
    axes[2].set_xlabel("Genomic Bins (10kb)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_png = "validation_prediction_comparison.png"
    plt.savefig(output_png, dpi=300)
    print(f"Comparison plot saved: {output_png}")

if __name__ == "__main__":
    evaluate_and_plot()
