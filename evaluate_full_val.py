import torch
import torch.nn as nn
import numpy as np

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

def run_full_evaluation():
    device = torch.device("cpu")
    model = FastGenomicPredictorWithDistance(bins=100)
    model.load_state_dict(torch.load("chromatin_predictor_checkpoint.pt", map_location=device))
    model.eval()

    val_dict = torch.load("val_data.pt")
    val_x, val_y = val_dict['x'], val_dict['y']

    correlations = []
    
    with torch.no_grad():
        for i in range(len(val_x)):
            x_sample = val_x[i:i+1]
            if x_sample.dim() == 2:
                x_sample = x_sample.unsqueeze(1)
            
            y_true = val_y[i].numpy().flatten()
            y_pred = model(x_sample).squeeze(0).numpy().flatten()
            
            r = np.corrcoef(y_true, y_pred)[0, 1]
            if not np.isnan(r):
                correlations.append(r)

    mean_r = np.mean(correlations)
    std_r = np.std(correlations)
    median_r = np.median(correlations)

    print(f"\n==========================================")
    print(f" Full Validation Results across {len(correlations)} windows (chr21)")
    print(f"==========================================")
    print(f" Mean Pearson Correlation (r)   : {mean_r:.4f}")
    print(f" Median Pearson Correlation (r) : {median_r:.4f}")
    print(f" Standard Deviation             : {std_r:.4f}")
    print(f" Min r: {np.min(correlations):.4f} | Max r: {np.max(correlations):.4f}")
    print(f"==========================================\n")

if __name__ == "__main__":
    run_full_evaluation()
