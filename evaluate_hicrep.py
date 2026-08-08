import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np

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

def run_hicrep_evaluation(max_diag=30):
    device = torch.device("cpu")
    model = DilatedGenomicPredictor(bins=100)
    model.load_state_dict(torch.load("chromatin_predictor_checkpoint.pt", map_location=device))
    model.eval()

    val_dict = torch.load("val_data.pt")
    val_x, val_y = val_dict['x'], val_dict['y']

    global_rs = []
    strata_rs = {k: [] for k in range(1, max_diag + 1)}

    with torch.no_grad():
        for i in range(len(val_x)):
            x_sample = val_x[i:i+1]
            if x_sample.dim() == 2:
                x_sample = x_sample.unsqueeze(1)

            y_true = val_y[i].numpy()
            y_pred = model(x_sample).squeeze(0).numpy()

            # Global Pearson
            r_global = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
            if not np.isnan(r_global):
                global_rs.append(r_global)

            # Stratum-Adjusted Correlation per diagonal distance k
            for k in range(1, max_diag + 1):
                diag_true = np.diag(y_true, k=k)
                diag_pred = np.diag(y_pred, k=k)
                if np.std(diag_true) > 1e-6 and np.std(diag_pred) > 1e-6:
                    r_k = np.corrcoef(diag_true, diag_pred)[0, 1]
                    if not np.isnan(r_k):
                        strata_rs[k].append(r_k)

    scc_proxy = np.mean([np.mean(strata_rs[k]) for k in strata_rs if len(strata_rs[k]) > 0])

    print("\n=======================================================")
    print(" Dilated CNN Evaluation Results across 64 windows (chr21)")
    print("=======================================================")
    print(f" Mean Global Pearson Correlation (r)  : {np.mean(global_rs):.4f}")
    print(f" Stratum-Adjusted Correlation (SCC)   : {scc_proxy:.4f}")
    print("-------------------------------------------------------")
    print(" Distance-Stratified Correlations (Sample Diagonal Offsets):")
    for k in [1, 5, 10, 20, 30]:
        dist_kb = k * 10
        avg_r_k = np.mean(strata_rs[k]) if len(strata_rs[k]) > 0 else 0.0
        print(f"   * Diagonal {k:02d} ({dist_kb:3d} kb separation) : r = {avg_r_k:.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    run_hicrep_evaluation()
