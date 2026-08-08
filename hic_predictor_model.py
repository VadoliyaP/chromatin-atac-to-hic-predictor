import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedHiCPredictor(nn.Module):
    def __init__(self, num_bins=1000, embedding_dim=32):
        super(DilatedHiCPredictor, self).__init__()
        self.num_bins = num_bins
        
        # 1D Feature Extraction with Exponential Dilation
        self.conv1d_1 = nn.Conv1d(1, embedding_dim, kernel_size=3, padding=1, dilation=1)
        self.conv1d_2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=2, dilation=2)
        self.conv1d_3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=4, dilation=4)
        self.conv1d_4 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=8, dilation=8)
        
        self.bn1d = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        
        # 2D Projection Refinement Layers
        self.conv2d_1 = nn.Conv2d(embedding_dim, 16, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(16)

    def forward(self, x):
        
        feat = self.relu(self.conv1d_1(x))
        feat = self.relu(self.conv1d_2(feat))
        feat = self.relu(self.conv1d_3(feat))
        feat = self.relu(self.conv1d_4(feat))
        feat = self.bn1d(feat) 
        batch_size, channels, length = feat.shape
        
        feat_expanded_X = feat.unsqueeze(3) 
        feat_expanded_Y = feat.unsqueeze(2) 
        
        matrix_2d = feat_expanded_X * feat_expanded_Y
        
        # Refine 2D contact topology
        out = self.relu(self.bn2d(self.conv2d_1(matrix_2d)))
        out = self.conv2d_2(out) 
        
        return out.squeeze(1)

if __name__ == "__main__":
    print("Initializing Dilated Convolutional Predictor Model...")
    model = DilatedHiCPredictor()
    
    sample_input = torch.randn(2, 1, 1000) 
    print(f"Sample Input Shape:  {sample_input.shape}")
    
    sample_output = model(sample_input)
    print(f"Sample Output Shape: {sample_output.shape} (Expected: [2, 1000, 1000])")
    print("\nModel Forward Pass Executed Successfully without shape conflicts!")
