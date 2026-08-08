import torch
import numpy as np
from production_dataset import ProductionGenomicDataset
from tqdm import tqdm

def convert_and_save(hic_path, bigwig_path, chrs, output_file, max_zero_fraction=0.75):
    print(f"\nProcessing chromosomes {chrs} -> {output_file}...")
    try:
        raw_ds = ProductionGenomicDataset(hic_path, bigwig_path, chromosomes=chrs)
    except Exception as e:
        print(f"Error initializing dataset for {chrs}: {e}")
        return False

    total_windows = len(raw_ds)
    print(f"Total candidate windows: {total_windows}")

    x_list, y_list = [], []
    skipped_sparse = 0

    for i in tqdm(range(total_windows), desc=f"Preprocessing {output_file}"):
        try:
            x, y = raw_ds[i]
            x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
            y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

            # Check sparsity on raw target Hi-C matrix
            zero_fraction = (y_tensor == 0).float().mean().item()
            if zero_fraction > max_zero_fraction:
                skipped_sparse += 1
                continue

            # Apply log1p transform to target Hi-C matrix
            y_log = torch.log1p(torch.clamp(y_tensor, min=0))

            # Standardize 1D input signals (z-score per window)
            x_std = (x_tensor - x_tensor.mean()) / (x_tensor.std() + 1e-6)

            x_list.append(x_std)
            y_list.append(y_log)
        except Exception as e:
            continue

    print(f"Filtered out {skipped_sparse}/{total_windows} sparse/empty windows.")

    if not x_list:
        print(f"Error: No valid non-sparse windows found for {chrs}.")
        return False

    X = torch.stack(x_list)
    Y = torch.stack(y_list)

    print(f"Saved compiled Tensors -> Shapes: X={X.shape}, Y={Y.shape}")
    torch.save({'x': X, 'y': Y}, output_file)
    print(f"Successfully generated {output_file}!")
    return True

if __name__ == "__main__":
    HIC_PATH = "data/ENCFF216ZNY.hic"
    BIGWIG_PATH = "data/ENCFF667MDI.bigWig"

    success_train = convert_and_save(HIC_PATH, BIGWIG_PATH, ['chr1', 'chr2'], "train_data.pt")
    success_val = convert_and_save(HIC_PATH, BIGWIG_PATH, ['chr21'], "val_data.pt")

    if success_train and success_val:
        print("\n--- Cleaned Preprocessing Complete! ---")
    else:
        print("\n--- Preprocessing failed. ---")
