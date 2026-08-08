import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading integrated profiles...")
hic_df = pd.read_csv("hic_matrix_output.csv")
bw_df = pd.read_csv("epigenetic_signal_output.csv")

START = 10_000_000
END = 11_000_000
RESOLUTION = 10_000
all_bins = list(range(START, END, RESOLUTION))
num_bins = len(all_bins)

print("Constructing a complete, fully-mapped 2D grid...")
full_matrix = pd.DataFrame(0.0, index=all_bins, columns=all_bins)

for _, row in hic_df.iterrows():
    bx, by, val = row['binX'], row['binY'], row['counts']
    if bx in full_matrix.index and by in full_matrix.columns:
        full_matrix.loc[bx, by] = val
        full_matrix.loc[by, bx] = val  

log_matrix = np.log10(full_matrix.values + 1)

print("Generating perfectly aligned dual-layer genomic plot...")
fig, (ax_hic, ax_bw) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                     gridspec_kw={'height_ratios': [3, 1]})

extent = [START, END, END, START]
im = ax_hic.imshow(log_matrix, cmap='YlOrRd', extent=extent, aspect='auto')

cbar = fig.colorbar(im, ax=ax_hic, pad=0.02)
cbar.set_label('Log10(Counts + 1)', rotation=270, labelpad=15)

ax_hic.set_title("Chromatin Architecture vs. Epigenetic Signal (chr1:10Mb-11Mb)", fontsize=14, pad=15)
ax_hic.set_ylabel("Genomic Position X (bp)")

# Plot 1D Epigenetic Track
ax_bw.fill_between(bw_df['bin'], 0, bw_df['signal_mean'], color='teal', alpha=0.7)
ax_bw.set_ylabel("Signal Intensity")
ax_bw.set_xlabel("Genomic Position (bp)")

ax_bw.ticklabel_format(style='plain', axis='x')

plt.tight_layout()
output_image = "hic_epigenetics_map.png"
plt.savefig(output_image, dpi=300)
print(f"\nSuccess! Aligned plot successfully saved to '{output_image}'!")
