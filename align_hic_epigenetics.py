import pandas as pd

print("Loading data profiles...")
hic_df = pd.read_csv("hic_matrix_output.csv")
bw_df = pd.read_csv("epigenetic_signal_output.csv")

signal_map = dict(zip(bw_df['bin'], bw_df['signal_mean']))

print("Aligning 1D epigenetic signals to 2D contact anchors...")
hic_df['signal_X'] = hic_df['binX'].map(signal_map)

hic_df['signal_Y'] = hic_df['binY'].map(signal_map)

hic_df['signal_product'] = hic_df['signal_X'] * hic_df['signal_Y']

hic_df = hic_df.sort_values(by='counts', ascending=False)

output_file = "integrated_hic_epigenetics.csv"
hic_df.to_csv(output_file, index=False)
print(f"Alignment complete! Combined dataset saved to '{output_file}'")

print("\nTop 5 Interacting Regions with Aligned Signals:")
print(hic_df.head(5).to_string(index=False))
