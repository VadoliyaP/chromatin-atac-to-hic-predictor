import winbbi
import os

print("--- Checking Files on Disk ---")
hic_path = "data/ENCFF216ZNY.hic"
bw_path = "data/ENCFF667MDI.bigWig"

print(f".hic file exists: {os.path.exists(hic_path)} ({os.path.getsize(hic_path) / (1024**3):.2f} GB)")
print(f".bigWig file exists: {os.path.exists(bw_path)} ({os.path.getsize(bw_path) / (1024**2):.2f} MB)")

print("\n--- Testing .bigWig File with BigWigReader ---")
bw = winbbi.BigWigReader()
bw.open(bw_path)

chroms = bw.get_chromosomes()
print("Total Chromosomes in header:", len(chroms))
print("chr1 length:", bw.get_chrom_size("chr1"))

# Fetch raw signal from chr1 (0 to 10,000 bp)
signal = bw.read_raw_signal("chr1", 0, 10000)
print("Successfully read chr1 signal array!")
print("Sample signal array:", signal)

bw.close()