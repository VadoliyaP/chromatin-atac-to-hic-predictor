import hic2cool
import os

hic_path = "data/ENCFF216ZNY.hic"
cool_path = "data/ENCFF216ZNY_10kb.cool"

resolution = 10000  

print(f"Starting conversion: {hic_path} -> {cool_path}")
print(f"Target resolution: {resolution // 1000}kb...")

hic2cool.hic2cool_convert(hic_path, cool_path, resolution)

print("\n--- Conversion Complete! ---")
print(f"Output .cool file size: {os.path.getsize(cool_path) / (1024**2):.2f} MB")
