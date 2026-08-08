import hicstraw
import pyBigWig
import pandas as pd
import numpy as np

def extract_genomic_data(hic_file, bw_file, chrom, start, end, resolution):
    """
    Extracts both Hi-C matrix data and BigWig signal values for a specified genomic window.
    """
    print(f"--- Processing {chrom}:{start}-{end} at {resolution}bp resolution ---")
    
    # Extract Hi-C Contacts using hicstraw
    print("Extracting Hi-C interactions...")
    result = hicstraw.straw('observed', 'NONE', hic_file, f"{chrom}:{start}:{end}", f"{chrom}:{start}:{end}", 'BP', resolution)
    
    hic_records = []
    for i in range(len(result)):
        hic_records.append({
            'binX': result[i].binX,
            'binY': result[i].binY,
            'counts': result[i].counts
        })
    
    hic_df = pd.DataFrame(hic_records)
    print(f"Extracted {len(hic_df)} interactive pairs.")
    
    # Extract BigWig Signals using pyBigWig
    print("Extracting BigWig epigenetic signals...")
    bw = pyBigWig.open(bw_file)
    
    bins = list(range(start, end, resolution))
    bw_records = []
    
    for b in bins:
        bin_end = min(b + resolution, end)
        signal = bw.stats(chrom, b, bin_end, type="mean")[0]
        signal = signal if signal is not None else 0.0
        
        bw_records.append({
            'bin': b,
            'signal_mean': signal
        })
    
    bw_df = pd.DataFrame(bw_records)
    bw.close()
    print("BigWig signals extracted successfully.")
    
    return hic_df, bw_df

if __name__ == "__main__":

    HIC_PATH = "data/ENCFF216ZNY.hic" 
    BIGWIG_PATH = "data/ENCFF667MDI.bigWig"
    
    CHROM = "chr1"
    START = 10_000_000
    END = 11_000_000
    RESOLUTION = 10_000  
    
    try:
        hic_matrix, signal_profile = extract_genomic_data(HIC_PATH, BIGWIG_PATH, CHROM, START, END, RESOLUTION)
        
        hic_matrix.to_csv("hic_matrix_output.csv", index=False)
        signal_profile.to_csv("epigenetic_signal_output.csv", index=False)
        print("\nSuccess! Saved outputs to 'hic_matrix_output.csv' and 'epigenetic_signal_output.csv'.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
