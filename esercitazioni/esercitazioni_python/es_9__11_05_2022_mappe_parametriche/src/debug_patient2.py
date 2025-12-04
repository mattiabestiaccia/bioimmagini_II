import pydicom
from pathlib import Path
import sys

data_dir = Path('data/DICOM/PAZIENTE2')
print(f"Checking {data_dir}")

files = sorted(data_dir.glob("*.dcm"))
print(f"Found {len(files)} files")

for f in files:
    print(f"Loading {f.name}...")
    try:
        ds = pydicom.dcmread(f)
        print(f"  Loaded. Shape: {ds.pixel_array.shape}")
    except Exception as e:
        print(f"  Error: {e}")
print("Done.")
