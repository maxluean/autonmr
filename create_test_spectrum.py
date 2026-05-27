# สร้างไฟล์นี้ชื่อ create_test_spectrum.py
import numpy as np
import pandas as pd

# สร้าง synthetic NMR spectrum
ppm_axis = np.linspace(0.5, 10.0, 20000)
spectrum = np.zeros(20000)

# สาร 40+ ชนิดจาก Plant Extract [1]
compounds = {
    'Alanine':   [(1.48, 1.0), (3.78, 0.7)],
    'Formate':   [(8.46, 1.0)],
    'Acetate':   [(1.92, 1.0)],
    'Citrate':   [(2.54, 1.0), (2.69, 0.95)],
    'Succinate': [(2.41, 1.0)],
    'Sucrose':   [(3.48, 0.7), (3.56, 0.6),
                  (3.68, 0.8), (5.41, 1.0)],
    'Glucose':   [(3.24, 0.6), (3.41, 0.7),
                  (4.65, 0.9), (5.23, 1.0)],
    'Tryptophan':[(7.20, 0.7), (7.54, 1.0)],
    'Tyrosine':  [(6.90, 1.0), (7.19, 0.95)],
    'Nicotinate':[(7.50, 0.7), (8.94, 1.0)],
}

for compound, peaks in compounds.items():
    for ppm, intensity in peaks:
        width = 0.012
        peak = intensity * (width**2) / (
            (ppm_axis - ppm)**2 + width**2
        )
        spectrum += peak

# เพิ่ม noise
noise = np.random.normal(0, 0.005 * np.max(spectrum), len(spectrum))
spectrum = np.maximum(spectrum + noise, 0)

# บันทึกเป็น CSV (2 columns: ppm, intensity)
df = pd.DataFrame({
    'ppm': ppm_axis,
    'intensity': spectrum
})
df.to_csv('test_plant_extract.csv', index=False, header=False)
print("Created: test_plant_extract.csv")
print(f"Points: {len(spectrum):,}")
print(f"PPM range: {ppm_axis[0]:.1f} - {ppm_axis[-1]:.1f}")