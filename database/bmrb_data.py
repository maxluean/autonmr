"""
BMRB (Biological Magnetic Resonance Bank) Data Module
Cross-validation data สำหรับ chemical shifts
ใช้ยืนยันค่าจาก HMDB
"""

# BMRB chemical shift statistics
# ข้อมูลจาก BMRB ใช้ cross-validate กับ HMDB
BMRB_NMR_DATA = {
    "Isoleucine": {"peaks_ppm": [0.93, 1.00, 1.25, 1.46, 3.68]},
    "Leucine": {"peaks_ppm": [0.95, 0.96, 1.69, 3.72]},
    "Valine": {"peaks_ppm": [0.98, 1.03, 2.26, 3.60]},
    "Alanine": {"peaks_ppm": [1.47, 3.77]},
    "Threonine": {"peaks_ppm": [1.32, 3.57, 4.24]},
    "Asparagine": {"peaks_ppm": [2.84, 2.94, 3.99]},
    "Aspartate": {"peaks_ppm": [2.67, 2.80, 3.90]},
    "Glutamate": {"peaks_ppm": [2.05, 2.11, 2.34, 3.74]},
    "Phenylalanine": {"peaks_ppm": [3.11, 3.27, 3.98, 7.32, 7.36, 7.41]},
    "Tyrosine": {"peaks_ppm": [3.04, 3.93, 6.89, 7.18]},
    "Tryptophan": {"peaks_ppm": [7.19, 7.28, 7.53, 7.72]},
    "Histidine": {"peaks_ppm": [3.23, 7.08, 7.98]},
    "Lysine": {"peaks_ppm": [1.45, 1.71, 1.89, 3.02, 3.75]},
    "Sarcosine": {"peaks_ppm": [2.73, 3.60]},
    "Glucose": {"peaks_ppm": [3.23, 3.40, 3.45, 3.48, 3.72, 3.83, 4.64, 5.22]},
    "Sucrose": {"peaks_ppm": [3.47, 3.55, 3.67, 3.75, 3.82, 4.04, 4.21, 5.40]},
    "Maltose": {"peaks_ppm": [3.27, 3.41, 3.61, 3.77, 3.94, 4.64, 5.21, 5.40]},
    "Fructose": {"peaks_ppm": [3.56, 3.69, 3.79, 3.88, 4.00, 4.10]},
    "Xylose": {"peaks_ppm": [3.21, 3.41, 3.61, 4.57]},
    "Citrate": {"peaks_ppm": [2.53, 2.68]},
    "Acetate": {"peaks_ppm": [1.91]},
    "Succinate": {"peaks_ppm": [2.40]},
    "Propionate": {"peaks_ppm": [1.05, 2.17]},
    "Formate": {"peaks_ppm": [8.45]},
    "Pyroglutamate": {"peaks_ppm": [2.03, 2.39, 2.50, 4.16]},
    "Guanosine": {"peaks_ppm": [5.91, 8.00]},
    "Uridine": {"peaks_ppm": [5.88, 5.91, 7.86]},
    "Adenosine": {"peaks_ppm": [6.07, 8.21, 8.34]},
    "Uracil": {"peaks_ppm": [5.79, 7.53]},
    "Ethanol": {"peaks_ppm": [1.17, 3.65]},
    "Methanol": {"peaks_ppm": [3.35]},
    "Choline": {"peaks_ppm": [3.19, 3.51, 4.06]},
    "O-Phosphocholine": {"peaks_ppm": [3.21, 3.57, 4.16]},
    "4-Aminobutyrate": {"peaks_ppm": [1.89, 2.29, 3.00]},
    "Nicotinate": {"peaks_ppm": [7.49, 8.24, 8.61, 8.93]},
}


def get_bmrb_shifts(compound_name):
    """ดึง chemical shifts จาก BMRB"""
    return BMRB_NMR_DATA.get(compound_name, None)


def get_all_bmrb_compounds():
    """ดึงรายชื่อสารทั้งหมดใน BMRB"""
    return list(BMRB_NMR_DATA.keys())