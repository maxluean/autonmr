"""
NP-MRD (Natural Products Magnetic Resonance Database) Data Module
เหมาะกับ Plant Extracts โดยเฉพาะ [1]
"""

# NP-MRD data สำหรับ Natural Products / Plant compounds
NPMRD_NMR_DATA = {
    "Glucose": {"peaks_ppm": [3.25, 3.42, 3.47, 3.50, 3.74, 3.85, 4.66, 5.24]},
    "Sucrose": {"peaks_ppm": [3.49, 3.57, 3.69, 3.77, 3.84, 4.06, 4.23, 5.42]},
    "Citrate": {"peaks_ppm": [2.55, 2.70]},
    "Acetate": {"peaks_ppm": [1.93]},
    "Succinate": {"peaks_ppm": [2.42]},
    "Formate": {"peaks_ppm": [8.47]},
    "Ethanol": {"peaks_ppm": [1.19, 3.67]},
    "Alanine": {"peaks_ppm": [1.49, 3.79]},
    "Chlorogenate": {"peaks_ppm": [6.34, 6.88, 7.07, 7.57]},
}


def get_npmrd_shifts(compound_name):
    """ดึง chemical shifts จาก NP-MRD"""
    return NPMRD_NMR_DATA.get(compound_name, None)


def get_all_npmrd_compounds():
    """ดึงรายชื่อสารทั้งหมดใน NP-MRD"""
    return list(NPMRD_NMR_DATA.keys())