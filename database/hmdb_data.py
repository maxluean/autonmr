"""
HMDB (Human Metabolome Database) Data Module
ข้อมูล chemical shifts จาก https://hmdb.ca
สำหรับสาร 40+ ชนิดที่พบใน Plant Extract A-N [1]
"""

# HMDB IDs สำหรับสารทั้งหมดที่ระบุไว้ใน Domain 1 [1]
HMDB_COMPOUND_IDS = {
    "Isoleucine": "HMDB0000172",
    "Leucine": "HMDB0000687",
    "Valine": "HMDB0000883",
    "Alanine": "HMDB0000161",
    "Threonine": "HMDB0000167",
    "Asparagine": "HMDB0000168",
    "Aspartate": "HMDB0000191",
    "Glutamate": "HMDB0000148",
    "Phenylalanine": "HMDB0000159",
    "Tyrosine": "HMDB0000158",
    "Tryptophan": "HMDB0000929",
    "Histidine": "HMDB0000177",
    "Lysine": "HMDB0000182",
    "Sarcosine": "HMDB0000271",
    "Glucose": "HMDB0000122",
    "Sucrose": "HMDB0000258",
    "Maltose": "HMDB0000163",
    "Fructose": "HMDB0000660",
    "Xylose": "HMDB0000098",
    "Citrate": "HMDB0000094",
    "Acetate": "HMDB0000042",
    "Succinate": "HMDB0000254",
    "Propionate": "HMDB0000237",
    "Formate": "HMDB0000142",
    "Pyroglutamate": "HMDB0000267",
    "Guanosine": "HMDB0000133",
    "Uridine": "HMDB0000296",
    "Adenosine": "HMDB0000050",
    "Cytosine": "HMDB0000630",
    "Uracil": "HMDB0000300",
    "Ethanol": "HMDB0000108",
    "Methanol": "HMDB0001875",
    "Choline": "HMDB0000097",
    "O-Phosphocholine": "HMDB0001565",
    "N-Acetylcysteine": "HMDB0001890",
    "4-Aminobutyrate": "HMDB0000112",
    "Chlorogenate": "HMDB0003164",
    "Nicotinate": "HMDB0001488",
    "S-Adenosylhomocysteine": "HMDB0000939",
    "Xanthurenate": "HMDB0000881",
}

# HMDB Chemical Shifts (1H NMR)
# ข้อมูลจาก HMDB experimental spectra
HMDB_NMR_DATA = {
    "Isoleucine": {
        "peaks_ppm": [0.94, 1.01, 1.26, 1.47, 3.67],
        "multiplicities": ["t", "d", "m", "m", "d"],
        "relative_intensities": [1.0, 0.9, 0.5, 0.4, 0.6],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Leucine": {
        "peaks_ppm": [0.96, 0.97, 1.70, 3.73],
        "multiplicities": ["d", "d", "m", "t"],
        "relative_intensities": [1.0, 0.95, 0.6, 0.5],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Valine": {
        "peaks_ppm": [0.99, 1.04, 2.27, 3.61],
        "multiplicities": ["d", "d", "m", "d"],
        "relative_intensities": [1.0, 0.95, 0.5, 0.6],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Alanine": {
        "peaks_ppm": [1.48, 3.78],
        "multiplicities": ["d", "q"],
        "relative_intensities": [1.0, 0.7],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Threonine": {
        "peaks_ppm": [1.33, 3.58, 4.25],
        "multiplicities": ["d", "d", "m"],
        "relative_intensities": [1.0, 0.8, 0.6],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Asparagine": {
        "peaks_ppm": [2.85, 2.95, 4.00],
        "multiplicities": ["dd", "dd", "dd"],
        "relative_intensities": [0.8, 0.8, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Aspartate": {
        "peaks_ppm": [2.68, 2.81, 3.89],
        "multiplicities": ["dd", "dd", "dd"],
        "relative_intensities": [1.0, 0.9, 0.7],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Glutamate": {
        "peaks_ppm": [2.06, 2.12, 2.35, 3.75],
        "multiplicities": ["m", "m", "m", "dd"],
        "relative_intensities": [0.6, 0.6, 1.0, 0.7],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Phenylalanine": {
        "peaks_ppm": [3.12, 3.28, 3.99, 7.33, 7.37, 7.42],
        "multiplicities": ["dd", "dd", "dd", "m", "m", "m"],
        "relative_intensities": [0.5, 0.5, 0.4, 0.8, 1.0, 0.7],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Tyrosine": {
        "peaks_ppm": [3.05, 3.94, 6.90, 7.19],
        "multiplicities": ["dd", "dd", "d", "d"],
        "relative_intensities": [0.5, 0.4, 1.0, 0.95],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Tryptophan": {
        "peaks_ppm": [7.20, 7.29, 7.54, 7.73],
        "multiplicities": ["t", "s", "d", "d"],
        "relative_intensities": [0.7, 0.8, 1.0, 0.9],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Histidine": {
        "peaks_ppm": [3.24, 7.09, 7.99],
        "multiplicities": ["dd", "s", "s"],
        "relative_intensities": [0.5, 1.0, 0.9],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Lysine": {
        "peaks_ppm": [1.46, 1.72, 1.90, 3.03, 3.76],
        "multiplicities": ["m", "m", "m", "t", "t"],
        "relative_intensities": [0.6, 0.7, 0.7, 0.8, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Sarcosine": {
        "peaks_ppm": [2.74, 3.61],
        "multiplicities": ["s", "s"],
        "relative_intensities": [1.0, 0.9],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Glucose": {
        "peaks_ppm": [3.24, 3.41, 3.46, 3.49, 3.73, 3.84, 4.65, 5.23],
        "multiplicities": ["dd", "t", "m", "t", "m", "m", "d", "d"],
        "relative_intensities": [0.6, 0.7, 0.7, 0.7, 0.8, 0.6, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Sucrose": {
        "peaks_ppm": [3.48, 3.56, 3.68, 3.76, 3.83, 4.05, 4.22, 5.41],
        "multiplicities": ["t", "dd", "s", "t", "m", "t", "d", "d"],
        "relative_intensities": [0.7, 0.6, 0.8, 0.7, 0.7, 0.8, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Maltose": {
        "peaks_ppm": [3.28, 3.42, 3.62, 3.78, 3.95, 4.65, 5.22, 5.41],
        "multiplicities": ["t", "dd", "m", "m", "m", "d", "d", "d"],
        "relative_intensities": [0.6, 0.6, 0.7, 0.7, 0.8, 0.9, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Fructose": {
        "peaks_ppm": [3.57, 3.70, 3.80, 3.89, 4.01, 4.11],
        "multiplicities": ["m", "m", "m", "dd", "dd", "d"],
        "relative_intensities": [0.7, 0.8, 0.7, 0.8, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Xylose": {
        "peaks_ppm": [3.22, 3.42, 3.62, 4.58],
        "multiplicities": ["dd", "t", "m", "d"],
        "relative_intensities": [0.7, 0.7, 0.8, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Citrate": {
        "peaks_ppm": [2.54, 2.69],
        "multiplicities": ["d", "d"],
        "relative_intensities": [1.0, 0.95],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Acetate": {
        "peaks_ppm": [1.92],
        "multiplicities": ["s"],
        "relative_intensities": [1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Succinate": {
        "peaks_ppm": [2.41],
        "multiplicities": ["s"],
        "relative_intensities": [1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Propionate": {
        "peaks_ppm": [1.06, 2.18],
        "multiplicities": ["t", "q"],
        "relative_intensities": [0.8, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Formate": {
        "peaks_ppm": [8.46],
        "multiplicities": ["s"],
        "relative_intensities": [1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Pyroglutamate": {
        "peaks_ppm": [2.04, 2.40, 2.51, 4.17],
        "multiplicities": ["m", "m", "m", "dd"],
        "relative_intensities": [0.7, 0.8, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Guanosine": {
        "peaks_ppm": [5.92, 8.01],
        "multiplicities": ["d", "s"],
        "relative_intensities": [0.8, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Uridine": {
        "peaks_ppm": [5.89, 5.92, 7.87],
        "multiplicities": ["d", "d", "d"],
        "relative_intensities": [0.7, 0.8, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Adenosine": {
        "peaks_ppm": [6.08, 8.22, 8.35],
        "multiplicities": ["d", "s", "s"],
        "relative_intensities": [0.7, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Cytosine": {
        "peaks_ppm": [5.99, 7.86],
        "multiplicities": ["d", "d"],
        "relative_intensities": [0.8, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Uracil": {
        "peaks_ppm": [5.80, 7.54],
        "multiplicities": ["d", "d"],
        "relative_intensities": [0.8, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Ethanol": {
        "peaks_ppm": [1.18, 3.66],
        "multiplicities": ["t", "q"],
        "relative_intensities": [1.0, 0.8],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Methanol": {
        "peaks_ppm": [3.36],
        "multiplicities": ["s"],
        "relative_intensities": [1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Choline": {
        "peaks_ppm": [3.20, 3.52, 4.07],
        "multiplicities": ["s", "m", "m"],
        "relative_intensities": [1.0, 0.5, 0.5],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "O-Phosphocholine": {
        "peaks_ppm": [3.22, 3.58, 4.17],
        "multiplicities": ["s", "m", "m"],
        "relative_intensities": [1.0, 0.6, 0.5],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "N-Acetylcysteine": {
        "peaks_ppm": [2.03, 2.94, 4.40],
        "multiplicities": ["s", "dd", "dd"],
        "relative_intensities": [1.0, 0.7, 0.6],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "4-Aminobutyrate": {
        "peaks_ppm": [1.90, 2.30, 3.01],
        "multiplicities": ["quintet", "t", "t"],
        "relative_intensities": [0.8, 1.0, 0.9],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Chlorogenate": {
        "peaks_ppm": [6.33, 6.87, 7.06, 7.56],
        "multiplicities": ["d", "d", "dd", "d"],
        "relative_intensities": [0.8, 0.7, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Nicotinate": {
        "peaks_ppm": [7.50, 8.25, 8.62, 8.94],
        "multiplicities": ["dd", "d", "dd", "s"],
        "relative_intensities": [0.7, 0.8, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "S-Adenosylhomocysteine": {
        "peaks_ppm": [6.05, 8.18, 8.34],
        "multiplicities": ["d", "s", "s"],
        "relative_intensities": [0.7, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
    "Xanthurenate": {
        "peaks_ppm": [6.98, 7.28, 7.58, 8.02],
        "multiplicities": ["d", "t", "d", "s"],
        "relative_intensities": [0.8, 0.7, 0.9, 1.0],
        "solvent": "D2O",
        "ph": 7.0,
        "frequency_mhz": 600
    },
}


def get_compound_data(compound_name):
    """ดึงข้อมูล NMR ของสารจาก HMDB"""
    if compound_name in HMDB_NMR_DATA:
        return {
            "hmdb_id": HMDB_COMPOUND_IDS.get(compound_name, "N/A"),
            "data": HMDB_NMR_DATA[compound_name]
        }
    return None


def get_all_compounds():
    """ดึงรายชื่อสารทั้งหมด"""
    return list(HMDB_NMR_DATA.keys())


def get_compounds_by_region(ppm_low, ppm_high):
    """หาสารที่มี peak ใน ppm range ที่กำหนด"""
    results = []
    for compound, data in HMDB_NMR_DATA.items():
        for ppm in data["peaks_ppm"]:
            if ppm_low <= ppm <= ppm_high:
                results.append(compound)
                break
    return results