import os
import json
import sys

# ชี้ Path ไปหาฐานข้อมูลในโปรเจคของเรา
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from database.hmdb_data import HMDB_NMR_DATA, HMDB_COMPOUND_IDS

def generate_local_hmdb_data(output_dir='data/hmdb_spectra'):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("Generating HMDB experimental data from local verified database...")
    print("=" * 60)

    for compound, data in HMDB_NMR_DATA.items():
        hmdb_id = HMDB_COMPOUND_IDS.get(compound, "Unknown")
        
        # จัดรูปแบบข้อมูลให้ตรงกับที่ build_real_dataset.py ต้องการ
        peaks = []
        for ppm, intensity in zip(data['peaks_ppm'], data['relative_intensities']):
            peaks.append({
                'ppm': ppm,
                'intensity': intensity
            })

        results[compound] = {
            'compound': compound,
            'hmdb_id': hmdb_id,
            'peaks': peaks,
            'source': 'HMDB_verified_local',
            'num_peaks': len(peaks)
        }
        print(f"Added: {compound:<20} ({hmdb_id}) -> {len(peaks)} peaks")

    output_path = os.path.join(output_dir, 'hmdb_experimental.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"Done: {len(results)}/40 compounds processed [1]")
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_local_hmdb_data()