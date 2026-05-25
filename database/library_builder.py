"""
Library Builder Module
สร้าง Cross-validated Reference Library จาก HMDB + BMRB + NP-MRD
ใช้สำหรับ annotation สาร 40+ ชนิดที่พบใน Plant Extract A-N [1]
"""

import json
import numpy as np
from database.hmdb_data import HMDB_NMR_DATA, HMDB_COMPOUND_IDS
from database.bmrb_data import BMRB_NMR_DATA
from database.npmrd_data import NPMRD_NMR_DATA


class LibraryBuilder:
    """
    สร้าง cross-validated reference library
    จาก multiple public NMR databases
    """
    
    def __init__(self, tolerance_ppm=0.02):
        """
        tolerance_ppm: ค่าความคลาดเคลื่อนที่ยอมรับได้
                       สำหรับ cross-validation ระหว่าง databases
        """
        self.tolerance = tolerance_ppm
        self.hmdb = HMDB_NMR_DATA
        self.bmrb = BMRB_NMR_DATA
        self.npmrd = NPMRD_NMR_DATA
    
    def build(self):
        """
        สร้าง comprehensive reference library
        
        Returns:
            dict: Cross-validated library ของสาร 40+ ชนิด [1]
        """
        library = {}
        
        for compound in self.hmdb.keys():
            entry = self._build_compound_entry(compound)
            library[compound] = entry
        
        return library
    
    def _build_compound_entry(self, compound):
        """
        สร้าง entry สำหรับสาร 1 ชนิด
        Cross-validate chemical shifts ระหว่าง databases
        """
        hmdb_data = self.hmdb.get(compound)
        bmrb_data = self.bmrb.get(compound)
        npmrd_data = self.npmrd.get(compound)
        
        if not hmdb_data:
            return None
        
        # นับจำนวน databases ที่มีข้อมูลสารนี้
        sources = ["HMDB"]
        if bmrb_data:
            sources.append("BMRB")
        if npmrd_data:
            sources.append("NP-MRD")
        
        # Cross-validate each peak
        validated_peaks = []
        for i, ppm in enumerate(hmdb_data["peaks_ppm"]):
            peak_sources = ["HMDB"]
            
            # ตรวจกับ BMRB
            if bmrb_data:
                for bmrb_ppm in bmrb_data["peaks_ppm"]:
                    if abs(ppm - bmrb_ppm) <= self.tolerance:
                        peak_sources.append("BMRB")
                        break
            
            # ตรวจกับ NP-MRD
            if npmrd_data:
                for npmrd_ppm in npmrd_data["peaks_ppm"]:
                    if abs(ppm - npmrd_ppm) <= self.tolerance:
                        peak_sources.append("NP-MRD")
                        break
            
            # คำนวณ confidence ตามจำนวน sources
            peak_confidence = len(peak_sources) / 3.0
            
            validated_peaks.append({
                "ppm": ppm,
                "multiplicity": hmdb_data["multiplicities"][i],
                "intensity": hmdb_data["relative_intensities"][i],
                "sources_validated": peak_sources,
                "num_sources": len(peak_sources),
                "confidence": round(peak_confidence, 3)
            })
        
        # Overall compound confidence
        avg_confidence = np.mean(
            [p["confidence"] for p in validated_peaks]
        )
        
        return {
            "hmdb_id": HMDB_COMPOUND_IDS.get(compound, "N/A"),
            "peaks": validated_peaks,
            "num_peaks": len(validated_peaks),
            "sources_available": sources,
            "num_sources": len(sources),
            "overall_confidence": round(float(avg_confidence), 3),
            "solvent": hmdb_data.get("solvent", "D2O"),
            "ph": hmdb_data.get("ph", 7.0),
            "frequency_mhz": hmdb_data.get("frequency_mhz", 600)
        }
    
    def save_library(self, output_path="database/reference_library.json"):
        """
        สร้างและบันทึก library เป็น JSON
        """
        library = self.build()
        
        output = {
            "metadata": {
                "name": "AutoNMR Reference Library",
                "version": "1.0.0",
                "domain": "Domain 1: Pattern Identification [1]",
                "sources": ["HMDB", "BMRB", "NP-MRD"],
                "tolerance_ppm": self.tolerance,
                "num_compounds": len(library),
                "sample_type": "Plant Extract A-N [1]"
            },
            "compounds": library
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Library saved: {output_path}")
        print(f"Compounds: {len(library)}")
        print(f"Sources: HMDB + BMRB + NP-MRD")
        
        # Summary
        self._print_summary(library)
        
        return output
    
    def _print_summary(self, library):
        """แสดงสรุปผล library"""
        print("\n" + "=" * 60)
        print("  REFERENCE LIBRARY SUMMARY")
        print("=" * 60)
        
        high_conf = 0
        med_conf = 0
        low_conf = 0
        
        for compound, data in library.items():
            if data is None:
                continue
            conf = data["overall_confidence"]
            if conf >= 0.8:
                high_conf += 1
            elif conf >= 0.5:
                med_conf += 1
            else:
                low_conf += 1
        
        print(f"  High confidence (>= 0.8)  : {high_conf} compounds")
        print(f"  Medium confidence (>= 0.5): {med_conf} compounds")
        print(f"  Low confidence (< 0.5)    : {low_conf} compounds")
        print(f"  Total                      : {len(library)} compounds")
        print("=" * 60)


# ===== MAIN =====
if __name__ == "__main__":
    builder = LibraryBuilder(tolerance_ppm=0.02)
    builder.save_library("database/reference_library.json")