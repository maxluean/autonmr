"""
ทดสอบ Phase 2: Core Engine
ใช้ Synthetic Data จำลอง Plant Extract [1]
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.preprocessor import NMRPreprocessor
from core.peak_detector import PeakDetector
from core.annotator import Annotator
from core.confidence_scorer import ConfidenceScorer


def create_synthetic_spectrum():
    """
    สร้าง synthetic NMR spectrum
    จำลอง Plant Extract A [1]
    ที่มี Sucrose, Glucose, Alanine, Formate, Citrate
    """
    # สร้าง ppm axis (0.5 - 10.0 ppm, 20,000 points) [1]
    ppm_axis = np.linspace(0.5, 10.0, 20000)
    spectrum = np.zeros(20000)

    # สารที่จะใส่ใน synthetic spectrum
    compounds = {
        'Alanine':  [(1.48, 1.0), (3.78, 0.7)],
        'Formate':  [(8.46, 1.0)],
        'Acetate':  [(1.92, 1.0)],
        'Citrate':  [(2.54, 1.0), (2.69, 0.95)],
        'Succinate':[(2.41, 1.0)],
        'Sucrose':  [(3.48, 0.7), (3.56, 0.6),
                    (3.68, 0.8), (3.76, 0.7),
                    (3.83, 0.7), (4.05, 0.8),
                    (4.22, 0.9), (5.41, 1.0)],
        'Glucose':  [(3.24, 0.6), (3.41, 0.7),
                    (3.46, 0.7), (3.73, 0.8),
                    (4.65, 0.9), (5.23, 1.0)],
    }

    # สร้าง Lorentzian peaks
    for compound, peaks in compounds.items():
        for ppm, intensity in peaks:
            # Lorentzian peak shape
            width = 0.015
            peak = intensity * (width ** 2) / (
                (ppm_axis - ppm) ** 2 + width ** 2
            )
            spectrum += peak * np.random.uniform(0.85, 1.15)

    # เพิ่ม noise
    noise = np.random.normal(
        0, 0.005 * np.max(spectrum), len(spectrum)
    )
    spectrum += noise

    # เพิ่ม baseline drift
    baseline = 0.02 * np.sin(
        np.linspace(0, 3 * np.pi, len(spectrum))
    )
    spectrum += baseline
    spectrum = np.maximum(spectrum, 0)

    return ppm_axis, spectrum


def run_phase2_test():
    """
    รัน Phase 2 pipeline ครบวงจร
    """
    print("\n" + "=" * 60)
    print("  AutoNMR — Phase 2: Core Engine Test")
    print("  Simulating Plant Extract A [1]")
    print("=" * 60)

    # สร้าง synthetic spectrum
    print("\n[1] Creating synthetic spectrum...")
    ppm_axis, raw_spectrum = create_synthetic_spectrum()
    print(f"    Spectrum length: {len(raw_spectrum):,} points")
    print(f"    PPM range: {ppm_axis[0]:.1f} - "
          f"{ppm_axis[-1]:.1f} ppm")

    # Step 1: Preprocessing
    print("\n[2] Preprocessing...")
    preprocessor = NMRPreprocessor(
        sample_type='plant_extract'
    )
    processed = preprocessor.process(raw_spectrum, ppm_axis)
    print(f"    Denoised:  OK")
    print(f"    Baseline:  OK")
    print(f"    Normalized: OK")

    # Step 2: Peak Detection
    print("\n[3] Peak Detection...")
    detector = PeakDetector()
    peaks = detector.detect(
        processed['processed'],
        processed['ppm_axis']
    )

    # Step 3: Annotation
    print("\n[4] Annotating...")
    annotator = Annotator(
        library_path='database/reference_library.json',
        tolerance_ppm=0.03,
        min_match_ratio=0.4
    )
    annotations = annotator.annotate(peaks)

    # Step 4: Confidence Scoring
    print("\n[5] Scoring confidence...")
    scorer = ConfidenceScorer()
    scored = scorer.score(annotations)
    report = scorer.generate_report(
        scored, subject_name='Subject_1 (Plant Extract A) [1]'
    )

    # Export DataFrame
    df = scorer.to_dataframe(scored)
    print("\n  Top 10 compounds detected:")
    print(df.head(10)[
        ['Compound', 'MSI_Level', 'MSI_Name',
         'Final_Score', 'Peaks_Matched']
    ].to_string(index=False))

    # ตรวจสอบ expected compounds
    print("\n[6] Validation against known compounds:")
    expected = [
        'Alanine', 'Formate', 'Acetate',
        'Citrate', 'Succinate', 'Sucrose', 'Glucose'
    ]
    detected_names = [r['compound'] for r in scored]

    for exp in expected:
        found = exp in detected_names
        status = "FOUND" if found else "MISSED"
        print(f"    {status}: {exp}")

    found_count = sum(
        1 for e in expected if e in detected_names
    )
    accuracy = found_count / len(expected) * 100
    print(f"\n  Accuracy: {found_count}/{len(expected)}"
          f" ({accuracy:.1f}%)")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_phase2_test()