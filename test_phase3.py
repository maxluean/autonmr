"""
ทดสอบ Phase 3: ML/AI Training + Ensemble
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.trainer import Trainer
from ml.synthetic_generator import SyntheticGenerator
from ml.feature_extractor import FeatureExtractor
from ml.random_forest_model import RandomForestAnnotator
from ml.cnn_model import CNNAnnotator
from ml.ensemble import EnsembleAnnotator
from core.annotator import Annotator
from core.preprocessor import NMRPreprocessor
from core.peak_detector import PeakDetector
from core.confidence_scorer import ConfidenceScorer


def run_phase3_test():
    """
    รัน Phase 3 ครบวงจร
    """
    print("\n" + "=" * 60)
    print("  AutoNMR — Phase 3: ML/AI Training Test")
    print("=" * 60)

    # ===== Step 1: Train Models =====
    trainer = Trainer(
        library_path='database/reference_library.json',
        n_samples=2000,
        cnn_epochs=30
    )
    train_results = trainer.run()

    # ===== Step 2: สร้าง Test Spectrum =====
    print("\n[Test] Creating test spectrum...")
    generator = SyntheticGenerator()
    test_compounds = [
        'Sucrose', 'Glucose', 'Alanine',
        'Formate', 'Citrate', 'Tryptophan',
        'Tyrosine', 'Maltose'
    ]
    ppm_axis, test_spectrum, true_labels = \
        generator.generate_spectrum(test_compounds)

    print(f"  Test compounds: {test_compounds}")

    # ===== Step 3: Preprocess =====
    preprocessor = NMRPreprocessor('plant_extract')
    processed = preprocessor.process(test_spectrum, ppm_axis)

    # ===== Step 4: Detect Peaks =====
    detector = PeakDetector()
    peaks = detector.detect(
        processed['processed'], processed['ppm_axis']
    )

    # ===== Step 5: Template Matching (Phase 2) =====
    annotator = Annotator(
        library_path='database/reference_library.json'
    )
    template_results = annotator.annotate(peaks)

    # ===== Step 6: RF Prediction =====
    print("\n[RF] Predicting...")
    rf_annotator = RandomForestAnnotator(
        train_results['compound_names']
    )
    rf_annotator.load()

    extractor = FeatureExtractor(ppm_axis=ppm_axis)
    feat = extractor.extract(
        processed['processed'], ppm_axis
    ).reshape(1, -1)
    rf_results = rf_annotator.predict(feat)

    # ===== Step 7: CNN Prediction =====
    print("\n[CNN] Predicting...")
    cnn_annotator = CNNAnnotator(
        train_results['compound_names']
    )
    cnn_annotator.load()
    cnn_results = cnn_annotator.predict(
        processed['processed']
    )

    # ===== Step 8: Ensemble =====
    print("\n[Ensemble] Combining results...")
    ensemble = EnsembleAnnotator(
        train_results['compound_names']
    )
    final_results = ensemble.combine(
        template_results, rf_results, cnn_results
    )

    # ===== Step 9: Confidence Scoring =====
    scorer = ConfidenceScorer()
    scored = scorer.score(final_results)
    report = scorer.generate_report(
        scored, subject_name='Test (Plant Extract) [1]'
    )

    # ===== Step 10: Validation =====
    print("\n[Validation] vs Known compounds:")
    detected = [r['compound'] for r in final_results[:20]]

    for comp in test_compounds:
        found = comp in detected
        status = "FOUND" if found else "MISSED"
        print(f"  {status}: {comp}")

    found_count = sum(
        1 for c in test_compounds if c in detected
    )
    accuracy = found_count / len(test_compounds) * 100
    print(f"\n  Accuracy: {found_count}/{len(test_compounds)}"
          f" ({accuracy:.1f}%)")

    # ===== Step 11: แสดง Ensemble Scores =====
    print("\n  Top 10 Ensemble Results:")
    print(f"  {'Compound':<25} {'Ensemble':>8}"
          f" {'Template':>9} {'RF':>6} {'CNN':>6}"
          f" {'Methods':>8}")
    print("  " + "-" * 70)

    for r in final_results[:10]:
        methods = "+".join(r.get('methods_agreed', []))
        print(
            f"  {r['compound']:<25}"
            f" {r['ensemble_score']:>8.4f}"
            f" {r.get('template_score', 0):>9.4f}"
            f" {r.get('rf_score', 0):>6.4f}"
            f" {r.get('cnn_score', 0):>6.4f}"
            f" {methods:>8}"
        )

    print("\n" + "=" * 60)
    print("  Phase 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_phase3_test()