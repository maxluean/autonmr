"""
Trainer Module
Training pipeline ครบวงจรสำหรับ Phase 3
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from ml.synthetic_generator import SyntheticGenerator
from ml.feature_extractor import FeatureExtractor
from ml.random_forest_model import RandomForestAnnotator
from ml.cnn_model import CNNAnnotator


class Trainer:
    """
    Training pipeline ครบวงจร:
    1. สร้าง synthetic data
    2. Extract features
    3. Train RF
    4. Train CNN
    5. Save models
    """

    def __init__(
        self,
        library_path='database/reference_library.json',
        n_samples=2000,
        cnn_epochs=30
    ):
        self.library_path = library_path
        self.n_samples = n_samples
        self.cnn_epochs = cnn_epochs

    def run(self):
        """
        รัน training pipeline ทั้งหมด
        """
        print("\n" + "=" * 60)
        print("  AutoNMR — Phase 3: ML/AI Training")
        print("  Plant Extract A-N [1]")
        print("=" * 60)

        # Step 1: สร้าง Synthetic Data
        print("\n[Step 1] Generating synthetic data...")
        generator = SyntheticGenerator(
            library_path=self.library_path
        )
        dataset = generator.build_training_set(
            n_samples=self.n_samples
        )

        X = dataset['X']
        y = dataset['y']
        ppm_axis = dataset['ppm_axis']
        compound_names = dataset['compound_names']

        # Step 2: Extract Features (สำหรับ RF)
        print("\n[Step 2] Extracting features for RF...")
        extractor = FeatureExtractor(ppm_axis=ppm_axis)
        X_feat = extractor.extract_batch(X, ppm_axis)

        # Step 3: Train Random Forest
        print("\n[Step 3] Training Random Forest...")
        rf_annotator = RandomForestAnnotator(compound_names)
        rf_metrics = rf_annotator.train(X_feat, y)
        rf_annotator.save()

        # Step 4: Train CNN
        print("\n[Step 4] Training 1D CNN...")
        cnn_annotator = CNNAnnotator(
            compound_names,
            spectrum_length=X.shape[1]
        )
        cnn_annotator.train(
            X, y, epochs=self.cnn_epochs
        )
        cnn_annotator.save()

        # Summary
        print("\n" + "=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print(f"  Samples trained  : {self.n_samples}")
        print(f"  Compounds        : {len(compound_names)}")
        print(f"  RF F1-Score      : {rf_metrics['f1']:.4f}")
        print(f"  RF Precision     : {rf_metrics['precision']:.4f}")
        print(f"  RF Recall        : {rf_metrics['recall']:.4f}")
        print(f"  Models saved to  : data/trained_models/")
        print("=" * 60)

        return {
            'rf_metrics': rf_metrics,
            'compound_names': compound_names,
            'n_samples': self.n_samples
        }