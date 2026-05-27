# retrain_with_real_data.py

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.abspath(__file__)
))

from ml.random_forest_model import RandomForestAnnotator
from ml.cnn_model import CNNAnnotator
from ml.feature_extractor import FeatureExtractor


def retrain():
    """
    Re-train RF + CNN ด้วย real HMDB data
    แทน synthetic data
    """
    print("\n" + "=" * 60)
    print("  Re-training with Real HMDB Data")
    print("=" * 60)

    # โหลด real dataset
    print("\nLoading real dataset...")
    X = np.load('data/datasets/real_mixture_X.npy')
    y = np.load('data/datasets/real_mixture_y.npy')
    compound_names = np.load(
        'data/datasets/real_mixture_compounds.npy',
        allow_pickle=True
    ).tolist()

    ppm_axis = np.linspace(0.5, 10.0, X.shape[1])

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Compounds: {len(compound_names)}")

    # Re-train Random Forest
    print("\nRe-training Random Forest...")
    extractor = FeatureExtractor(ppm_axis=ppm_axis)
    X_feat = extractor.extract_batch(X, ppm_axis)

    rf = RandomForestAnnotator(compound_names)
    rf_metrics = rf.train(X_feat, y)
    rf.save('data/trained_models/rf_real.pkl')

    # Re-train CNN
    print("\nRe-training CNN...")
    cnn = CNNAnnotator(compound_names, X.shape[1])
    cnn.train(X, y, epochs=30)
    cnn.save('data/trained_models/cnn_real.pt')

    print("\n" + "=" * 60)
    print("  Re-training Complete!")
    print(f"  RF F1-Score: {rf_metrics['f1']:.4f}")
    print(f"  Saved: data/trained_models/rf_real.pkl")
    print(f"  Saved: data/trained_models/cnn_real.pt")
    print("=" * 60)


if __name__ == "__main__":
    retrain()