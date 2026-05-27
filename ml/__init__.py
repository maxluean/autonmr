"""
ML/AI Module — Phase 3
AutoNMR: Automated NMR Annotation System

Train ML models จาก synthetic data
ที่สร้างจาก reference library (Phase 1)
เพื่อ annotate Plant Extract A-N [1]

Files:
    synthetic_generator.py  → สร้าง training data
    feature_extractor.py    → Feature engineering
    random_forest_model.py  → Train Random Forest
    cnn_model.py            → Train 1D CNN
    ensemble.py             → รวม Template + RF + CNN
"""

# from ml.synthetic_generator import SyntheticGenerator
from ml.feature_extractor import FeatureExtractor
from ml.random_forest_model import RandomForestAnnotator
from ml.cnn_model import CNNAnnotator
from ml.ensemble import EnsembleAnnotator
# from ml.trainer import Trainer

__all__ = [
    'FeatureExtractor',
    'RandomForestAnnotator',
    'CNNAnnotator',
    'EnsembleAnnotator'
]