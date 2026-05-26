"""
Core Engine Module — Phase 2
AutoNMR: Automated NMR Annotation System

แทนที่กระบวนการ manual ทีละภาพ [1]
ด้วย automated pipeline ที่รันได้ทันที

Files:
    preprocessor.py      → Denoise, Baseline, Normalize, Align
    peak_detector.py     → Auto peak detection
    annotator.py         → Template matching vs reference library
    confidence_scorer.py → MSI Level 1-4 scoring
"""

from core.preprocessor import NMRPreprocessor
from core.peak_detector import PeakDetector
from core.annotator import Annotator
from core.confidence_scorer import ConfidenceScorer

__all__ = [
    'NMRPreprocessor',
    'PeakDetector',
    'Annotator',
    'ConfidenceScorer'
]