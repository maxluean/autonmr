"""
Random Forest Annotator
Train Random Forest สำหรับ NMR annotation
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score
)


class RandomForestAnnotator:
    """
    Random Forest สำหรับ annotate NMR spectrum [1]
    Input : feature vector (จาก FeatureExtractor)
    Output: compound predictions (multi-label)
    """

    def __init__(self, compound_names):
        self.compound_names = compound_names
        self.model = None
        self.threshold = 0.3

    def train(self, X_feat, y,
              test_size=0.2, random_state=42):
        """
        Train Random Forest model

        Parameters:
            X_feat: feature matrix (n_samples, n_features)
            y     : label matrix (n_samples, n_compounds)
        """
        print("\nTraining Random Forest...")
        print("=" * 50)

        # Split data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_feat, y,
            test_size=test_size,
            random_state=random_state
        )
        print(f"  Train: {len(X_tr)} samples")
        print(f"  Val  : {len(X_val)} samples")

        # Train model
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
        self.model = MultiOutputClassifier(base_rf, n_jobs=-1)
        self.model.fit(X_tr, y_tr)

        # Evaluate
        y_pred = self.model.predict(X_val)
        metrics = self._evaluate(y_val, y_pred)

        print(f"\n  Results:")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print("=" * 50)

        return metrics

    def predict(self, X_feat):
        """
        ทำนาย compounds จาก feature vector

        Returns:
            list: [{'compound': name, 'rf_score': prob}, ...]
        """
        if self.model is None:
            return []

        # predict_proba จาก MultiOutputClassifier
        probas = self.model.predict_proba(X_feat)

        results = []
        for i, compound in enumerate(self.compound_names):
            # probas[i] = array of [prob_0, prob_1]
            if len(probas[i][0]) > 1:
                prob = float(probas[i][0][1])
            else:
                prob = float(probas[i][0][0])

            if prob >= self.threshold:
                results.append({
                    'compound': compound,
                    'rf_score': round(prob, 4)
                })

        return sorted(
            results,
            key=lambda x: x['rf_score'],
            reverse=True
        )

    def _evaluate(self, y_true, y_pred):
        """คำนวณ metrics"""
        return {
            'f1': f1_score(
                y_true, y_pred,
                average='weighted',
                zero_division=0
            ),
            'precision': precision_score(
                y_true, y_pred,
                average='weighted',
                zero_division=0
            ),
            'recall': recall_score(
                y_true, y_pred,
                average='weighted',
                zero_division=0
            )
        }

    def save(self, path='data/trained_models/rf_model.pkl'):
        """บันทึก model"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'compound_names': self.compound_names,
                'threshold': self.threshold
            }, f)
        print(f"RF model saved: {path}")

    def load(self, path='data/trained_models/rf_model.pkl'):
        """โหลด model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.compound_names = data['compound_names']
        self.threshold = data['threshold']
        print(f"RF model loaded: {path}")