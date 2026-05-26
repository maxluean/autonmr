"""
Ensemble Annotator
รวม Template Matching + Random Forest + CNN
เพื่อ annotation ที่แม่นยำที่สุด [1]
"""

import numpy as np


class EnsembleAnnotator:
    """
    รวมผลจาก 3 methods:
    Template Matching (Phase 2) + RF + CNN
    """

    # Weights ของแต่ละ method
    WEIGHTS = {
        'template': 0.40,
        'rf':       0.30,
        'cnn':      0.30
    }

    def __init__(self, compound_names):
        self.compound_names = compound_names

    def combine(
        self,
        template_results,
        rf_results,
        cnn_results
    ):
        """
        รวมผลจากทั้ง 3 methods

        Parameters:
            template_results : list จาก Annotator (Phase 2)
            rf_results       : list จาก RandomForestAnnotator
            cnn_results      : list จาก CNNAnnotator

        Returns:
            list: ensemble predictions
        """
        # รวมทุก compound ที่พบจากทุก method
        all_compounds = {}

        # Template Matching scores
        for r in template_results:
            comp = r['compound']
            score = r.get('final_score', 0)
            if comp not in all_compounds:
                all_compounds[comp] = {
                    'template': 0,
                    'rf': 0,
                    'cnn': 0,
                    'methods': []
                }
            all_compounds[comp]['template'] = score
            all_compounds[comp]['methods'].append('template')

        # RF scores
        for r in rf_results:
            comp = r['compound']
            score = r.get('rf_score', 0)
            if comp not in all_compounds:
                all_compounds[comp] = {
                    'template': 0,
                    'rf': 0,
                    'cnn': 0,
                    'methods': []
                }
            all_compounds[comp]['rf'] = score
            all_compounds[comp]['methods'].append('rf')

        # CNN scores
        for r in cnn_results:
            comp = r['compound']
            score = r.get('cnn_score', 0)
            if comp not in all_compounds:
                all_compounds[comp] = {
                    'template': 0,
                    'rf': 0,
                    'cnn': 0,
                    'methods': []
                }
            all_compounds[comp]['cnn'] = score
            all_compounds[comp]['methods'].append('cnn')

        # คำนวณ ensemble score
        ensemble_results = []
        for compound, scores in all_compounds.items():

            # Weighted sum
            ensemble_score = (
                self.WEIGHTS['template'] * scores['template']
                + self.WEIGHTS['rf'] * scores['rf']
                + self.WEIGHTS['cnn'] * scores['cnn']
            )

            # Bonus: พบจากหลาย methods → น่าเชื่อถือกว่า
            unique_methods = set(scores['methods'])
            if len(unique_methods) >= 3:
                ensemble_score *= 1.20  # +20%
            elif len(unique_methods) == 2:
                ensemble_score *= 1.10  # +10%

            ensemble_score = min(1.0, ensemble_score)

            ensemble_results.append({
                'compound': compound,
                'ensemble_score': round(ensemble_score, 4),
                'template_score': round(scores['template'], 4),
                'rf_score': round(scores['rf'], 4),
                'cnn_score': round(scores['cnn'], 4),
                'methods_agreed': list(unique_methods),
                'num_methods': len(unique_methods)
            })

        # Sort by ensemble score
        ensemble_results.sort(
            key=lambda x: x['ensemble_score'],
            reverse=True
        )

        return ensemble_results