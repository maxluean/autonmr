"""
Annotator Module
เปรียบเทียบ detected peaks กับ reference library (Phase 1)
เพื่อระบุสารใน Plant Extract A-N [1]
"""

import json
import numpy as np


class Annotator:
    """
    Template Matching Annotator
    เปรียบเทียบ peaks กับ reference_library.json (Phase 1)
    """

    def __init__(self,
                 library_path='database/reference_library.json',
                 tolerance_ppm=0.03,
                 min_match_ratio=0.4):
        """
        Parameters:
            library_path: path ของ reference library (Phase 1)
            tolerance_ppm: ค่าความคลาดเคลื่อนที่ยอมรับ
            min_match_ratio: สัดส่วนขั้นต่ำของ peaks ที่ต้อง match
        """
        self.library = self._load_library(library_path)
        self.tolerance = tolerance_ppm
        self.min_match_ratio = min_match_ratio

    def _load_library(self, path):
        """โหลด reference library จาก Phase 1"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"Library loaded: {len(data['compounds'])} compounds")
            return data['compounds']
        except FileNotFoundError:
            print(f"Library not found: {path}")
            return {}

    def annotate(self, detected_peaks):
        """
        Annotate detected peaks

        Parameters:
            detected_peaks: dict จาก PeakDetector

        Returns:
            list: [
                {
                    'compound': ชื่อสาร,
                    'overall_score': คะแนนรวม,
                    'position_score': คะแนนตำแหน่ง,
                    'intensity_score': คะแนน intensity,
                    'coverage_score': คะแนน coverage,
                    'matched_peaks': peaks ที่ match,
                    'num_matched': จำนวน peaks ที่ match,
                    'num_ref_peaks': จำนวน peaks ใน library
                }
            ]
        """
        results = []

        query_ppm = detected_peaks['ppm']
        query_int = detected_peaks['intensity']

        if len(query_ppm) == 0:
            print("No peaks detected!")
            return []

        # เปรียบเทียบกับสารทุกชนิดใน library
        for compound, ref_data in self.library.items():
            if ref_data is None:
                continue

            ref_peaks = ref_data.get('peaks', [])
            if not ref_peaks:
                continue

            # คำนวณ scores
            match_result = self._match_compound(
                query_ppm, query_int,
                ref_peaks, compound
            )

            if match_result['match_ratio'] >= self.min_match_ratio:
                # คำนวณ overall score
                overall = self._calculate_overall_score(
                    match_result
                )
                match_result['overall_score'] = overall
                match_result['compound'] = compound

                # เพิ่ม database confidence
                db_conf = ref_data.get('overall_confidence', 0.5)
                match_result['db_confidence'] = db_conf

                # Final score
                match_result['final_score'] = (
                    overall * 0.8 + db_conf * 0.2
                )

                results.append(match_result)

        # Sort by final score
        results.sort(
            key=lambda x: x['final_score'],
            reverse=True
        )

        return results

    def _match_compound(self, query_ppm, query_int,
                        ref_peaks, compound_name):
        """
        จับคู่ detected peaks กับ reference compound
        """
        ref_ppms = [p['ppm'] for p in ref_peaks]
        ref_ints = [p['intensity'] for p in ref_peaks]

        matched_pairs = []
        position_errors = []
        intensity_diffs = []

        for ref_ppm, ref_int in zip(ref_ppms, ref_ints):
            # หา peak ที่ใกล้ที่สุดใน query
            distances = np.abs(query_ppm - ref_ppm)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist <= self.tolerance:
                matched_pairs.append({
                    'query_ppm': float(query_ppm[min_idx]),
                    'ref_ppm': ref_ppm,
                    'error_ppm': float(min_dist),
                    'query_intensity': float(query_int[min_idx]),
                    'ref_intensity': ref_int
                })
                position_errors.append(min_dist)

                # Intensity difference
                max_q_int = np.max(query_int)
                if max_q_int > 0:
                    norm_q = query_int[min_idx] / max_q_int
                    intensity_diffs.append(abs(norm_q - ref_int))

        num_matched = len(matched_pairs)
        num_ref = len(ref_peaks)
        match_ratio = num_matched / num_ref if num_ref > 0 else 0

        # Position score
        if position_errors:
            avg_error = np.mean(position_errors)
            position_score = 1 - (avg_error / self.tolerance)
        else:
            position_score = 0.0

        # Intensity score
        if intensity_diffs:
            intensity_score = 1 - np.mean(intensity_diffs)
        else:
            intensity_score = 0.0

        # Coverage score
        coverage_score = match_ratio

        return {
            'matched_peaks': matched_pairs,
            'num_matched': num_matched,
            'num_ref_peaks': num_ref,
            'match_ratio': match_ratio,
            'position_score': round(float(position_score), 4),
            'intensity_score': round(float(intensity_score), 4),
            'coverage_score': round(float(coverage_score), 4)
        }

    def _calculate_overall_score(self, match_result):
        """
        คำนวณ overall score จาก 3 components
        Position (0.45) + Intensity (0.30) + Coverage (0.25)
        """
        score = (
            0.45 * match_result['position_score'] +
            0.30 * match_result['intensity_score'] +
            0.25 * match_result['coverage_score']
        )
        return round(float(score), 4)