"""
Confidence Scorer Module
ให้คะแนน annotation ตามมาตรฐาน MSI (Metabolomics Standards Initiative)

MSI Levels:
    Level 1 (Confirmed)  → แน่ใจมาก
    Level 2 (Probable)   → น่าจะใช่
    Level 3 (Putative)   → อาจจะใช่
    Level 4 (Tentative)  → ไม่แน่ใจ
"""

import numpy as np
import pandas as pd
from datetime import datetime


class ConfidenceScorer:
    """
    ให้ MSI Confidence Level กับ annotation results
    """

    # MSI Thresholds
    THRESHOLDS = {
        'level_1': 0.85,
        'level_2': 0.65,
        'level_3': 0.45,
    }

    # MSI Level descriptions
    LEVELS = {
        1: {
            'name': 'Confirmed',
            'description': 'จับคู่ได้กับ >= 3 databases '
                         '+ intensity match',
            'color': 'green',
            'icon': '[L1]'
        },
        2: {
            'name': 'Probable',
            'description': 'จับคู่ได้กับ 2 databases',
            'color': 'yellow',
            'icon': '[L2]'
        },
        3: {
            'name': 'Putative',
            'description': 'จับคู่ได้กับ 1 database',
            'color': 'orange',
            'icon': '[L3]'
        },
        4: {
            'name': 'Tentative',
            'description': 'ไม่แน่ใจ ต้องการ expert validate',
            'color': 'red',
            'icon': '[L4]'
        }
    }

    def score(self, annotation_results):
        """
        ให้ MSI Level กับ annotation results ทั้งหมด

        Parameters:
            annotation_results: list จาก Annotator

        Returns:
            list: annotation_results พร้อม MSI Level
        """
        scored_results = []

        for result in annotation_results:
            scored = result.copy()
            final_score = result.get('final_score', 0)
            db_conf = result.get('db_confidence', 0.5)

            # กำหนด MSI Level
            level = self._determine_level(
                final_score, db_conf
            )
            scored['msi_level'] = level
            scored['msi_name'] = self.LEVELS[level]['name']
            scored['msi_description'] = \
                self.LEVELS[level]['description']
            scored['msi_icon'] = self.LEVELS[level]['icon']

            scored_results.append(scored)

        return scored_results

    def _determine_level(self, score, db_confidence):
        """
        กำหนด MSI Level จาก score และ db_confidence
        """
        if score >= self.THRESHOLDS['level_1'] \
                and db_confidence >= 0.8:
            return 1
        elif score >= self.THRESHOLDS['level_2'] \
                and db_confidence >= 0.5:
            return 2
        elif score >= self.THRESHOLDS['level_3']:
            return 3
        else:
            return 4

    def generate_report(self, scored_results,
                        subject_name='Unknown'):
        """
        สร้าง annotation report สำหรับ Subject 1-N [1]

        Parameters:
            scored_results: list จาก score()
            subject_name: ชื่อ Subject เช่น 'Subject_1'

        Returns:
            dict: report สรุปผล
        """
        if not scored_results:
            return {'error': 'No annotations found'}

        # แยกตาม MSI Level
        by_level = {1: [], 2: [], 3: [], 4: []}
        for r in scored_results:
            level = r.get('msi_level', 4)
            by_level[level].append(r)

        report = {
            'subject': subject_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_compounds': len(scored_results),
                'level_1_confirmed': len(by_level[1]),
                'level_2_probable': len(by_level[2]),
                'level_3_putative': len(by_level[3]),
                'level_4_tentative': len(by_level[4]),
            },
            'compounds_by_level': {
                'level_1': [
                    r['compound'] for r in by_level[1]
                ],
                'level_2': [
                    r['compound'] for r in by_level[2]
                ],
                'level_3': [
                    r['compound'] for r in by_level[3]
                ],
                'level_4': [
                    r['compound'] for r in by_level[4]
                ],
            },
            'detailed_results': scored_results
        }

        self._print_report(report)
        return report

    def _print_report(self, report):
        """แสดง report สรุป"""
        print("\n" + "=" * 60)
        print(f"  ANNOTATION REPORT: {report['subject']}")
        print("=" * 60)
        print(f"  Total compounds: "
              f"{report['summary']['total_compounds']}")
        print(f"  [L1] Confirmed : "
              f"{report['summary']['level_1_confirmed']}")
        print(f"  [L2] Probable  : "
              f"{report['summary']['level_2_probable']}")
        print(f"  [L3] Putative  : "
              f"{report['summary']['level_3_putative']}")
        print(f"  [L4] Tentative : "
              f"{report['summary']['level_4_tentative']}")
        print("=" * 60)

        # แสดง Level 1
        if report['compounds_by_level']['level_1']:
            print("\n  [L1] CONFIRMED:")
            for comp in report['compounds_by_level']['level_1']:
                print(f"       • {comp}")

        # แสดง Level 2
        if report['compounds_by_level']['level_2']:
            print("\n  [L2] PROBABLE:")
            for comp in report['compounds_by_level']['level_2']:
                print(f"       • {comp}")

        print("=" * 60)

    def to_dataframe(self, scored_results):
        """
        แปลง results เป็น DataFrame สำหรับ export
        """
        rows = []
        for r in scored_results:
            rows.append({
                'Compound': r.get('compound', ''),
                'MSI_Level': r.get('msi_level', 4),
                'MSI_Name': r.get('msi_name', ''),
                'Final_Score': r.get('final_score', 0),
                'Position_Score': r.get('position_score', 0),
                'Intensity_Score': r.get('intensity_score', 0),
                'Coverage_Score': r.get('coverage_score', 0),
                'Peaks_Matched': r.get('num_matched', 0),
                'Peaks_Reference': r.get('num_ref_peaks', 0),
                'DB_Confidence': r.get('db_confidence', 0)
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                ['MSI_Level', 'Final_Score'],
                ascending=[True, False]
            ).reset_index(drop=True)

        return df