"""
Peak Detector Module
ตรวจจับ peaks อัตโนมัติจาก preprocessed NMR spectrum

แทนที่การ manual identify peaks ทีละภาพ [1]
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths


class PeakDetector:
    """
    ตรวจจับ peaks จาก NMR spectrum อัตโนมัติ
    รองรับ Plant Extract A-N [1]
    """

    def __init__(self):
        # NMR regions สำหรับ Plant Extract [1]
        self.regions = {
            'aliphatic': {
                'range': (0.5, 1.5),
                'compounds': [
                    'Isoleucine', 'Leucine',
                    'Valine', 'Ethanol'
                ],
                'prominence_factor': 0.02
            },
            'mid_field': {
                'range': (1.5, 3.0),
                'compounds': [
                    'Alanine', 'Glutamate',
                    'Citrate', 'Acetate',
                    'Succinate', '4-Aminobutyrate'
                ],
                'prominence_factor': 0.02
            },
            'sugar': {
                'range': (3.0, 5.5),
                'compounds': [
                    'Glucose', 'Sucrose',
                    'Maltose', 'Fructose',
                    'Xylose', 'Choline'
                ],
                'prominence_factor': 0.015
            },
            'anomeric': {
                'range': (4.5, 5.5),
                'compounds': [
                    'Glucose', 'Sucrose', 'Maltose'
                ],
                'prominence_factor': 0.025
            },
            'aromatic': {
                'range': (6.5, 9.0),
                'compounds': [
                    'Phenylalanine', 'Tyrosine',
                    'Tryptophan', 'Histidine',
                    'Formate', 'Nicotinate',
                    'Chlorogenate'
                ],
                'prominence_factor': 0.02
            }
        }

    def detect(self, spectrum, ppm_axis):
        """
        ตรวจจับ peaks จาก preprocessed spectrum

        Parameters:
            spectrum: np.array — preprocessed spectrum
            ppm_axis: np.array — chemical shift (ppm)

        Returns:
            dict: {
                'ppm': peak positions,
                'intensity': peak heights,
                'width': peak widths,
                'prominence': peak prominences,
                'region': ชื่อ region ของแต่ละ peak,
                'num_peaks': จำนวน peaks ทั้งหมด
            }
        """
        all_peaks = {
            'ppm': [],
            'intensity': [],
            'width': [],
            'prominence': [],
            'region': []
        }

        # ตรวจจับ peaks ตาม region
        for region_name, region_info in self.regions.items():
            ppm_low, ppm_high = region_info['range']
            prominence = region_info['prominence_factor']

            # หา index ของ region นี้
            mask = (ppm_axis >= ppm_low) & \
                   (ppm_axis <= ppm_high)
            region_spectrum = spectrum.copy()
            region_spectrum[~mask] = 0

            # ตรวจจับ peaks ใน region นี้
            peaks = self._detect_region(
                region_spectrum,
                ppm_axis,
                prominence_factor=prominence
            )

            if peaks and len(peaks['ppm']) > 0:
                all_peaks['ppm'].extend(peaks['ppm'])
                all_peaks['intensity'].extend(peaks['intensity'])
                all_peaks['width'].extend(peaks['width'])
                all_peaks['prominence'].extend(peaks['prominence'])
                all_peaks['region'].extend(
                    [region_name] * len(peaks['ppm'])
                )

        # แปลงเป็น numpy arrays
        if all_peaks['ppm']:
            sort_idx = np.argsort(all_peaks['ppm'])
            all_peaks = {
                'ppm': np.array(all_peaks['ppm'])[sort_idx],
                'intensity': np.array(
                    all_peaks['intensity']
                )[sort_idx],
                'width': np.array(
                    all_peaks['width']
                )[sort_idx],
                'prominence': np.array(
                    all_peaks['prominence']
                )[sort_idx],
                'region': np.array(
                    all_peaks['region']
                )[sort_idx],
                'num_peaks': len(all_peaks['ppm'])
            }
        else:
            all_peaks = {
                'ppm': np.array([]),
                'intensity': np.array([]),
                'width': np.array([]),
                'prominence': np.array([]),
                'region': np.array([]),
                'num_peaks': 0
            }

        self._print_summary(all_peaks)
        return all_peaks

    def _detect_region(self, spectrum, ppm_axis,
                       prominence_factor=0.02,
                       min_distance=3):
        """
        ตรวจจับ peaks ใน region หนึ่ง
        """
        max_intensity = np.max(spectrum)
        if max_intensity == 0:
            return None

        min_prominence = prominence_factor * max_intensity

        # Find peaks
        peak_indices, properties = find_peaks(
            spectrum,
            prominence=min_prominence,
            distance=min_distance,
            width=1
        )

        if len(peak_indices) == 0:
            return None

        # คำนวณ peak widths
        widths, _, _, _ = peak_widths(
            spectrum,
            peak_indices,
            rel_height=0.5
        )

        # แปลง index → ppm
        ppm_step = (ppm_axis[-1] - ppm_axis[0]) / len(ppm_axis)
        widths_ppm = widths * ppm_step

        return {
            'ppm': ppm_axis[peak_indices].tolist(),
            'intensity': spectrum[peak_indices].tolist(),
            'width': widths_ppm.tolist(),
            'prominence': properties['prominences'].tolist()
        }

    def _print_summary(self, peaks):
        """แสดงสรุป peaks ที่ตรวจพบ"""
        print(f"\n  Peak Detection Summary:")
        print(f"  Total peaks detected: {peaks['num_peaks']}")

        if peaks['num_peaks'] > 0:
            regions = {}
            for r in peaks['region']:
                regions[r] = regions.get(r, 0) + 1

            for region, count in regions.items():
                print(f"  • {region:12s}: {count} peaks")