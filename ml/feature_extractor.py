"""
Feature Extractor Module
แปลง NMR spectrum → feature vector
สำหรับ Random Forest model
"""

import numpy as np


class FeatureExtractor:
    """
    สร้าง feature vector จาก NMR spectrum
    สำหรับ train Random Forest [1]
    """

    def __init__(self, ppm_axis=None):
        self.ppm_axis = ppm_axis

        # NMR regions สำหรับ Plant Extract [1]
        self.regions = [
            (0.5,  1.5,  'aliphatic'),
            (1.5,  3.0,  'mid_field'),
            (3.0,  4.5,  'sugar_low'),
            (4.5,  5.5,  'sugar_high'),
            (5.5,  7.0,  'olefinic'),
            (7.0,  8.5,  'aromatic'),
            (8.5, 10.0,  'downfield'),
        ]

    def extract(self, spectrum, ppm_axis=None):
        """
        สร้าง feature vector จาก spectrum 1 ตัว

        Returns:
            np.array: feature vector
        """
        if ppm_axis is None:
            ppm_axis = self.ppm_axis
        if ppm_axis is None:
            ppm_axis = np.linspace(
                0.5, 10.0, len(spectrum)
            )

        features = []

        # 1. Global statistics
        features.extend(self._global_stats(spectrum))

        # 2. Regional statistics
        features.extend(
            self._regional_stats(spectrum, ppm_axis)
        )

        # 3. Peak features
        features.extend(self._peak_features(spectrum))

        # 4. Intensity ratios
        features.extend(
            self._intensity_ratios(spectrum, ppm_axis)
        )

        return np.array(features)

    def _global_stats(self, spectrum):
        """Global statistics ของ spectrum"""
        s = spectrum
        return [
            np.mean(s),
            np.std(s),
            np.max(s),
            np.min(s),
            np.median(s),
            np.sum(s > np.mean(s) + np.std(s)),  # peaks above mean
            np.percentile(s, 75),
            np.percentile(s, 90),
            np.percentile(s, 95),
        ]

    def _regional_stats(self, spectrum, ppm_axis):
        """Statistics แยกตาม NMR region"""
        features = []

        for ppm_low, ppm_high, _ in self.regions:
            mask = (
                (ppm_axis >= ppm_low) &
                (ppm_axis <= ppm_high)
            )
            region = spectrum[mask]

            if len(region) > 0:
                features.extend([
                    np.max(region),
                    np.mean(region),
                    np.sum(region),
                    np.std(region),
                    np.sum(region > np.mean(region)),
                ])
            else:
                features.extend([0, 0, 0, 0, 0])

        return features

    def _peak_features(self, spectrum):
        """Features จาก peaks"""
        from scipy.signal import find_peaks

        # หา peaks
        min_prom = np.max(spectrum) * 0.02
        peaks, props = find_peaks(
            spectrum,
            prominence=min_prom,
            distance=3
        )

        if len(peaks) == 0:
            return [0] * 8

        intensities = spectrum[peaks]
        prominences = props['prominences']

        return [
            len(peaks),
            np.mean(intensities),
            np.max(intensities),
            np.std(intensities),
            np.mean(prominences),
            np.max(prominences),
            np.sum(intensities > np.median(intensities)),
            len(peaks) / len(spectrum),
        ]

    def _intensity_ratios(self, spectrum, ppm_axis):
        """
        Intensity ratios ระหว่าง regions
        สำคัญสำหรับแยก Sugar vs Amino Acid [1]
        """
        features = []
        region_sums = []

        for ppm_low, ppm_high, _ in self.regions:
            mask = (
                (ppm_axis >= ppm_low) &
                (ppm_axis <= ppm_high)
            )
            region_sums.append(np.sum(spectrum[mask]))

        total = sum(region_sums) if sum(region_sums) > 0 else 1

        # Ratio ของแต่ละ region ต่อ total
        for s in region_sums:
            features.append(s / total)

        # Cross-region ratios ที่สำคัญ
        # Sugar / Aliphatic [1]
        sugar = region_sums[2] + region_sums[3]
        aliphatic = region_sums[0]
        features.append(
            sugar / aliphatic if aliphatic > 0 else 0
        )

        # Aromatic / Total [1]
        aromatic = region_sums[5]
        features.append(aromatic / total)

        return features

    def extract_batch(self, X, ppm_axis=None):
        """
        สร้าง feature vectors จาก spectra หลายตัว

        Parameters:
            X: np.array (n_samples, spectrum_length)

        Returns:
            np.array: (n_samples, n_features)
        """
        print(f"\nExtracting features from {len(X)} spectra...")

        features = []
        for i, spectrum in enumerate(X):
            feat = self.extract(spectrum, ppm_axis)
            features.append(feat)

            if (i + 1) % 500 == 0:
                print(f"  Extracted: {i+1}/{len(X)}")

        X_feat = np.array(features)
        print(f"  Feature shape: {X_feat.shape}")
        return X_feat