"""
Synthetic NMR Spectrum Generator
สร้าง training data จาก reference library (Phase 1)
จำลอง Plant Extract A-N [1]
"""

import numpy as np
import json
from itertools import combinations


class SyntheticGenerator:
    """
    สร้าง synthetic NMR spectra สำหรับ train ML models
    จำลองสภาพจริงของ Plant Extract A-N [1]
    """

    def __init__(
        self,
        library_path='database/reference_library.json',
        spectrum_length=20000,
        ppm_range=(0.5, 10.0)
    ):
        """
        Parameters:
            library_path   : path ของ reference library (Phase 1)
            spectrum_length: ความยาว spectrum [1] (~20,000 points)
            ppm_range      : range ของ chemical shift
        """
        self.library = self._load_library(library_path)
        self.spectrum_length = spectrum_length
        self.ppm_range = ppm_range
        self.ppm_axis = np.linspace(
            ppm_range[0], ppm_range[1], spectrum_length
        )
        self.compound_names = list(self.library.keys())
        self.num_compounds = len(self.compound_names)

        print(f"Generator ready:")
        print(f"  Compounds : {self.num_compounds}")
        print(f"  Spectrum  : {spectrum_length:,} points")
        print(f"  PPM range : {ppm_range[0]} - {ppm_range[1]}")

    def _load_library(self, path):
        """โหลด reference library จาก Phase 1"""
        with open(path, 'r') as f:
            data = json.load(f)
        return data['compounds']

    # =========================================
    # สร้าง spectrum ของสาร 1 ชนิด
    # =========================================
    def _make_compound_spectrum(
        self, compound_name,
        intensity_scale=1.0,
        shift_noise=0.005,
        intensity_noise=0.15,
        width_variation=0.3
    ):
        """
        สร้าง spectrum ของสาร 1 ชนิด
        พร้อม augmentation ให้สมจริง

        Parameters:
            compound_name  : ชื่อสาร
            intensity_scale: ขนาด intensity (concentration)
            shift_noise    : noise ของ chemical shift (ppm)
            intensity_noise: noise ของ intensity
            width_variation: variation ของ peak width
        """
        spectrum = np.zeros(self.spectrum_length)
        ref_data = self.library.get(compound_name)

        if ref_data is None:
            return spectrum

        peaks = ref_data.get('peaks', [])

        for peak in peaks:
            ref_ppm = peak['ppm']
            ref_int = peak['intensity']

            # เพิ่ม random variation
            actual_ppm = ref_ppm + np.random.normal(
                0, shift_noise
            )
            actual_int = ref_int * intensity_scale * (
                1 + np.random.uniform(
                    -intensity_noise, intensity_noise
                )
            )
            base_width = np.random.uniform(0.008, 0.020)
            actual_width = base_width * (
                1 + np.random.uniform(
                    -width_variation, width_variation
                )
            )

            # Lorentzian peak shape (เหมือน NMR จริง)
            peak_signal = actual_int * (actual_width ** 2) / (
                (self.ppm_axis - actual_ppm) ** 2
                + actual_width ** 2
            )
            spectrum += peak_signal

        return spectrum

    # =========================================
    # สร้าง spectrum ที่มีหลายสาร
    # =========================================
    def generate_spectrum(self, compounds_present):
        """
        สร้าง synthetic spectrum
        ที่มีสารหลายชนิดรวมกัน (เหมือน Plant Extract [1])

        Parameters:
            compounds_present: list ของสารที่ต้องการ

        Returns:
            tuple: (ppm_axis, spectrum, label_vector)
        """
        spectrum = np.zeros(self.spectrum_length)

        for compound in compounds_present:
            if compound not in self.library:
                continue

            # สุ่ม concentration (intensity scale)
            intensity_scale = np.random.uniform(0.3, 1.5)

            compound_spectrum = self._make_compound_spectrum(
                compound,
                intensity_scale=intensity_scale
            )
            spectrum += compound_spectrum

        # เพิ่ม noise
        noise_level = np.max(spectrum) * np.random.uniform(
            0.005, 0.03
        )
        noise = np.random.normal(
            0, noise_level, self.spectrum_length
        )
        spectrum += noise

        # เพิ่ม baseline drift
        baseline_amp = np.max(spectrum) * np.random.uniform(
            0.01, 0.05
        )
        baseline = baseline_amp * np.sin(
            np.linspace(
                0,
                np.random.uniform(1, 4) * np.pi,
                self.spectrum_length
            )
        )
        spectrum += baseline
        spectrum = np.maximum(spectrum, 0)

        # สร้าง label vector (multi-label)
        labels = np.zeros(self.num_compounds)
        for compound in compounds_present:
            if compound in self.compound_names:
                idx = self.compound_names.index(compound)
                labels[idx] = 1

        return self.ppm_axis, spectrum, labels

    # =========================================
    # สร้าง Training Dataset
    # =========================================
    def build_training_set(self, n_samples=2000):
        """
        สร้าง training dataset ครบชุด
        จำลอง Plant Extract A-N [1]

        Parameters:
            n_samples: จำนวน synthetic spectra

        Returns:
            dict: {
                'X': spectra matrix (n_samples, spectrum_length),
                'y': label matrix (n_samples, num_compounds),
                'ppm_axis': ppm axis,
                'compound_names': list of compound names
            }
        """
        print(f"\nBuilding training set ({n_samples} samples)...")
        print("=" * 50)

        X = np.zeros((n_samples, self.spectrum_length))
        y = np.zeros((n_samples, self.num_compounds))

        for i in range(n_samples):
            # สุ่มจำนวนสาร (5-35 ชนิด) เหมือน Plant Extract [1]
            n_compounds = np.random.randint(
                5, min(35, self.num_compounds)
            )

            # สุ่มเลือกสาร
            selected = np.random.choice(
                self.compound_names,
                size=n_compounds,
                replace=False
            ).tolist()

            # สร้าง spectrum
            _, spectrum, labels = self.generate_spectrum(
                selected
            )

            X[i] = spectrum
            y[i] = labels

            if (i + 1) % 500 == 0:
                print(f"  Generated: {i+1}/{n_samples}")

        print(f"\n  Done!")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Compounds: {self.num_compounds}")
        print("=" * 50)

        return {
            'X': X,
            'y': y,
            'ppm_axis': self.ppm_axis,
            'compound_names': self.compound_names
        }