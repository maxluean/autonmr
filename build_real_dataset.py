# build_real_dataset.py
# สร้าง training dataset จาก real NMR data

import numpy as np
import json
import pandas as pd
import os
from glob import glob


class RealDatasetBuilder:
    """
    สร้าง training dataset จาก real NMR data
    ที่ดึงมาจาก HMDB + MetaboLights
    """

    def __init__(
        self,
        hmdb_path='data/hmdb_spectra/hmdb_experimental.json',
        metabo_dir='data/real_nmr',
        spectrum_length=20000,
        ppm_range=(0.5, 10.0)
    ):
        self.spectrum_length = spectrum_length
        self.ppm_range = ppm_range
        self.ppm_axis = np.linspace(
            ppm_range[0], ppm_range[1], spectrum_length
        )

        # โหลด HMDB data
        self.hmdb_data = self._load_hmdb(hmdb_path)

        # โหลด MetaboLights data
        self.metabo_files = glob(
            os.path.join(metabo_dir, '*.csv')
        )

        # Compound names จาก Domain 1 [1]
        self.compound_names = list(self.hmdb_data.keys())

        print(f"Real data loaded:")
        print(f"  HMDB compounds: {len(self.hmdb_data)}")
        print(f"  MetaboLights files: {len(self.metabo_files)}")

    def _load_hmdb(self, path):
        """โหลด HMDB experimental data"""
        if not os.path.exists(path):
            print(f"HMDB data not found: {path}")
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def _peaks_to_spectrum(self, peaks, augment=True):
        """
        แปลง peak list → spectrum array
        ใช้ Lorentzian peak shape
        """
        spectrum = np.zeros(self.spectrum_length)

        for peak in peaks:
            ppm = peak['ppm']
            intensity = peak.get('intensity', 1.0)

            # ตรวจสอบ range
            if not (self.ppm_range[0] <= ppm
                    <= self.ppm_range[1]):
                continue

            # Augmentation
            if augment:
                ppm += np.random.normal(0, 0.005)
                intensity *= np.random.uniform(0.8, 1.2)

            # Peak width
            width = np.random.uniform(0.008, 0.020) \
                if augment else 0.012

            # Lorentzian
            peak_signal = intensity * (width ** 2) / (
                (self.ppm_axis - ppm) ** 2 + width ** 2
            )
            spectrum += peak_signal

        return spectrum

    def build_from_hmdb(self, n_per_compound=50):
        """
        สร้าง training data จาก HMDB experimental spectra
        แต่ละสารสร้าง n_per_compound ตัวอย่าง
        (ด้วย augmentation)
        """
        print(f"\nBuilding dataset from HMDB...")
        print(f"  {len(self.hmdb_data)} compounds × "
              f"{n_per_compound} augmentations")

        X_list = []
        y_list = []

        for compound, data in self.hmdb_data.items():
            if compound not in self.compound_names:
                continue

            comp_idx = self.compound_names.index(compound)
            peaks = data['peaks']

            for _ in range(n_per_compound):
                # สร้าง spectrum พร้อม augmentation
                spectrum = self._peaks_to_spectrum(
                    peaks, augment=True
                )

                # เพิ่ม noise
                noise_level = (
                    np.max(spectrum) *
                    np.random.uniform(0.01, 0.05)
                )
                noise = np.random.normal(
                    0, noise_level, self.spectrum_length
                )
                spectrum = np.maximum(
                    spectrum + noise, 0
                )

                # Label vector
                label = np.zeros(len(self.compound_names))
                label[comp_idx] = 1

                X_list.append(spectrum)
                y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"  Dataset shape: X={X.shape}, y={y.shape}")
        return X, y

    def build_mixture_dataset(self, n_samples=2000):
        """
        สร้าง mixture dataset
        จำลอง Plant Extract A-N [1]
        โดยใช้ real peaks จาก HMDB
        """
        print(f"\nBuilding mixture dataset...")
        print(f"  {n_samples} Plant Extract mixtures [1]")

        X_list = []
        y_list = []

        for i in range(n_samples):
            # สุ่มสาร 5-35 ชนิด เหมือน Plant Extract [1]
            n_comp = np.random.randint(
                5, min(35, len(self.compound_names))
            )
            selected = np.random.choice(
                self.compound_names,
                size=n_comp,
                replace=False
            )

            # รวม spectrum ของสารที่เลือก
            mixture = np.zeros(self.spectrum_length)
            label = np.zeros(len(self.compound_names))

            for compound in selected:
                if compound not in self.hmdb_data:
                    continue

                peaks = self.hmdb_data[compound]['peaks']
                scale = np.random.uniform(0.3, 1.5)

                comp_spectrum = self._peaks_to_spectrum(
                    peaks, augment=True
                )
                mixture += comp_spectrum * scale

                # Set label
                idx = self.compound_names.index(compound)
                label[idx] = 1

            # Noise + baseline
            noise = np.random.normal(
                0,
                np.max(mixture) * 0.02,
                self.spectrum_length
            )
            baseline = (
                np.max(mixture) * 0.01 *
                np.sin(np.linspace(
                    0, 3 * np.pi, self.spectrum_length
                ))
            )
            mixture = np.maximum(
                mixture + noise + baseline, 0
            )

            X_list.append(mixture)
            y_list.append(label)

            if (i + 1) % 500 == 0:
                print(f"  Generated: {i+1}/{n_samples}")

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"  Done! Shape: X={X.shape}, y={y.shape}")
        return X, y

    def save_dataset(self, X, y, name='real_dataset'):
        """บันทึก dataset"""
        os.makedirs('data/datasets', exist_ok=True)
        np.save(f'data/datasets/{name}_X.npy', X)
        np.save(f'data/datasets/{name}_y.npy', y)
        np.save(
            f'data/datasets/{name}_compounds.npy',
            np.array(self.compound_names)
        )
        print(f"Saved: data/datasets/{name}_*.npy")


if __name__ == "__main__":
    builder = RealDatasetBuilder()

    # สร้าง mixture dataset จาก real HMDB data
    X, y = builder.build_mixture_dataset(n_samples=2000)
    builder.save_dataset(X, y, 'real_mixture')