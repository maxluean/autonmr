"""
NMR Preprocessor Module
ทำความสะอาด raw NMR spectrum ก่อน annotation

4 ขั้นตอน [1]:
1. Noise Reduction      → Savitzky-Golay Filter
2. Baseline Correction  → Asymmetric Least Squares (ALS)
3. Normalization        → Probabilistic Quotient (PQN)
4. Peak Alignment       → Correlation Optimized Warping (COW)
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class NMRPreprocessor:
    """
    Preprocessor สำหรับ NMR spectrum
    Input : Raw spectrum (20,000+ features) [1]
    Output: Cleaned spectrum พร้อม annotation
    """

    def __init__(self, sample_type='plant_extract'):
        """
        Parameters:
            sample_type: ประเภทตัวอย่าง
                        'plant_extract' สำหรับ Subject 1-N [1]
        """
        self.sample_type = sample_type

        # Parameters สำหรับ Plant Extract [1]
        self.params = {
            'plant_extract': {
                'noise': {
                    'window_length': 11,
                    'polyorder': 3
                },
                'baseline': {
                    'lam': 1e6,
                    'p': 0.01,
                    'max_iter': 10
                },
                'normalization': {
                    'method': 'pqn'
                },
                'alignment': {
                    'reference_idx': 0,
                    'max_shift': 50
                }
            }
        }

    # =========================================
    # Step 1: Noise Reduction
    # =========================================
    def denoise(self, spectrum):
        """
        ลด noise จาก raw spectrum
        ใช้ Savitzky-Golay Filter

        Parameters:
            spectrum: np.array (20,000+ points) [1]

        Returns:
            np.array: Denoised spectrum
        """
        p = self.params[self.sample_type]['noise']

        # ตรวจสอบว่า window_length ต้องเป็นเลขคี่
        window = p['window_length']
        if window % 2 == 0:
            window += 1

        denoised = savgol_filter(
            spectrum,
            window_length=window,
            polyorder=p['polyorder']
        )

        # ป้องกัน negative values
        denoised = np.maximum(denoised, 0)

        return denoised

    # =========================================
    # Step 2: Baseline Correction
    # =========================================
    def correct_baseline(self, spectrum):
        """
        แก้ไข baseline drift
        ใช้ Asymmetric Least Squares (ALS)

        Parameters:
            spectrum: np.array (denoised)

        Returns:
            np.array: Baseline-corrected spectrum
        """
        p = self.params[self.sample_type]['baseline']
        baseline = self._als_baseline(
            spectrum,
            lam=p['lam'],
            p=p['p'],
            max_iter=p['max_iter']
        )
        corrected = spectrum - baseline

        # ป้องกัน negative values
        corrected = np.maximum(corrected, 0)

        return corrected

    def _als_baseline(self, spectrum, lam, p, max_iter):
        """
        Asymmetric Least Squares algorithm
        """
        n = len(spectrum)
        # สร้าง difference matrix
        D = diags(
            [1, -2, 1],
            [0, 1, 2],
            shape=(n - 2, n)
        )
        w = np.ones(n)

        for _ in range(max_iter):
            W = diags(w, 0)
            Z = W + lam * D.T @ D
            baseline = spsolve(Z, w * spectrum)
            w = p * (spectrum > baseline) + \
                (1 - p) * (spectrum <= baseline)

        return baseline

    # =========================================
    # Step 3: Normalization
    # =========================================
    def normalize(self, spectrum):
        """
        Normalize spectrum
        ใช้ Probabilistic Quotient Normalization (PQN)
        เพื่อลด variation ระหว่าง Subject 1-N [1]

        Parameters:
            spectrum: np.array (baseline-corrected)

        Returns:
            np.array: Normalized spectrum
        """
        method = self.params[self.sample_type]\
                            ['normalization']['method']

        if method == 'pqn':
            return self._pqn_normalize(spectrum)
        else:
            return self._minmax_normalize(spectrum)

    def _pqn_normalize(self, spectrum):
        """
        Probabilistic Quotient Normalization
        """
        # Reference = median ของ spectrum
        reference = np.median(
            spectrum[spectrum > 0]
        ) if np.any(spectrum > 0) else 1.0

        if reference == 0:
            return spectrum

        # หาร spectrum ด้วย reference
        quotients = spectrum / reference

        # Median quotient = normalization factor
        nonzero_q = quotients[quotients > 0]
        if len(nonzero_q) == 0:
            return spectrum

        pqn_factor = np.median(nonzero_q)

        if pqn_factor == 0:
            return spectrum

        normalized = spectrum / pqn_factor

        return normalized

    def _minmax_normalize(self, spectrum):
        """
        Min-Max Normalization (fallback)
        """
        s_min = np.min(spectrum)
        s_max = np.max(spectrum)

        if s_max == s_min:
            return spectrum

        return (spectrum - s_min) / (s_max - s_min)

    # =========================================
    # Step 4: Peak Alignment
    # =========================================
    def align(self, spectra_list, reference_idx=0):
        """
        จัดแนว peaks ระหว่าง Subject 1-N [1]
        ใช้ Cross-correlation alignment

        Parameters:
            spectra_list: list ของ spectra ทุก Subject [1]
            reference_idx: index ของ reference spectrum

        Returns:
            list: Aligned spectra
        """
        if len(spectra_list) <= 1:
            return spectra_list

        reference = spectra_list[reference_idx]
        aligned = []

        for i, spectrum in enumerate(spectra_list):
            if i == reference_idx:
                aligned.append(spectrum)
            else:
                shifted = self._align_single(
                    spectrum,
                    reference
                )
                aligned.append(shifted)

        return aligned

    def _align_single(self, spectrum, reference):
        """
        Align spectrum หนึ่งตัวกับ reference
        ใช้ Cross-correlation
        """
        max_shift = self.params[self.sample_type]\
                               ['alignment']['max_shift']

        # Cross-correlation
        correlation = np.correlate(
            spectrum - np.mean(spectrum),
            reference - np.mean(reference),
            mode='full'
        )

        # หา optimal shift
        center = len(correlation) // 2
        search_range = correlation[
            center - max_shift:center + max_shift + 1
        ]
        optimal_shift = np.argmax(search_range) - max_shift

        # Shift spectrum
        if optimal_shift > 0:
            aligned = np.zeros_like(spectrum)
            aligned[optimal_shift:] = spectrum[:-optimal_shift]
        elif optimal_shift < 0:
            aligned = np.zeros_like(spectrum)
            aligned[:optimal_shift] = spectrum[-optimal_shift:]
        else:
            aligned = spectrum.copy()

        return aligned

    # =========================================
    # Full Pipeline
    # =========================================
    def process(self, spectrum, ppm_axis=None):
        """
        รัน preprocessing pipeline ครบ 4 ขั้นตอน

        Parameters:
            spectrum: np.array — raw NMR spectrum [1]
            ppm_axis: np.array — chemical shift axis

        Returns:
            dict: {
                'processed': processed spectrum,
                'ppm_axis': ppm axis,
                'steps': intermediate results
            }
        """
        steps = {}

        # Step 1: Denoise
        step1 = self.denoise(spectrum)
        steps['denoised'] = step1

        # Step 2: Baseline Correction
        step2 = self.correct_baseline(step1)
        steps['baseline_corrected'] = step2

        # Step 3: Normalize
        step3 = self.normalize(step2)
        steps['normalized'] = step3

        # สร้าง ppm_axis ถ้าไม่มี
        if ppm_axis is None:
            ppm_axis = np.linspace(0.5, 10.0, len(spectrum))

        return {
            'processed': step3,
            'ppm_axis': ppm_axis,
            'steps': steps,
            'sample_type': self.sample_type
        }

    def process_batch(self, spectra_dict):
        """
        รัน preprocessing สำหรับหลาย Subject พร้อมกัน [1]
        แทนที่การทำ manual ทีละภาพ [1]

        Parameters:
            spectra_dict: {
                'Subject_1': (spectrum, ppm_axis),
                'Subject_2': (spectrum, ppm_axis),
                ...
            }

        Returns:
            dict: Processed spectra ทุก Subject [1]
        """
        print(f"\nProcessing {len(spectra_dict)} subjects...")
        print("=" * 50)

        results = {}
        spectra_list = []
        subject_names = []

        # Process แต่ละ Subject
        for subject, (spectrum, ppm) in spectra_dict.items():
            print(f"  Processing: {subject}")
            result = self.process(spectrum, ppm)
            results[subject] = result
            spectra_list.append(result['processed'])
            subject_names.append(subject)

        # Align ทุก Subject เข้าหากัน
        if len(spectra_list) > 1:
            print("\n  Aligning spectra across subjects...")
            aligned = self.align(spectra_list)
            for i, subject in enumerate(subject_names):
                results[subject]['processed'] = aligned[i]
                results[subject]['aligned'] = True

        print(f"\n  Done! Processed {len(results)} subjects")
        print("=" * 50)

        return results