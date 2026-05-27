# download_metabo.py
# ดึง NMR datasets ที่เกี่ยวกับ Plant Extract [1]

import requests
import os
import json

def search_plant_nmr_studies():
    """
    ค้นหา studies ที่เกี่ยวกับ Plant Extract + NMR
    จาก MetaboLights API
    """
    # MetaboLights API endpoint
    search_url = (
        "https://www.ebi.ac.uk/metabolights/ws/studies"
        "/search?query=plant+extract+NMR"
        "&technology=NMR"
    )

    response = requests.get(search_url, timeout=30)

    if response.status_code == 200:
        studies = response.json()
        print(f"Found {len(studies)} studies")
        return studies
    return []


def download_study(study_id, output_dir='data/real_nmr'):
    """
    Download NMR data จาก study ที่เลือก
    """
    os.makedirs(output_dir, exist_ok=True)

    # ดึง study files
    files_url = (
        f"https://www.ebi.ac.uk/metabolights"
        f"/ws/studies/{study_id}/files"
    )
    response = requests.get(files_url, timeout=30)

    if response.status_code != 200:
        print(f"Cannot access: {study_id}")
        return []

    files = response.json()
    downloaded = []

    for file_info in files:
        filename = file_info.get('file', '')

        # ดึงเฉพาะไฟล์ NMR data
        if any(ext in filename.lower()
               for ext in ['.csv', '.txt', '.nmrml']):

            file_url = (
                f"https://www.ebi.ac.uk/metabolights"
                f"/ws/studies/{study_id}"
                f"/download?file={filename}"
            )

            save_path = os.path.join(
                output_dir, f"{study_id}_{filename}"
            )

            r = requests.get(file_url, timeout=60)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
                downloaded.append(save_path)
                print(f"Downloaded: {filename}")

    return downloaded


# Studies ที่แนะนำสำหรับ Plant Extract NMR
RECOMMENDED_STUDIES = [
    "MTBLS1",    # Plant metabolomics NMR
    "MTBLS17",   # Plant extract NMR
    "MTBLS136",  # Metabolite profiling plants
    "MTBLS267",  # Plant NMR metabolomics
]


if __name__ == "__main__":
    for study_id in RECOMMENDED_STUDIES:
        print(f"\nDownloading: {study_id}")
        files = download_study(study_id)
        print(f"Downloaded {len(files)} files")