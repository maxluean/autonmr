"""
Database Module
สร้าง Reference Library จาก Public NMR Databases
สำหรับ annotation สาร 40+ ชนิดที่พบใน Plant Extract A-N [1]

Databases:
- HMDB (Human Metabolome Database) : 220,000+ metabolites
- BMRB (Biological Magnetic Resonance Bank) : 15,000+ entries
- NP-MRD (Natural Products MRD) : 50,000+ natural products
"""

from database.hmdb_data import (
    HMDB_NMR_DATA,
    HMDB_COMPOUND_IDS,
    get_compound_data,
    get_all_compounds,
    get_compounds_by_region
)
from database.bmrb_data import (
    BMRB_NMR_DATA,
    get_bmrb_shifts,
    get_all_bmrb_compounds
)
from database.npmrd_data import (
    NPMRD_NMR_DATA,
    get_npmrd_shifts,
    get_all_npmrd_compounds
)
from database.library_builder import LibraryBuilder