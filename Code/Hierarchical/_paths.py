"""sys.path setup shared by every module in Code/Hierarchical/.

The pipeline reuses the benchmark layer (client factory, retry guards, tracking,
prompt context) and the schemas, which live in sibling folders that are not
packages. Importing this module first makes them importable, the same way
Code/Analysis/benchmark_dataframe.py reaches into Code/Benchmark/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

for _p in (
    PROJECT_ROOT,                         # tracking/
    PROJECT_ROOT / "Code" / "Schemas",
    PROJECT_ROOT / "Code" / "Benchmark",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
