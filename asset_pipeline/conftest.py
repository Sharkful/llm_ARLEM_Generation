import sys
from pathlib import Path

# Add asset_pipeline/ to sys.path so all modules import without a package prefix.
sys.path.insert(0, str(Path(__file__).parent))
