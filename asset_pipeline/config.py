import glob
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API keys (all optional individually -- only the ones actually used
#    need to be set; llm/client_factory.py raises a clear error if a
#    provider is requested without its key present) ────────────────────
ANTHROPIC_API_KEY: str | None = (os.getenv("ANTHROPIC_API_KEY") or "").strip() or None
OPENAI_API_KEY: str | None = (os.getenv("OPENAI_API_KEY") or "").strip() or None
GEMINI_API_KEY: str | None = (os.getenv("GEMINI_API_KEY") or "").strip() or None

# ── LLM defaults (config-level choice, never hardcoded in pipeline code) ─
DEFAULT_LLM_PROVIDER = os.getenv("ASSET_PIPELINE_LLM_PROVIDER", "anthropic")
DEFAULT_LLM_MODEL = os.getenv("ASSET_PIPELINE_LLM_MODEL", "claude-sonnet-4-6")

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
}

# ── Paths ─────────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).parent.resolve()

LIBRARY_DIR = Path(
    os.getenv("ASSET_PIPELINE_LIBRARY_DIR", str(PACKAGE_DIR / "library"))
).resolve()
CATALOG_PATH = LIBRARY_DIR / "catalog.json"
MATERIALS_DIR = LIBRARY_DIR / "materials"
TEXTURES_DIR = MATERIALS_DIR / "textures"
INTAKE_DIR = Path(
    os.getenv("ASSET_PIPELINE_INTAKE_DIR", str(PACKAGE_DIR / "intake"))
).resolve()

for _dir in (
    LIBRARY_DIR,
    LIBRARY_DIR / "generated",
    LIBRARY_DIR / "imported",
    LIBRARY_DIR / "composites",
    LIBRARY_DIR / "materials" / "textures",
    LIBRARY_DIR / "textures",
    LIBRARY_DIR / "previews",
    INTAKE_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ── External tool binaries ───────────────────────────────────────────
OPENSCAD_CANDIDATES = [
    r"C:\Program Files\OpenSCAD\openscad.exe",
    r"C:\Program Files (x86)\OpenSCAD\openscad.exe",
    "openscad",  # on PATH
]

BLENDER_CANDIDATES = [
    "blender",  # on PATH
]


def _find_desktop_blender() -> str | None:
    """Find any versioned desktop Blender install, newest version first.

    Version-agnostic on purpose: a hardcoded version list goes stale the
    moment a machine has a newer Blender than the list anticipated. Sorting
    the glob matches descending picks e.g. 'Blender 5.2' over 'Blender 4.2'
    without a code change.
    """
    patterns = [
        r"C:\Program Files\Blender Foundation\Blender *\blender.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Blender Foundation\Blender *\blender.exe"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/snap/bin/blender",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    if not matches:
        return None
    return sorted(matches, reverse=True)[0]


def _find_store_blender() -> str | None:
    """Match any MS Store Blender install regardless of version string.

    Mirrors viewer/arlem_preview_toolkit/arlem_preview_toolkit/prefab_to_glb.py
    exactly so both tools agree on where Blender lives.
    """
    pattern = r"C:\Program Files\WindowsApps\BlenderFoundation.Blender_*\Blender\blender.exe"
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _find_on_path_or_candidates(hint: str | None, candidates: list[str | None]) -> str | None:
    ordered = ([hint] if hint else []) + candidates
    for c in ordered:
        if not c:
            continue
        if Path(c).is_file():
            return str(Path(c))
        # Bare command name (no path separators) -- resolve via where/which
        if not Path(c).suffix and "\\" not in c and "/" not in c:
            try:
                result = subprocess.run(
                    ["where", c] if sys.platform == "win32" else ["which", c],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return result.stdout.strip().splitlines()[0]
            except Exception:
                pass
    return None


def find_openscad(hint: str | None = None) -> str | None:
    return _find_on_path_or_candidates(hint or os.getenv("ASSET_PIPELINE_OPENSCAD_BIN"), OPENSCAD_CANDIDATES)


def find_blender(hint: str | None = None) -> str | None:
    # Desktop install first, MS Store install last: the Store package's
    # blender.exe sits under C:\Program Files\WindowsApps with ACLs that
    # deny direct execution from a normal shell, so it only works as a
    # last resort (and only on machines where those ACLs are relaxed).
    return _find_on_path_or_candidates(
        hint or os.getenv("ASSET_PIPELINE_BLENDER_BIN"),
        [_find_desktop_blender()] + BLENDER_CANDIDATES + [_find_store_blender()],
    )


OPENSCAD_BIN = find_openscad()
BLENDER_BIN = find_blender()

# ── Mesh processing thresholds (Stage 4) ─────────────────────────────
MAX_TRIANGLE_COUNT = int(os.getenv("ASSET_PIPELINE_MAX_TRIANGLES", "20000"))
CATALOG_MATCH_CONFIDENCE_THRESHOLD = float(
    os.getenv("ASSET_PIPELINE_CATALOG_MATCH_THRESHOLD", "0.85")
)

# Procedural texture output resolution (Stage 5). Square, power of two.
TEXTURE_SIZE = int(os.getenv("ASSET_PIPELINE_TEXTURE_SIZE", "512"))

# Stage 6c: imported textures larger than this on either axis get a
# downscaled runtime derivative (the original is kept alongside -- planet
# maps run 8k-21k px and re-downloading costs a human).
TEXTURE_MAX_DIM = int(os.getenv("ASSET_PIPELINE_TEXTURE_MAX_DIM", "4096"))

# Fraction of original triangle count to keep when a mesh exceeds
# MAX_TRIANGLE_COUNT (Stage 4 mesh normalization). E.g. 0.5 = decimate to
# roughly half the original face count.
DECIMATE_RATIO = float(os.getenv("ASSET_PIPELINE_DECIMATE_RATIO", "0.5"))

BLENDER_TIMEOUT_SECONDS = int(os.getenv("ASSET_PIPELINE_BLENDER_TIMEOUT", "120"))
