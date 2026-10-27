"""
Lab resources zip: naming, layout, zipping and validation (#92).

The headset downloads ``<lab id>.zip``, unpacks it into ``lab_resources/`` and
opens ``lab_resources/<lab id>/``, taking the first ``.json`` it finds there. So
the zip's single top-level folder must have the zip's own name:

    <Name>Lab.zip
      <Name>Lab/
        Audios/<audioClip>.wav     24 kHz, 16-bit PCM, mono; one per narrated clip
        Textures/
        Videos/
        <Lab_Name>.json            the lab, json_lab v2.0 format

See NARRATION.md ("Resources zip") for the rules behind each check.

    python "Code/Narration/bundle.py" names    <lab.json> [--bundle-name NAME]
    python "Code/Narration/bundle.py" zip      <bundle folder> [--out ZIP]
    python "Code/Narration/bundle.py" validate <zip or bundle folder> [...]

``validate`` exits with code 1 if any bundle has errors.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import wave
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHEMAS = PROJECT_ROOT / "Code" / "Schemas"

AUDIO_DIR = "Audios"
TEXTURE_DIR = "Textures"
VIDEO_DIR = "Videos"
SUBDIRS = (AUDIO_DIR, TEXTURE_DIR, VIDEO_DIR)

# The one audio format the bundle carries. Both cloud providers return this
# natively, so nothing is resampled or lossy-encoded.
WAV_SAMPLE_RATE = 24_000
WAV_SAMPLE_WIDTH = 2  # bytes, i.e. 16-bit
WAV_CHANNELS = 1

SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
# A fixed timestamp keeps zips byte-identical when their contents are.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


# ── Names ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BundleNames:
    folder: str      # zip stem and its top-level folder, e.g. "DnaStructureReplicationLab"
    json_file: str   # e.g. "Dna_Structure_Replication.json"

    @property
    def zip_file(self) -> str:
        return f"{self.folder}.zip"


def _name_words(text: str) -> list[str]:
    """Words of a labId, split on any non-alphanumeric run, minus standalone
    'lab' words (so a labId like 'vsepr_theory_lab' doesn't become 'LabLab')."""
    words = [w for w in re.split(r"[^A-Za-z0-9]+", text or "") if w]
    return [w for w in words if w.lower() != "lab"]


def _cap(word: str) -> str:
    # Capitalize the first letter only, so acronyms like 'GP1' or 'AR' survive.
    return word[:1].upper() + word[1:]


def bundle_names(lab_id: str, override: str | None = None) -> BundleNames:
    """Zip/folder and JSON names for a lab.

    Derived from ``labId``: 'dna_structure_replication' gives
    ``DnaStructureReplicationLab`` and ``Dna_Structure_Replication.json``.
    ``override`` replaces the folder (and zip) name, e.g. to match the id a lab
    has on the server; the JSON name still comes from ``labId`` when it can,
    since the headset opens whichever ``.json`` it finds.
    """
    words = _name_words(lab_id)
    if override is not None:
        if not SAFE_NAME.match(override):
            raise ValueError(
                f"bundle name {override!r} must use only letters, digits, '_' and '-'"
            )
        folder = override
    elif words:
        folder = "".join(_cap(w) for w in words) + "Lab"
    else:
        raise ValueError(
            f"labId {lab_id!r} has no usable words for a bundle name; pass a bundle name"
        )
    if not words:
        words = _name_words(re.sub(r"Lab$", "", folder)) or [folder]
    return BundleNames(folder=folder, json_file="_".join(_cap(w) for w in words) + ".json")


# ── Lab references ────────────────────────────────────────────────────


def _clips(lab):
    """Every clip dict in the lab, in module and clip order. Wrong shapes are
    skipped here and reported by ``audio_clip_problems``."""
    modules = lab.get("modules") if isinstance(lab, dict) else None
    if not isinstance(modules, list):
        return
    for module in modules:
        clips = module.get("clips") if isinstance(module, dict) else None
        if isinstance(clips, list):
            yield from (c for c in clips if isinstance(c, dict))


def audio_clip_refs(lab: dict) -> list[str]:
    """Every ``Clip.audioClip`` in the lab, in module and clip order, exactly as
    written: the headset looks up that raw string, so names are not trimmed.
    ``None`` and ``""`` mean no audio."""
    return [
        clip["audioClip"] for clip in _clips(lab)
        if isinstance(clip.get("audioClip"), str) and clip["audioClip"] != ""
    ]


def audio_clip_problems(lab) -> list[str]:
    """Shape problems that stop clips' audio from being checked, one message each."""
    if not isinstance(lab, dict):
        return [f"the lab JSON is a {type(lab).__name__}, not an object"]
    modules = lab.get("modules")
    if not isinstance(modules, list):
        return [f"'modules' is a {type(modules).__name__}, not a list"]
    problems = []
    for m, module in enumerate(modules, 1):
        if not isinstance(module, dict):
            problems.append(f"module {m} is a {type(module).__name__}, not an object")
            continue
        clips = module.get("clips")
        if not isinstance(clips, list):
            problems.append(f"module {m} 'clips' is a {type(clips).__name__}, not a list")
            continue
        for c, clip in enumerate(clips, 1):
            if not isinstance(clip, dict):
                problems.append(f"module {m} clip {c} is a {type(clip).__name__}, not an object")
            elif clip.get("audioClip") is not None and not isinstance(clip["audioClip"], str):
                problems.append(f"module {m} clip {c} audioClip is a "
                                f"{type(clip['audioClip']).__name__}, not a string")
    return problems


def sound_effect_refs(lab: dict) -> list[str]:
    """``CheckAngleComponent.audioClipSuccess`` values: sound effects, not narration."""
    found: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            value = obj.get("audioClipSuccess")
            if isinstance(value, str) and value != "":
                found.append(value)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(lab)
    return found


# ── Building ──────────────────────────────────────────────────────────


def create_layout(parent: Path, names: BundleNames) -> Path:
    """Create ``<parent>/<folder>/{Audios,Textures,Videos}`` and return the folder."""
    folder = Path(parent) / names.folder
    for sub in SUBDIRS:
        (folder / sub).mkdir(parents=True, exist_ok=True)
    return folder


def write_wav(path: Path, pcm: bytes) -> None:
    """Write 16-bit mono 24 kHz PCM samples as a WAV file."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(WAV_CHANNELS)
        w.setsampwidth(WAV_SAMPLE_WIDTH)
        w.setframerate(WAV_SAMPLE_RATE)
        w.writeframes(pcm)


def zip_bundle(folder: Path, zip_path: Path | None = None) -> Path:
    """Zip a bundle folder as ``<folder name>.zip``, its contents under one
    top-level folder of the same name.

    Entries are sorted and carry a fixed timestamp, so the same contents always
    give the same bytes. Every directory gets its own entry, so empty
    ``Textures/`` and ``Videos/`` survive. The zip is written to a temp file and
    moved into place, so a crash never leaves a half-written zip. The finished
    zip is readable by everyone (0644), so a web server can serve it.
    """
    # Resolve so '.' or '..' still give the folder's real name.
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"bundle folder not found: {folder}")
    zip_path = Path(zip_path).resolve() if zip_path else folder.parent / f"{folder.name}.zip"
    if zip_path.stem != folder.name:
        raise ValueError(
            f"zip name {zip_path.name!r} must match its folder {folder.name!r}; "
            "the headset opens the folder named after the zip"
        )

    paths = sorted(folder.rglob("*"), key=lambda p: p.relative_to(folder).as_posix())
    fd, tmp = tempfile.mkstemp(dir=zip_path.parent, suffix=".zip.tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_dir_info(folder.name + "/"), b"")
            for p in paths:
                arc = f"{folder.name}/{p.relative_to(folder).as_posix()}"
                if p.is_dir():
                    zf.writestr(_dir_info(arc + "/"), b"")
                else:
                    info = zipfile.ZipInfo(arc, date_time=_ZIP_DATE)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    zf.writestr(info, p.read_bytes())
        os.chmod(tmp, 0o644)  # mkstemp creates the file owner-only (0600)
        os.replace(tmp, zip_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return zip_path


def _dir_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
    info.external_attr = (0o40755 << 16) | 0x10  # directory, for unzip tools and Windows
    return info


# ── Validation ────────────────────────────────────────────────────────


@dataclass
class Report:
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _listing(path: Path) -> tuple[set[str], set[str], dict]:
    """(dirs, files, reader) for a zip or a folder, as '/'-separated paths
    relative to where the zip would be unpacked. ``reader[name]()`` gives bytes."""
    dirs: set[str] = set()
    files: set[str] = set()
    reader: dict = {}
    if path.is_dir():
        base = path.parent
        dirs.add(path.name)
        for p in path.rglob("*"):
            rel = p.relative_to(base).as_posix()
            if p.is_dir():
                dirs.add(rel)
            else:
                files.add(rel)
                reader[rel] = p.read_bytes
    else:
        # Bundles are a few MB, so read entries up front and close the zip.
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                name = info.filename.rstrip("/")
                if not name:
                    continue
                if info.is_dir():
                    dirs.add(name)
                else:
                    files.add(name)
                    reader[name] = (lambda data=zf.read(info.filename): data)
    # Parent folders are implied by the paths under them.
    for name in list(files) + list(dirs):
        parts = name.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return dirs, files, reader


def _check_wav(data: bytes) -> str | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype())
            frames = w.getnframes()
    except (wave.Error, EOFError) as exc:
        return f"not a readable PCM WAV ({exc})"
    want = (WAV_CHANNELS, WAV_SAMPLE_WIDTH, WAV_SAMPLE_RATE, "NONE")
    if fmt != want:
        ch, width, rate, comp = fmt
        return (f"is {rate} Hz, {8 * width}-bit, {ch} channel(s), compression {comp}; "
                f"expected {WAV_SAMPLE_RATE} Hz, 16-bit, mono PCM")
    if frames == 0:
        return "has no audio"
    return None


def validate_bundle(path: Path | str) -> Report:
    """Check a resources zip (or the folder it is built from) against the layout.

    Errors: a damaged zip, a wrong top-level folder, missing subfolders, not
    exactly one lab JSON, a JSON that isn't a valid json_lab ``Lab`` (or whose
    modules/clips have the wrong shape), stray files, files in Audios/ that
    aren't flat lowercase ``.wav``, a WAV in the wrong format, and a clip
    ``audioClip`` (exactly as written) with no matching ``Audios/<name>.wav``.
    Warnings: audio files no clip refers to, and sound effects
    (``audioClipSuccess``) with no file in Audios/.
    """
    report = Report(str(path))
    # Resolve so '.' or '..' still give the folder's real name.
    path = Path(path).resolve()
    err, warn = report.errors.append, report.warnings.append

    if not path.exists():
        err("path does not exist")
        return report
    if path.is_file() and not zipfile.is_zipfile(path):
        err("not a zip file")
        return report

    try:
        dirs, files, reader = _listing(path)
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as exc:
        what = "zip is damaged" if path.is_file() else "cannot read the bundle folder"
        err(f"{what} ({exc or type(exc).__name__})")
        return report
    tops = {n.split("/")[0] for n in dirs | files}
    if len(tops) != 1:
        err(f"needs exactly one top-level folder, found {sorted(tops) or 'none'}")
        return report
    root = tops.pop()
    if root in files:
        err(f"top-level entry {root!r} is a file, not a folder")
        return report
    if path.is_file() and root != path.stem:
        err(f"top-level folder {root!r} must match the zip name {path.stem!r}")

    for sub in SUBDIRS:
        if f"{root}/{sub}" not in dirs:
            err(f"missing folder {root}/{sub}/")

    # Directly inside the root: the one lab JSON and nothing else.
    top_files = sorted(f for f in files if f.count("/") == 1)
    jsons = [f for f in top_files if f.lower().endswith(".json")]
    for f in top_files:
        if f not in jsons:
            err(f"unexpected file {f} (only the lab JSON belongs at the top)")
    extra_dirs = sorted(d for d in dirs if d.count("/") == 1 and d.split("/")[1] not in SUBDIRS)
    for d in extra_dirs:
        err(f"unexpected folder {d}/")
    if len(jsons) != 1:
        err(f"needs exactly one lab JSON in {root}/, found {len(jsons)}")

    # Audios/: flat, .wav only, the right format.
    audio_prefix = f"{root}/{AUDIO_DIR}/"
    wav_names: set[str] = set()
    for f in sorted(files):
        if not f.startswith(audio_prefix):
            continue
        rel = f[len(audio_prefix):]
        stem, dot, ext = rel.rpartition(".")
        if "/" in rel:
            err(f"{f}: Audios/ must be flat (no subfolders)")
        elif ext != "wav" or not dot:
            # Exactly '.wav': the headset's filesystem is case-sensitive.
            err(f"{f}: Audios/ holds only .wav files (lowercase extension)")
        elif not SAFE_NAME.match(stem):
            err(f"{f}: file name must use only letters, digits, '_' and '-'")
        else:
            wav_names.add(stem)
            problem = _check_wav(reader[f]())
            if problem:
                err(f"{f} {problem}")
    for d in sorted(dirs):
        if d.startswith(audio_prefix):
            err(f"{d}/: Audios/ must be flat (no subfolders)")

    if len(jsons) != 1:
        return report
    try:
        lab = json.loads(reader[jsons[0]]())
    except (ValueError, UnicodeDecodeError) as exc:
        err(f"{jsons[0]} is not valid JSON ({exc})")
        return report
    try:
        if str(_SCHEMAS) not in sys.path:
            sys.path.insert(0, str(_SCHEMAS))
        from json_lab import Lab
        Lab.model_validate(lab)
    except ImportError as exc:
        err(f"cannot load the json_lab schema to check {jsons[0]} ({exc})")
    except Exception as exc:  # pydantic.ValidationError, kept generic for one line
        first = str(exc).splitlines()
        err(f"{jsons[0]} is not a valid json_lab Lab: " + " ".join(first[:3]))

    for problem in audio_clip_problems(lab):
        err(f"{jsons[0]}: {problem}")
    refs = audio_clip_refs(lab)
    for name in dict.fromkeys(refs):
        if not SAFE_NAME.match(name):
            err(f"clip audioClip {name!r} is not a safe file name "
                "(letters, digits, '_' and '-' only; no spaces or extension)")
        elif name not in wav_names:
            err(f"clip audioClip {name!r} has no {AUDIO_DIR}/{name}.wav")
    effects = sound_effect_refs(lab)
    for name in sorted(wav_names - set(refs) - set(effects)):
        warn(f"{AUDIO_DIR}/{name}.wav is not used by any clip")
    for name in dict.fromkeys(effects):
        if name not in wav_names:
            warn(f"sound effect audioClipSuccess {name!r} has no {AUDIO_DIR}/{name}.wav "
                 "(sound effects are outside narration)")
    return report


# ── CLI ───────────────────────────────────────────────────────────────


def _print_report(r: Report) -> None:
    print(f"{'OK  ' if r.ok else 'FAIL'} {r.path}")
    for e in r.errors:
        print(f"  error:   {e}")
    for w in r.warnings:
        print(f"  warning: {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lab resources zip: names, zipping, validation (#92).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_names = sub.add_parser("names", help="print the bundle names for a lab file")
    p_names.add_argument("lab", type=Path)
    p_names.add_argument("--bundle-name", help="folder/zip name to use instead of the derived one")

    p_zip = sub.add_parser("zip", help="zip a bundle folder as <folder>.zip")
    p_zip.add_argument("folder", type=Path)
    p_zip.add_argument("--out", type=Path, help="zip path (its name must match the folder)")

    p_val = sub.add_parser("validate", help="check zips or bundle folders against the layout")
    p_val.add_argument("paths", type=Path, nargs="+")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "names":
            lab = json.loads(args.lab.read_text(encoding="utf-8"))
            names = bundle_names(lab.get("labId", ""), args.bundle_name)
            print(f"zip:    {names.zip_file}\nfolder: {names.folder}/\njson:   {names.json_file}")
            return 0
        if args.cmd == "zip":
            print(zip_bundle(args.folder, args.out))
            return 0
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    reports = [validate_bundle(p) for p in args.paths]
    for r in reports:
        _print_report(r)
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
