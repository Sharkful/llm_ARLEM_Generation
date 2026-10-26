"""
Offline smoke test for the lab resources zip (#92): no API keys, no audio
services.

    python "Code/Narration/smoke_test.py"

Builds bundles from committed wave-2 labs with short generated tones in place
of narration. Checks:
- names derived from messy labIds
- the folder layout, and a zip that is byte-for-byte reproducible
- unpacking the way the headset does (lab_resources/<zip stem>/, first .json)
- one deliberate break per validator check, each caught
- every wave-2 json_lab lab bundles and validates
- the CLI exit codes

Outputs go to a temp dir that is removed afterwards. Exit code 1 on any failure.
"""

import contextlib
import io
import json
import math
import shutil
import struct
import sys
import tempfile
import wave
import zipfile
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle  # noqa: E402
from bundle import (  # noqa: E402
    BundleNames,
    audio_clip_refs,
    bundle_names,
    create_layout,
    validate_bundle,
    write_wav,
    zip_bundle,
)

ROOT = bundle.PROJECT_ROOT
OUTPUTS = ROOT / "Artifacts" / "Data" / "Benchmark" / "Outputs"
SAMPLE_LAB = OUTPUTS / "claude-sonnet-5_json_lab_L4_20260710_190528_output.json"

FAILED = False
TMP = Path(tempfile.mkdtemp(prefix="narration_smoke_"))


def check(cond, msg):
    global FAILED
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED = True


def tone(seconds=0.2, rate=24_000, freq=440.0):
    n = int(seconds * rate)
    return b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)
    )


PCM = tone()


def build(lab: dict, parent: Path, names: BundleNames | None = None, audio=True) -> Path:
    """A bundle folder for ``lab``: layout, one tone per audioClip, the lab JSON."""
    names = names or bundle_names(lab["labId"])
    folder = create_layout(parent, names)
    if audio:
        for name in dict.fromkeys(audio_clip_refs(lab)):
            write_wav(folder / bundle.AUDIO_DIR / f"{name}.wav", PCM)
    (folder / names.json_file).write_text(json.dumps(lab, indent=2), encoding="utf-8")
    return folder


def has_error(report, needle):
    return any(needle in e for e in report.errors)


def fresh(tag: str) -> Path:
    d = TMP / tag
    d.mkdir(parents=True)
    return d


lab = json.loads(SAMPLE_LAB.read_text(encoding="utf-8"))
refs = audio_clip_refs(lab)

# ── Names ─────────────────────────────────────────────────────────────

cases = {
    "dna_structure_replication": ("DnaStructureReplicationLab", "Dna_Structure_Replication.json"),
    "vsepr_theory_lab": ("VseprTheoryLab", "Vsepr_Theory.json"),
    "lab_vectors_projectile_motion": ("VectorsProjectileMotionLab", "Vectors_Projectile_Motion.json"),
    "GP1-AR-Vectors-Projectile-01": ("GP1ARVectorsProjectile01Lab", "GP1_AR_Vectors_Projectile_01.json"),
    "dna-structure-replication-201": ("DnaStructureReplication201Lab", "Dna_Structure_Replication_201.json"),
}
check(all(bundle_names(k) == BundleNames(*v) for k, v in cases.items()),
      "names come from labId: words capitalized, standalone 'lab' dropped, acronyms kept")
n = bundle_names("dna_structure_replication", "FullTransmissionLab")
check(n.folder == "FullTransmissionLab" and n.zip_file == "FullTransmissionLab.zip"
      and n.json_file == "Dna_Structure_Replication.json",
      "a bundle name override sets the folder and zip; the JSON name still comes from labId")
check(bundle_names("", "FooLab").json_file == "Foo.json",
      "with an empty labId, the JSON name falls back to the override")
for bad_id in ("", "___", "lab"):
    try:
        bundle_names(bad_id)
        ok = False
    except ValueError:
        ok = True
    check(ok, f"labId {bad_id!r} without an override is an error")
try:
    bundle_names("x", "../evil")
    ok = False
except ValueError:
    ok = True
check(ok, "an unsafe bundle name override is rejected")

# ── Build, zip, unpack like the headset ───────────────────────────────

work = fresh("good")
folder = build(lab, work)
names = bundle_names(lab["labId"])
check(all((folder / s).is_dir() for s in bundle.SUBDIRS),
      "layout has Audios/, Textures/ and Videos/")
check(validate_bundle(folder).ok, "the bundle folder validates before zipping")

zpath = zip_bundle(folder)
check(zpath.name == names.zip_file and zpath.parent == work, "zip is <folder>.zip beside the folder")
r = validate_bundle(zpath)
check(r.ok and not r.warnings, f"the zip validates with no warnings ({len(refs)} clips with audio)")

(work / "copy").mkdir()
again = zip_bundle(folder, work / "copy" / names.zip_file)
check(zpath.read_bytes() == again.read_bytes(), "zipping the same folder twice gives identical bytes")

try:
    zip_bundle(folder, work / "Other.zip")
    ok = False
except ValueError:
    ok = True
check(ok, "zip_bundle refuses a zip name that doesn't match its folder")

# The headset unpacks into lab_resources/ and opens lab_resources/<zip stem>/.
lab_resources = fresh("headset") / "lab_resources"
with zipfile.ZipFile(zpath) as zf:
    zf.extractall(lab_resources)
opened = lab_resources / zpath.stem
first_json = next((f for f in sorted(opened.iterdir()) if f.name.endswith(".json")), None)
check(first_json is not None and first_json.name == names.json_file,
      "after unpacking, lab_resources/<zip stem>/ holds the lab JSON")
check((opened / "Textures").is_dir() and (opened / "Videos").is_dir(),
      "empty Textures/ and Videos/ survive unpacking")
check(all((opened / "Audios" / f"{a}.wav").is_file() for a in refs),
      "every clip's audioClip resolves to Audios/<name>.wav")
with wave.open(str(opened / "Audios" / f"{refs[0]}.wav")) as w:
    check((w.getframerate(), w.getsampwidth(), w.getnchannels()) == (24_000, 2, 1),
          "WAVs are 24 kHz, 16-bit, mono")

# ── One break per check ───────────────────────────────────────────────


def broken(tag, mutate, as_zip=True, names=None):
    d = fresh(tag)
    f = build(lab, d, names=names)
    mutate(f)
    return validate_bundle(zip_bundle(f) if as_zip else f)


r = broken("rename", lambda f: None)
renamed = Path(r.path).with_name("WrongNameLab.zip")
Path(r.path).rename(renamed)
check(has_error(validate_bundle(renamed), "must match the zip name"),
      "a zip renamed away from its folder is an error")

check(has_error(broken("no_videos", lambda f: (f / "Videos").rmdir()), "missing folder"),
      "a missing Videos/ folder is an error")
check(has_error(broken("two_json", lambda f: (f / "Extra.json").write_text("{}")), "exactly one lab JSON"),
      "a second JSON file is an error")
check(has_error(broken("stray", lambda f: (f / "README.txt").write_text("x")), "unexpected file"),
      "a stray file at the top is an error")
check(has_error(broken("stray_dir", lambda f: (f / "Models").mkdir()), "unexpected folder"),
      "an unexpected folder at the top is an error")


def wav_with(path, rate=24_000, width=2, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * channels * 100 if width == 2 else b"\x00" * channels * 100)


first_wav = lambda f: f / "Audios" / f"{refs[0]}.wav"  # noqa: E731
check(has_error(broken("rate", lambda f: wav_with(first_wav(f), rate=44_100)), "44100 Hz"),
      "a 44.1 kHz WAV is an error")
check(has_error(broken("stereo", lambda f: wav_with(first_wav(f), channels=2)), "2 channel"),
      "a stereo WAV is an error")
check(has_error(broken("8bit", lambda f: wav_with(first_wav(f), width=1)), "8-bit"),
      "an 8-bit WAV is an error")
check(has_error(broken("empty_wav", lambda f: write_wav(first_wav(f), b"")), "has no audio"),
      "an empty WAV is an error")
check(has_error(broken("not_wav", lambda f: first_wav(f).write_bytes(b"ID3 not a wav")), "not a readable"),
      "a non-WAV file with a .wav name is an error")
check(has_error(broken("missing_wav", lambda f: first_wav(f).unlink()), f"has no Audios/{refs[0]}.wav"),
      "a clip whose audio file is missing is an error")
check(has_error(broken("mp3", lambda f: (f / "Audios" / "intro.mp3").write_bytes(b"x")), "only .wav"),
      "an mp3 in Audios/ is an error")
check(has_error(broken("nested", lambda f: (f / "Audios" / "sub").mkdir()), "must be flat"),
      "a subfolder in Audios/ is an error")
r = broken("orphan", lambda f: write_wav(f / "Audios" / "unused_clip.wav", PCM))
check(r.ok and any("not used by any clip" in w for w in r.warnings),
      "an audio file no clip uses is a warning, not an error")


def drop_modules(f):
    p = f / bundle_names(lab["labId"]).json_file
    d = json.loads(p.read_text())
    del d["modules"]
    p.write_text(json.dumps(d))


check(has_error(broken("bad_lab", drop_modules), "not a valid json_lab Lab"),
      "a JSON that isn't a valid json_lab Lab is an error")
check(has_error(broken("bad_json", lambda f: (f / bundle_names(lab["labId"]).json_file).write_text("{oops")),
                "not valid JSON"),
      "a malformed JSON file is an error")

d = fresh("macosx")
z = zip_bundle(build(lab, d))
with zipfile.ZipFile(z, "a") as zf:
    zf.writestr("__MACOSX/._x", b"")
check(has_error(validate_bundle(z), "exactly one top-level folder"),
      "a second top-level folder (e.g. __MACOSX/) is an error")
junk = fresh("junk") / "NotAZipLab.zip"
junk.write_bytes(b"not a zip")
check(has_error(validate_bundle(junk), "not a zip file"), "a non-zip file is an error")

# ── Every wave-2 json_lab lab bundles and validates ───────────────────

labs = sorted(glob(str(OUTPUTS / "*json_lab_L[34]_*_output.json")))
bad = []
for i, path in enumerate(labs):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "modules" not in d:
        continue
    names_i = BundleNames(f"Wave2Lab{i:03d}", bundle_names(d["labId"]).json_file)
    rep = validate_bundle(zip_bundle(build(d, fresh(f"wave2_{i:03d}"), names=names_i)))
    if not rep.ok:
        bad.append((Path(path).name, rep.errors[:2]))
check(not bad and len(labs) >= 100,
      f"all {len(labs)} wave-2 json_lab labs bundle and validate" + (f": {bad[:3]}" if bad else ""))

# ── CLI ───────────────────────────────────────────────────────────────

out = io.StringIO()
with contextlib.redirect_stdout(out):
    code_ok = bundle.main(["validate", str(zpath)])
    code_bad = bundle.main(["validate", str(junk)])
check(code_ok == 0 and code_bad == 1 and "OK" in out.getvalue() and "FAIL" in out.getvalue(),
      "validate exits 0 for a good bundle and 1 for a bad one")
out = io.StringIO()
with contextlib.redirect_stdout(out):
    code = bundle.main(["names", str(SAMPLE_LAB)])
check(code == 0 and names.zip_file in out.getvalue(), "names prints the derived zip name")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not FAILED else "\nSOME FAILED")
sys.exit(1 if FAILED else 0)
