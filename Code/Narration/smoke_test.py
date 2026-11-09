"""
Offline smoke test for narration: no API keys, no paid calls.

    python "Code/Narration/smoke_test.py"

Part 1 (the lab resources zip, #92) builds bundles from committed wave-2 labs with
short generated tones in place of narration. Checks:
- names derived from messy labIds
- the folder layout, and a zip that is byte-for-byte reproducible
- unpacking the way the headset does (lab_resources/<zip stem>/, first .json)
- one deliberate break per validator check, each caught
- malformed lab JSON, damaged zips (bad data, unsupported compression,
  encryption), unreadable folder files, `.` paths, symlinked zips and folders,
  names with stray spaces and upper-case extensions are reported as errors
  (or handled) rather than crashing or passing, and finished zips are readable
  by everyone (0644; not checked on Windows)
- every wave-2 json_lab lab bundles and validates
- the CLI exit codes

Part 1b (the spoken-form rules and glossary, #94): rule-by-rule cases, cases from
topics we have no lab for, what must be left alone, flags for undecided terms,
glossary layers, versions and errors, and the whole wave-2 corpus (no ordinary
word changed, a second pass changes nothing).

Part 2 (the narration CLI, #93) runs narrate_lab.py with a fake voice service:
clip names (generated, sanitized, deduplicated, shared, dropped), the bundled lab,
cost from token usage, reruns that skip unchanged clips, voice changes and
--overwrite, transient retries, failures and --resume, fail-fast errors, the
length limit, served-model warnings, unsupported inputs and the CLI flags.

Part 3 runs the real OpenAI and Gemini clients against local mock servers and
checks the requests we send (model, voice, style, PCM/SSE formats, no SDK retries)
and how responses are parsed (SSE audio and usage; raw PCM and WAV inline audio).
Part 2 and 3 need the project venv (pip install -r requirements.txt).

Outputs go to a temp dir that is removed afterwards. Exit code 1 on any failure.
"""

import contextlib
import csv
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import wave
import zipfile
import zlib
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
check(has_error(broken("two_json", lambda f: (f / "Extra.json").write_text("{}", encoding="utf-8")), "exactly one lab JSON"),
      "a second JSON file is an error")
check(has_error(broken("stray", lambda f: (f / "README.txt").write_text("x", encoding="utf-8")), "unexpected file"),
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
    d = json.loads(p.read_text(encoding="utf-8"))
    del d["modules"]
    p.write_text(json.dumps(d), encoding="utf-8")


check(has_error(broken("bad_lab", drop_modules), "not a valid json_lab Lab"),
      "a JSON that isn't a valid json_lab Lab is an error")
check(has_error(broken("bad_json", lambda f: (f / bundle_names(lab["labId"]).json_file).write_text("{oops", encoding="utf-8")),
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

# ── Review fixes: report problems instead of crashing or passing ──────


def safe_validate(p):
    """validate_bundle, but an exception becomes a failed check instead of a crash."""
    try:
        return validate_bundle(p)
    except Exception as exc:  # noqa: BLE001 - the point is to catch anything
        return exc


def with_json(tag, edit):
    """A bundle folder whose lab JSON has been changed by ``edit(dict)``."""
    d = fresh(tag)
    f = build(lab, d)
    p = f / bundle_names(lab["labId"]).json_file
    data = json.loads(p.read_text(encoding="utf-8"))
    edit(data)
    p.write_text(json.dumps(data), encoding="utf-8")
    return f


def set_first_audio(value):
    def edit(data):
        data["modules"][0]["clips"][0]["audioClip"] = value
    return edit


malformed = {
    "audioClip is a number": set_first_audio(3),
    "modules is an object": lambda data: data.__setitem__("modules", {"a": 1}),
    "a module is a string": lambda data: data["modules"].__setitem__(0, "module"),
    "a clip is a string": lambda data: data["modules"][0]["clips"].__setitem__(0, "clip"),
}
for i, (what, edit) in enumerate(malformed.items()):
    rep = safe_validate(with_json(f"malformed_{i}", edit))
    check(not isinstance(rep, Exception) and not rep.ok,
          f"malformed lab JSON ({what}) is an error, not a crash"
          + (f" (raised {type(rep).__name__})" if isinstance(rep, Exception) else ""))

bad_shape_zip = zip_bundle(with_json("malformed_cli", set_first_audio(3)))
out = io.StringIO()
try:
    with contextlib.redirect_stdout(out):
        code = bundle.main(["validate", str(bad_shape_zip), str(zpath)])
    reported = out.getvalue()
except Exception as exc:  # noqa: BLE001
    code, reported = f"raised {type(exc).__name__}", ""
check(code == 1 and str(bad_shape_zip) in reported and str(zpath) in reported,
      f"validate on a malformed bundle plus a good one still reports both (exit {code})")


def flip_byte_in(zip_file: Path, entry: str) -> None:
    """Corrupt one byte in the middle of an entry's stored data."""
    with zipfile.ZipFile(zip_file) as zf:
        info = zf.getinfo(entry)
    raw = bytearray(zip_file.read_bytes())
    off = info.header_offset
    name_len, extra_len = struct.unpack("<HH", raw[off + 26: off + 30])
    start = off + 30 + name_len + extra_len
    raw[start + info.compress_size // 2] ^= 0xFF
    zip_file.write_bytes(bytes(raw))


dmg = zip_bundle(build(lab, fresh("damaged")))
flip_byte_in(dmg, f"{names.folder}/Audios/{refs[0]}.wav")
rep = safe_validate(dmg)
check(not isinstance(rep, Exception) and has_error(rep, "damaged"),
      "a damaged zip (one flipped byte) is reported as damaged, not a crash"
      + (f" (raised {type(rep).__name__})" if isinstance(rep, Exception) else ""))

dot_work = fresh("dot")
dot_folder = build(lab, dot_work)
here = Path.cwd()
dot_rep = dot_zip = None
try:
    os.chdir(dot_folder)
    dot_rep = safe_validate(Path("."))
    try:
        dot_zip = zip_bundle(Path("."))
    except Exception as exc:  # noqa: BLE001
        dot_zip = exc
finally:
    os.chdir(here)
check(isinstance(dot_rep, bundle.Report) and dot_rep.ok,
      "validate '.' from inside the bundle folder works"
      + (f": {dot_rep.errors[:1]}" if isinstance(dot_rep, bundle.Report) else f" (got {dot_rep!r})"))
check(isinstance(dot_zip, Path) and dot_zip == dot_work.resolve() / names.zip_file,
      f"zip '.' from inside the bundle folder writes {names.zip_file} beside it")

rep = safe_validate(with_json("trailing_space", set_first_audio(refs[0] + " ")))
check(not isinstance(rep, Exception) and has_error(rep, "not a safe file name"),
      "an audioClip with a stray space is an error even when the trimmed name has a file")
rep = safe_validate(with_json("blank_name", set_first_audio("   ")))
check(not isinstance(rep, Exception) and has_error(rep, "not a safe file name"),
      "a whitespace-only audioClip is an error (the headset would look it up)")


def upper_ext(f):
    p = f / "Audios" / f"{refs[0]}.wav"
    p.rename(p.with_suffix(".WAV"))


rep = broken("upper_ext", upper_ext)
check(has_error(rep, "lowercase") and has_error(rep, f"has no Audios/{refs[0]}.wav"),
      "an upper-case .WAV is an error and doesn't count as the clip's file")
if os.name == "nt":
    # Windows has no Unix permission bits: st_mode reads 0o666 whatever chmod sets.
    check(True, "the finished zip's 0644 mode is not checked on Windows")
else:
    check(zpath.stat().st_mode & 0o777 == 0o644,
          f"the finished zip is readable by everyone (mode {oct(zpath.stat().st_mode & 0o777)})")


def set_method(zip_file: Path, entry: str, method: int | None = None, encrypt=False) -> None:
    """Rewrite one entry's compression method or encryption flag in both its local
    header and its central-directory record."""
    with zipfile.ZipFile(zip_file) as zf:
        info = zf.getinfo(entry)
    raw = bytearray(zip_file.read_bytes())
    headers = [info.header_offset]
    i = 0
    while (i := raw.find(b"PK\x01\x02", i)) >= 0:
        n = struct.unpack("<H", raw[i + 28: i + 30])[0]
        if raw[i + 46: i + 46 + n].decode() == entry:
            headers.append(i + 2)  # central fields sit 2 bytes later than local ones
        i += 4
    for h in headers:
        if method is not None:
            raw[h + 8: h + 10] = struct.pack("<H", method)
        if encrypt:
            flags = struct.unpack("<H", raw[h + 6: h + 8])[0] | 0x1
            raw[h + 6: h + 8] = struct.pack("<H", flags)
    zip_file.write_bytes(bytes(raw))


json_entry = f"{names.folder}/{names.json_file}"
z = zip_bundle(build(lab, fresh("method99")))
set_method(z, json_entry, method=99)
rep = safe_validate(z)
check(isinstance(rep, bundle.Report) and has_error(rep, "damaged"),
      "an entry with an unsupported compression method is reported as damaged, not a crash"
      + ("" if isinstance(rep, bundle.Report) else f" (raised {type(rep).__name__})"))
z = zip_bundle(build(lab, fresh("encrypted")))
set_method(z, json_entry, encrypt=True)
rep = safe_validate(z)
check(isinstance(rep, bundle.Report) and has_error(rep, "damaged"),
      "an encrypted entry is reported as damaged, not a crash"
      + ("" if isinstance(rep, bundle.Report) else f" (raised {type(rep).__name__})"))

real_listing = bundle._listing
bundle._listing = lambda path: (_ for _ in ()).throw(zlib.error())
try:
    rep = safe_validate(zpath)
finally:
    bundle._listing = real_listing
check(isinstance(rep, bundle.Report) and any(e.rstrip().endswith("(error)") for e in rep.errors),
      "an error with no message still names its type (e.g. 'zip is damaged (error)')"
      + (f": {rep.errors[:1]}" if isinstance(rep, bundle.Report) else ""))

link_work = fresh("symlinks")
try:
    os.symlink(link_work / "missing.wav", link_work / "probe")
    (link_work / "probe").unlink()
    can_link = True
except (OSError, NotImplementedError):
    can_link = False
if can_link:
    f = build(lab, link_work / "broken")
    os.symlink(link_work / "missing.wav", f / "Audios" / "zzz.wav")
    rep = safe_validate(f)
    check(isinstance(rep, bundle.Report) and has_error(rep, "cannot be read"),
          "an unreadable file in a bundle folder (broken symlink) is an error, not a crash"
          + ("" if isinstance(rep, bundle.Report) else f" (raised {type(rep).__name__})"))

    z = zip_bundle(build(lab, link_work / "zip"))
    target = z.with_name("cache.zip")
    z.rename(target)
    os.symlink(target, z)
    rep = safe_validate(z)
    check(isinstance(rep, bundle.Report) and rep.ok,
          f"a symlink named {names.zip_file} pointing at cache.zip validates by the link's name"
          + (f": {rep.errors[:1]}" if isinstance(rep, bundle.Report) else ""))

    target_dir = build(lab, link_work / "build_123")
    moved = target_dir.rename(target_dir.parent / "built")
    link = link_work / names.folder
    os.symlink(moved, link)
    try:
        made = zip_bundle(link)
    except Exception as exc:  # noqa: BLE001
        made = exc
    with zipfile.ZipFile(made) if isinstance(made, Path) else contextlib.nullcontext() as zf:
        tops = {n.split("/")[0] for n in zf.namelist()} if zf else set()
    check(isinstance(made, Path) and made.name == names.zip_file and tops == {names.folder},
          f"zipping a symlinked folder {names.folder} -> built/ is named after the link"
          + (f" (got {made.name}, top folders {sorted(tops)})" if isinstance(made, Path) else f" (raised {made!r})"))
else:
    check(True, "symlink checks skipped: this system can't create symlinks")


def upper_json(f):
    p = f / names.json_file
    p.rename(p.with_suffix(".JSON"))


rep = broken("upper_json", upper_json, as_zip=False)
check(has_error(rep, "exactly one lab JSON") and has_error(rep, ".JSON"),
      "an upper-case .JSON lab file is an error (the headset looks for .json)")

# ── Every wave-2 json_lab lab bundles and validates ───────────────────

labs = sorted(glob(str(OUTPUTS / "*json_lab_L[34]_*_output.json")))
bad, skipped, bundled = [], [], 0
for i, path in enumerate(labs):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "modules" not in d:
        skipped.append(Path(path).name)
        continue
    names_i = BundleNames(f"Wave2Lab{i:03d}", bundle_names(d["labId"]).json_file)
    rep = validate_bundle(zip_bundle(build(d, fresh(f"wave2_{i:03d}"), names=names_i)))
    bundled += 1
    if not rep.ok:
        bad.append((Path(path).name, rep.errors[:2]))
check(not bad and not skipped and bundled >= 100,
      f"{bundled}/{len(labs)} wave-2 json_lab labs bundle and validate"
      + (f"; skipped (no modules): {skipped[:3]}" if skipped else "")
      + (f"; failed: {bad[:3]}" if bad else ""))

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

# ══ Part 1b: spoken-form rules and glossary (#94) ═════════════════════

print("\n-- Part 1b: spoken-form rules and glossary")
try:
    import spoken_rules  # noqa: E402
    import spoken_form  # noqa: E402
    import yaml  # noqa: E402
    HAVE_RULES = True
except ImportError as exc:
    HAVE_RULES = False
    check(False, f"spoken-form rules import (run in the project venv): {exc}")

if HAVE_RULES:
    SHARED = spoken_rules.load_glossary()

    def say(text, *context, topic=None, glossary=None):
        """The rules' output for text, in a lab whose narration is text plus context."""
        g = glossary or (spoken_rules.load_glossary(topic) if topic else SHARED)
        ctx = spoken_rules.LabContext.from_texts([text, *context], topic)
        return spoken_rules.apply_rules(text, g, ctx)

    # Each case: (narration, expected spoken text, other narration in the same lab).
    # Grouped by rule; the second group is written from topics we have no lab for,
    # to show the rules carry over.
    CASES = {
        "wave-2 notation": [
            ("The H-O-H bond angle is 104.5°.", "The H O H bond angle is 104 point 5 degrees.", ()),
            ("Methane, CH₄, and XeF4 both apply VSEPR.", "Methane, C H 4, and X e F 4 both apply vesper.", ()),
            ("Read the strand 5' to 3': A-T-G-C-C.", "Read the strand 5 prime to 3 prime: A, T, G, C, C.", ()),
            ("It reads 3'-TACGGTCA-5'.", "It reads 3 prime, T, A, C, G, G, T, C, A, 5 prime.", ()),
            ("Polymerase moves 5'→3' and adds to the 3' end.",
             "Polymerase moves 5 prime to 3 prime and adds to the 3 prime end.", ()),
            ("It travels in the 3'-to-5' direction.", "It travels in the 3 prime to 5 prime direction.", ()),
            ("Mars orbits at 1.52 AU; Earth at 1 AU.",
             "Mars orbits at 1 point 5 2 astronomical units; Earth at 1 astronomical unit.", ()),
            ("Press NEXT to go on.", "Press next to go on.", ("Then the next step.",)),
            ("The heart beats 150-200 times.", "The heart beats 150 to 200 times.", ()),
            ("So vx = v cosθ and vy = 0.", "So v x equals v cos theta and v y equals 0.",
             ("Each vector has an x component.",)),
            ("At maximum height, Vy is zero. By symmetry it falls back.",
             "At maximum height, V y is zero. By symmetry it falls back.", ("the vertical component",)),
            ("the 45° result and a 180° turn, at 45°", "the 45 degree result and a 180 degree turn, at 45 degrees", ()),
            ("Track the pulmonary trunk/arteries and/or veins.",
             "Track the pulmonary trunk or arteries and or veins.", ()),
            ("Earth’s orbit doesn’t change.", "Earth’s orbit doesn’t change.", ()),
        ],
        "other topics": [
            ("A 9 V battery drives 0.5 A through a 220 Ω resistor.",
             "A 9 volt battery drives 0 point 5 amps through a 220 ohm resistor.", ("in the circuit",)),
            ("Gravity accelerates it at 9.8 m/s² near the surface.",
             "Gravity accelerates it at 9 point 8 meters per second squared near the surface.", ()),
            ("Speed is measured in m/s.", "Speed is measured in meters per second.", ()),
            ("Glucose, C6H12O6, burns in O2: 2H2 + O2 → 2H2O.",
             "Glucose, C 6 H 12 O 6, burns in O 2: 2 H 2 plus O 2 to 2 H 2 O.", ()),
            ("The 5' cap and the 3' end protect the mRNA.",
             "The 5 prime cap and the 3 prime end protect the mRNA.", ()),
            ("Its 5' and 3' untranslated regions differ.", "Its 5 prime and 3 prime untranslated regions differ.",
             ("This RNA is spliced.",)),
            ("The doorway is 7' high and she is 5'10\".", "The doorway is 7 feet high and she is 5 feet 10 inches.",
             ("How tall is it?",)),
            ("Polaris sits at a declination of 89°15'.", "Polaris sits at a declination of 89 degrees 15 arcminutes.", ()),
            ("The result is significant if p < 0.05.", "The result is significant if p is less than 0 point 0 5.", ()),
            ("Water boils at 100 °C and it was -40°F outside.",
             "Water boils at 100 degrees Celsius and it was minus 40 degrees Fahrenheit outside.", ()),
            ("Expect 10–15% growth.", "Expect 10 to 15 percent growth.", ()),
            ("Do NOT touch the probe.", "Do not touch the probe.", ("It is not hot.",)),
            ("The ball leaves with speed v0, then v₀ halves.", "The ball leaves with speed v zero, then v zero halves.", ()),
            ("Leave a 5mm gap; it is 1 mm wide.", "Leave a 5 millimeter gap; it is 1 millimeter wide.", ()),
            ("The change Δx is small.", "The change delta x is small.", ()),
            ("Its pH is 7.4.", "Its P H is 7 point 4.", ()),
            ("ClF3, BrF5 and I3- form; Ca2+ and Na+ ions; Cl- too.",
             "C l F 3, B r F 5 and I 3 minus form; C a 2 plus and N a plus ions; C l minus too.", ()),
        ],
        "left alone": [
            ("In the 1990's and the 1990s, 3D models came in.", "In the 1990's and the 1990s, 3D models came in.", ()),
            ("He said it is in Iron Age tools, as we saw.", "He said it is in Iron Age tools, as we saw.", ()),
            ("It moves by itself, an ox and my ax.", "It moves by itself, an ox and my ax.", ("the x component",)),
            ("Take 2 C atoms and 3 H atoms.", "Take 2 C atoms and 3 H atoms.", ()),
            ("A T-shirt, an X-ray and a U-turn.", "A T-shirt, an X-ray and a U-turn.", ()),
            ("The year 2024 had 365 days.", "The year 2024 had 365 days.", ()),
            ("Jupiter at 5.2.", "Jupiter at 5 point 2.", ()),
        ],
    }
    for group, cases in CASES.items():
        bad = []
        for text, expected, context in cases:
            r = say(text, *context)
            if r.spoken_text != expected:
                bad.append(f"{text!r} -> {r.spoken_text!r} (expected {expected!r})")
        check(not bad, f"rules, {group}: {len(cases) - len(bad)}/{len(cases)} cases"
              + ("".join(f"\n       {b}" for b in bad)))

    r = say("The CMB is ancient light.")
    check(r.spoken_text == "The CMB is ancient light." and r.unresolved == ["CMB"] and "CMB" in r.flags[0],
          "an acronym not in the glossary is left for the voice and flagged")
    r = say("Set it to 5' first.")
    check(r.spoken_text == "Set it to 5' first." and r.unresolved == ["5'"] and "no clues" in r.flags[0],
          "a prime with no clues in the lab is left as written and flagged")
    r = say("It points AWAY from the fork.")
    check(r.spoken_text == "It points AWAY from the fork." and r.unresolved == ["AWAY"],
          "a capitalized word never written in lowercase in the lab is flagged, not guessed")
    r = say("Sulfate, SO4²⁻, then x².")
    check(r.spoken_text == "Sulfate, SO4²⁻, then x squared." and "⁻" in r.unresolved,
          "a superscript charge is left as written and flagged; a superscript power is still said")
    r = say("The angle ∡ is shown.")
    check(r.unresolved == ["∡"] and "no rule" in r.flags[0], "a symbol with no rule is flagged")
    r = say("Set it to 5' first.", topic="dna_structure_and_replication")
    check(r.spoken_text == "Set it to 5 prime first." and not r.flags,
          "a topic glossary pin decides a prime the lab's own words don't")
    r = say("Then 5'-ATGC-3' and the 3' end.", topic="vsepr_molecular_geometry")
    check(r.spoken_text == "Then 5 prime, A, T, G, C, 3 prime and the 3 prime end.",
          "local patterns (strand ends, 'the 3' end') are primes in any lab")
    changes = say("XeF4 has 109.5° angles.").changes
    check([c["rule"] for c in changes] == ["degrees", "formula", "decimal"]
          and changes[1] == {"rule": "formula", "original": "XeF4", "spoken": "X e F 4"},
          "every change records its rule, the original and the spoken text")

    # Glossary layers, versions and errors.
    gdir = fresh("glossary")
    extra = gdir / "extra.yaml"
    extra.write_text("terms:\n  CMB: C M B\npins:\n  number_prime: feet\n", encoding="utf-8")
    run_folder = gdir / "run"
    run_folder.mkdir()
    (run_folder / spoken_rules.RUN_GLOSSARY).write_text("terms:\n  CMB: cosmic microwave background\n",
                                                        encoding="utf-8")
    g1 = spoken_rules.load_glossary(None, [extra])
    g2 = spoken_rules.load_glossary(None, [extra], run_folder)
    check(say("The CMB at 5'.", glossary=g1).spoken_text == "The C M B at 5 feet."
          and say("The CMB.", glossary=g2).spoken_text == "The cosmic microwave background."
          and len(g2.files) == 3 and g2.files[-1].endswith("glossary.yaml"),
          "glossary layers: an extra file adds terms and pins, a run folder's glossary.yaml wins")
    check(g1.version != SHARED.version and spoken_rules.load_glossary(None, [extra]).version == g1.version
          and spoken_rules.load_glossary("dna_structure_and_replication").version != SHARED.version,
          "the glossary version is a content hash: same files same version, any change a new one")
    errors = []
    for body in ("- just a list\n", "terms:\n  X: [1, 2]\n", "units:\n  q: {say: 5}\n",
                 "pins:\n  number_prime: furlongs\n", "colours: {}\n", "terms: {a: b\n"):
        (gdir / "bad.yaml").write_text(body, encoding="utf-8")
        try:
            spoken_rules.load_glossary(None, [gdir / "bad.yaml"])
            errors.append(f"accepted: {body!r}")
        except spoken_rules.GlossaryError:
            pass
    check(not errors, "a malformed glossary is a clear GlossaryError, not a crash" + "".join(errors))
    for path in sorted((spoken_rules.GLOSSARY_DIR / "topics").glob("*.yaml")):
        spoken_rules.load_glossary(path.stem)
    check(True, "every committed topic glossary loads")

    # The whole wave-2 corpus: no crash, no ordinary word changed, stable on a rerun.
    n_clips, word_flags, unstable, flagged, changed = 0, [], [], 0, 0
    for path, lab_ in spoken_rules._labs([OUTPUTS]):
        topic = spoken_rules._topic_of(path)
        g = spoken_rules.load_glossary(topic)
        ctx = spoken_rules.LabContext.from_lab(lab_, topic)
        for c in spoken_rules._clips(lab_):
            n_clips += 1
            r = spoken_rules.apply_rules(c["narration"], g, ctx)
            changed += bool(r.changes)
            flagged += bool(r.flags)
            f, _ = spoken_form.check_spoken(c["narration"], r.spoken_text)
            if f:
                word_flags.append((c["narration"][:80], f))
            if spoken_rules.apply_rules(r.spoken_text, g, ctx).spoken_text != r.spoken_text:
                unstable.append(r.spoken_text[:80])
    check(n_clips > 1700 and not word_flags and not unstable and flagged <= 5,
          f"all {n_clips} wave-2 clips: {changed} changed, no ordinary word changed, running the rules "
          f"twice changes nothing, {flagged} flagged" + "".join(f"\n       {w}" for w in word_flags[:5]))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = spoken_rules.main(["preview", str(SAMPLE_LAB)])
    check(code == 0 and "5 prime to 3 prime" in out.getvalue() and "number prime -> prime" in out.getvalue(),
          "spoken_rules.py preview shows the written and spoken text")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = spoken_rules.main(["terms", str(OUTPUTS), "--json"])
    report = json.loads(out.getvalue())
    check(code == 0 and report["clips"] == n_clips and report["rules"]["formula"] > 0,
          "spoken_rules.py terms reports rule counts and the terms to review")

# ══ Part 2: the narration CLI with a fake voice service (#93) ═════════

print("\n-- Part 2: narration CLI")
try:
    import narrate_lab  # noqa: E402  (first: it puts Code/Benchmark on the path)
    import benchmark  # noqa: E402  (the transient retryer and error helpers)
    import tenacity  # noqa: E402
    import tts  # noqa: E402
    HAVE_NARRATION = True
except ImportError as exc:
    HAVE_NARRATION = False
    check(False, f"narration modules import (run in the project venv): {exc}")

if HAVE_NARRATION:
    benchmark._TRANSIENT_WAIT = tenacity.wait_none()   # no real back-off in tests
    from narrate_lab import Settings, plan_clips, bundled_lab, run as narrate

    class HTTPError(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.status_code = status

    class FakeTTS:
        """Returns a 0.2 s tone with fixed usage; ``script(n, text)`` may return an
        exception to raise for call n (0-based)."""
        def __init__(self, script=None, served="gpt-4o-mini-tts-2025-12-15", usage=(50, 1000)):
            self.calls, self.script, self.served, self.usage = [], script, served, usage

        def synthesize(self, text):
            n = len(self.calls)
            self.calls.append(text)
            exc = self.script(n, text) if self.script else None
            if exc:
                raise exc
            tin, tout = self.usage if self.usage else (None, None)
            return tts.Synthesis(PCM, tin, tout, self.served)

    def settings_for(path, voice="marin", provider="openai"):
        return Settings(input_path=Path(path), provider=provider, voice=voice,
                        style=tts.STYLE_INSTRUCTION, spoken_form="none", topic=None,
                        bundle_name=None)

    quiet = lambda *a, **k: None  # noqa: E731
    COST_PER_CLIP = 50 * 0.60 / 1e6 + 1000 * 12.00 / 1e6   # gpt-4o-mini-tts rates

    def manifest_of(run_dir):
        return json.loads((Path(run_dir) / "narration_manifest.json").read_text(encoding="utf-8"))

    # ── Planning edge cases ──────────────────────────────────────────
    edge = json.loads(SAMPLE_LAB.read_text(encoding="utf-8"))
    m1, m2 = edge["modules"][0]["clips"], edge["modules"][1]["clips"]
    first_name = m1[0]["audioClip"]
    m1[1]["audioClip"] = None                              # narration, no name -> generated
    m2[0]["audioClip"] = "  Helix Twist.mp3 "              # unsafe -> sanitized
    m2[1]["audioClip"] = first_name                        # same name, other words -> deduplicated
    edge["modules"][2]["clips"][0]["audioClip"] = first_name
    edge["modules"][2]["clips"][0]["narration"] = m1[0]["narration"]   # same name + words -> shared
    edge["modules"][2]["clips"][1]["narration"] = "   "    # name, no narration -> dropped
    edge["modules"][3]["clips"][0].pop("audioClip", None)
    edge["modules"][3]["clips"][0]["narration"] = None     # neither -> silent
    for i in (4, 5):                                       # a clipName used twice, no names
        edge["modules"][i]["clips"][0].update(clipName="recap", audioClip=None)
    edge["modules"][6]["clips"][0].update(clipName=first_name, audioClip=None)  # = another's name
    edge_path = fresh("edge_input") / "edge_json_lab_L4_x_output.json"
    edge_path.write_text(json.dumps(edge, indent=2), encoding="utf-8")
    plans = plan_clips(edge)
    by_pos = {(p.module, p.clip): p for p in plans}
    check(by_pos[(1, 2)].name == m1[1]["clipName"] and by_pos[(1, 2)].name_change == "generated",
          "a missing name is the clipName alone when no other clip uses that clipName")
    check(by_pos[(5, 1)].name == "m05_recap" and by_pos[(6, 1)].name == "m06_recap",
          "a clipName used by two clips gets the module prefix (m05_recap, m06_recap)")
    check(by_pos[(7, 1)].name == f"m07_{first_name}",
          "a clipName equal to another clip's audio name gets the module prefix")
    check(by_pos[(2, 1)].name == "Helix_Twist" and by_pos[(2, 1)].name_change == "sanitized",
          "an unsafe audioClip ('  Helix Twist.mp3 ') is sanitized to Helix_Twist")
    check(by_pos[(2, 2)].name == f"{first_name}_2" and by_pos[(2, 2)].name_change == "deduplicated",
          "a reused name with different narration gets a _2 suffix")
    check(by_pos[(3, 1)].action == "shared" and by_pos[(3, 1)].name == first_name,
          "a reused name with the same narration shares one file")
    check(by_pos[(3, 2)].action == "dropped" and by_pos[(3, 2)].name is None,
          "a name on a clip with blank narration is dropped")
    check(by_pos[(4, 1)].action == "silent", "a clip with neither is silent")
    b = bundled_lab(edge, plans)
    check(b["modules"][0]["clips"][1]["audioClip"] == by_pos[(1, 2)].name
          and "audioClip" not in b["modules"][2]["clips"][1]
          and edge["modules"][2]["clips"][1]["audioClip"] is not None,
          "renames and drops apply to the bundled copy only")

    fake = FakeTTS()
    edge_run = TMP / "runs" / "edge"
    code = narrate(settings_for(edge_path), edge_run, provider=fake, log=quiet)
    em = manifest_of(edge_run)
    zip_ok = em["bundle"]["validation"] and em["bundle"]["validation"]["ok"]
    check(code == 0 and em["status"] == "complete" and zip_ok,
          "the edge-case lab narrates, and its zip validates (dropped names removed)")
    check(len(fake.calls) == sum(p.action == "narrate" for p in plans),
          "each name is synthesized once (a shared clip makes no extra request)")
    check(json.loads(edge_path.read_text(encoding="utf-8")) == edge, "the input lab file is never modified")
    check(em["counts"]["names"] == {"generated": 4, "sanitized": 1, "deduplicated": 1, "dropped": 1}
          and em["counts"]["shared"] == 1 and em["counts"]["silent"] == 1,
          "the manifest counts every name change")

    # ── A real lab, end to end ───────────────────────────────────────
    real_run = TMP / "runs" / "real"
    fake = FakeTTS()
    sample_bytes = SAMPLE_LAB.read_bytes()
    code = narrate(settings_for(SAMPLE_LAB), real_run, provider=fake, log=quiet)
    rm = manifest_of(real_run)
    zip_file = real_run / rm["bundle"]["zip_file"]
    check(code == 0 and rm["status"] == "complete" and validate_bundle(zip_file).ok,
          f"a wave-2 lab narrates into a valid {rm['bundle']['zip_file']}")
    check(rm["counts"]["generated"] == len(refs) == len(fake.calls),
          f"one request per narrated clip ({len(refs)})")
    check(abs(rm["totals"]["kept_audio_cost_usd"] - len(refs) * COST_PER_CLIP) < 1e-9
          and abs(rm["totals"]["total_cost_usd"] - len(refs) * COST_PER_CLIP) < 1e-9
          and rm["totals"]["input_tokens"] == 50 * len(refs),
          "cost comes from reported token usage at the tracking/pricing.py rates")
    rc = rm["calls"]
    check(len(rc) == len(refs) == rm["totals"]["calls"] and rm["totals"]["calls_failed"] == 0
          and all(x["step"] == "tts" and x["run"] == 1 and x["status"] == "ok" and x["kind"] == "first"
                  and abs(x["cost_usd"] - COST_PER_CLIP) < 1e-12 and x["started_at"] for x in rc)
          and sorted(x["clip"] for x in rc) == sorted(refs)
          and all(c["calls"] == 1 and abs(c["spend_usd"] - COST_PER_CLIP) < 1e-12
                  for c in rm["clips"] if c["status"] == "generated"),
          "every request is logged with its run, clip and cost, and each clip counts its requests")
    check(abs(rm["totals"]["audio_seconds"] - 0.2 * len(refs)) < 1e-6, "audio seconds are recorded")
    check(rm["source"]["lab_name"] == "dna_structure_and_replication" and rm["source"]["level"] == "L4"
          and rm["source"]["source_model"] == "claude-sonnet-5",
          "topic, level and source model come from the paired metrics file")
    check(rm["voice_service"]["model"] == "gpt-4o-mini-tts-2025-12-15"
          and rm["voice_service"]["served_models"] == ["gpt-4o-mini-tts-2025-12-15"]
          and rm["voice_service"]["sample_rate"] == 24000 and not rm["warnings"],
          "the manifest records model, served model and audio format; no warnings")
    check(rm["provenance"]["git_commit"] and rm["runs"][0]["finished_at"],
          "provenance (git commit) and run timestamps are recorded")
    check(SAMPLE_LAB.read_bytes() == sample_bytes, "the committed wave-2 output is untouched")
    check(fake.calls[0] == lab["modules"][0]["clips"][0]["narration"].strip(),
          "--spoken-form none sends the narration as written")

    fake = FakeTTS()
    code = narrate(settings_for(SAMPLE_LAB), real_run, provider=fake, log=quiet)
    rm2 = manifest_of(real_run)
    check(code == 0 and not fake.calls and rm2["counts"]["skipped"] == len(refs)
          and rm2["totals"]["this_run_cost_usd"] == 0 and rm2["calls"] == rm["calls"]
          and rm2["totals"]["total_cost_usd"] == rm["totals"]["total_cost_usd"]
          and len(rm2["runs"]) == 2,
          "a rerun skips every unchanged clip, spends nothing, keeps the call log, and logs a second run")
    orphan = real_run / rm["bundle"]["folder"] / "Audios" / "no_longer_used.wav"
    write_wav(orphan, PCM)
    fake = FakeTTS()
    narrate(settings_for(SAMPLE_LAB, voice="cedar"), real_run, provider=fake, log=quiet)
    check(len(fake.calls) == len(refs), "changing the voice regenerates every clip (source hash)")
    check(not orphan.exists() and any("no_longer_used.wav" in w for w in manifest_of(real_run)["warnings"])
          and not validate_bundle(real_run / rm["bundle"]["zip_file"]).warnings,
          "audio for a name no clip uses any more is removed before zipping")
    fake = FakeTTS()
    narrate(settings_for(SAMPLE_LAB, voice="cedar"), real_run, overwrite=True, provider=fake, log=quiet)
    check(len(fake.calls) == len(refs), "--overwrite regenerates every clip")
    ot = manifest_of(real_run)["totals"]
    check(ot["calls"] == 3 * len(refs) and abs(ot["total_cost_usd"] - 3 * len(refs) * COST_PER_CLIP) < 1e-9
          and abs(ot["this_run_cost_usd"] - len(refs) * COST_PER_CLIP) < 1e-9
          and abs(ot["kept_audio_cost_usd"] - len(refs) * COST_PER_CLIP) < 1e-9,
          "regenerated audio still counts: the total is every run's spend, beside the kept audio's cost")

    # ── Retries, failures, resume ────────────────────────────────────
    fake = FakeTTS(script=lambda n, t: HTTPError(503) if n == 0 else None)
    r = TMP / "runs" / "transient"
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    tm = manifest_of(r)
    check(code == 0 and tm["clips"][0]["transient_retries"] == 1 and tm["totals"]["transient_retries"] == 1,
          "a 503 is retried and counted as a transient retry")
    te = tm["clips"][0]["transient_errors"]
    check(len(te) == 1 and te[0]["status"] == 503 and te[0]["type"] == "HTTPError"
          and te[0]["reason"] == "the provider was temporarily unavailable (HTTP 503)",
          "each retried error is recorded with its status, type and a readable reason")
    c0 = tm["clips"][0]
    mine = [x for x in tm["calls"] if x["clip"] == c0["audio_name"]]
    check([(x["attempt"], x["kind"], x["status"]) for x in mine] == [(1, "first", "error"), (2, "retry", "ok")]
          and mine[0]["cost_usd"] == 0.0 and mine[0]["error_type"] == "HTTPError"
          and "HTTP 503" in mine[0]["error_reason"]
          and c0["calls"] == 2 and c0["calls_failed"] == 1 and abs(c0["spend_usd"] - COST_PER_CLIP) < 1e-12
          and tm["totals"]["calls_failed"] == 1 and tm["totals"]["calls_cost_unknown"] == 0,
          "a retried clip logs both requests; an HTTP error reply costs $0 (not billed)")
    fake = FakeTTS(script=lambda n, t: HTTPError(429) if n < 2 else None)
    r = TMP / "runs" / "transient_twice"
    narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    te = manifest_of(r)["clips"][0]["transient_errors"]
    check([e["reason"] for e in te] == ["rate limit or quota reached: too many requests for now (HTTP 429)"] * 2,
          "two retries on one clip record both reasons")

    # ── Quotas and rate limits: stop on a used-up quota, wait as asked otherwise ──
    def gemini_429(quota_id, delay="33s", limit="100"):
        """A google-genai style 429 (error body in .details, status in .code)."""
        exc = Exception("429 RESOURCE_EXHAUSTED. You exceeded your current quota")
        exc.code = 429
        exc.details = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                 "message": "You exceeded your current quota, please check your plan",
                                 "details": [
                                     {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                      "violations": [{"quotaId": quota_id, "quotaValue": limit}]},
                                     {"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                      "retryDelay": delay}]}}
        return exc

    fake = FakeTTS(script=lambda n, t: gemini_429("GenerateRequestsPerDayPerProjectPerModel"))
    r = TMP / "runs" / "daily_quota"
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    dq = manifest_of(r)
    check(code == 1 and len(fake.calls) == 1 and dq["status"] == "failed"
          and "daily request quota" in dq["clips"][0]["error_reason"]
          and "GenerateRequestsPerDayPerProjectPerModel, limit 100" in dq["clips"][0]["error_reason"]
          and any("daily request quota" in w for w in dq["warnings"]),
          "a used-up daily quota (Gemini QuotaFailure ...PerDay...) stops the run after one request")
    billing = HTTPError(429)
    billing.body = {"message": "You exceeded your current quota", "type": "insufficient_quota",
                    "code": "insufficient_quota"}
    fake = FakeTTS(script=lambda n, t: billing)
    r = TMP / "runs" / "insufficient_quota"
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    check(code == 1 and len(fake.calls) == 1
          and "billing" in manifest_of(r)["clips"][0]["error_reason"],
          "OpenAI's insufficient_quota 429 (no credit left) stops the run after one request")
    per_minute = gemini_429("GenerateRequestsPerMinutePerProjectPerModel", delay="12.5s")
    long_wait = gemini_429("GenerateRequestsPerMinutePerProjectPerModel", delay="3600s")
    with_header = HTTPError(429)
    with_header.response = type("R", (), {"headers": {"retry-after": "7"}})()
    check(narrate_lab.server_wait(per_minute) == 12.5 and narrate_lab.server_wait(long_wait) == 60.0
          and narrate_lab.server_wait(with_header) == 7.0 and narrate_lab.server_wait(HTTPError(429)) is None
          and not narrate_lab._is_fatal(per_minute)
          and "asked to wait 12.5 s" in narrate_lab.describe_error(per_minute)["reason"],
          "a per-minute rate limit is retried after the wait the provider asks for (capped at 60 s)")
    fake = FakeTTS(script=lambda n, t: HTTPError(429))
    r = TMP / "runs" / "rate_limited_in_a_row"
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    rr = manifest_of(r)
    n_tries = narrate_lab.RATE_LIMITED_IN_A_ROW * benchmark._TRANSIENT_ATTEMPTS
    check(code == 1 and len(fake.calls) == n_tries and rr["status"] == "failed"
          and any("in a row failed on rate limits" in w for w in rr["warnings"])
          and rr["clips"][narrate_lab.RATE_LIMITED_IN_A_ROW]["error"].startswith("not attempted"),
          f"{narrate_lab.RATE_LIMITED_IN_A_ROW} clips in a row failing on rate limits stop the run "
          f"({n_tries} requests, not one per clip for the whole lab)")
    fake = FakeTTS(script=lambda n, t: HTTPError(503))
    r = TMP / "runs" / "transient_exhausted"
    narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    c0 = manifest_of(r)["clips"][0]
    check(c0["status"] == "failed" and c0["transient_retries"] == 2 and len(c0["transient_errors"]) == 2
          and c0["error_reason"] == "the provider was temporarily unavailable (HTTP 503)",
          "when retries run out, the retried errors and the final reason are both recorded")

    target_text = lab["modules"][1]["clips"][0]["narration"].strip()
    fake = FakeTTS(script=lambda n, t: ValueError("boom") if t == target_text else None)
    r = TMP / "runs" / "failed"
    (r).mkdir(parents=True)
    stale = r / rm["bundle"]["zip_file"]
    stale.write_bytes(b"stale")
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    fm = manifest_of(r)
    check(code == 1 and fm["status"] == "incomplete" and fm["counts"]["failed"] == 1
          and fm["counts"]["generated"] == len(refs) - 1,
          "one failing clip leaves the run incomplete; the other clips still narrate")
    failed_clip = next(c for c in fm["clips"] if c["status"] == "failed")
    check(failed_clip["error_reason"] == "unexpected error" and failed_clip["error"] == "ValueError: boom",
          "a failed clip records the error and a reason")
    check(fm["totals"]["calls_cost_unknown"] == 1
          and next(x for x in fm["calls"] if x["status"] == "error")["cost_usd"] is None
          and abs(fm["totals"]["total_cost_usd"] - (len(refs) - 1) * COST_PER_CLIP) < 1e-9,
          "a failure with no reply and no usage has unknown cost (null), counted, never guessed")
    check(not stale.exists() and fm["bundle"]["validation"] is None,
          "an incomplete run builds no zip and removes a stale one")
    fake = FakeTTS()
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    fm2 = manifest_of(r)
    check(code == 0 and fake.calls == [target_text] and fm2["status"] == "complete",
          "rerunning the same folder narrates only the failed clip and completes")
    fc = next(c for c in fm2["clips"] if c["spoken_text"] == target_text)
    check(len(fm2["calls"]) == len(refs) + 1 and [x["run"] for x in fm2["calls"]][-1] == 2
          and fc["calls"] == 2 and fc["calls_failed"] == 1
          and abs(fm2["totals"]["total_cost_usd"] - len(refs) * COST_PER_CLIP) < 1e-9,
          "a resume appends to the call log: the failed clip shows both runs' requests")

    for status in (401, 402, 403, 404):
        fake = FakeTTS(script=lambda n, t, s=status: HTTPError(s))
        r = TMP / "runs" / f"fatal_{status}"
        code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
        fm = manifest_of(r)
        check(code == 1 and len(fake.calls) == 1 and fm["status"] == "failed"
              and fm["clips"][1]["error"].startswith("not attempted"),
              f"HTTP {status} stops the run after one request")
    fake = FakeTTS(script=lambda n, t: tts.TTSFormatError("odd audio"))
    r = TMP / "runs" / "fatal_format"
    narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    check(len(fake.calls) == 1 and manifest_of(r)["status"] == "failed",
          "audio in an unexpected format stops the run (it would be billed for every clip)")

    long_lab = json.loads(SAMPLE_LAB.read_text(encoding="utf-8"))
    long_lab["modules"][0]["clips"][0]["narration"] = "word " * 1000
    long_path = fresh("long_input") / "long_json_lab_L4_x_output.json"
    long_path.write_text(json.dumps(long_lab), encoding="utf-8")
    fake = FakeTTS()
    r = TMP / "runs" / "long"
    narrate(settings_for(long_path), r, provider=fake, log=quiet)
    lm = manifest_of(r)
    check(lm["clips"][0]["status"] == "failed" and "never split" in lm["clips"][0]["error"]
          and len(fake.calls) == len(refs) - 1,
          "a clip over the provider's character limit fails without a request")

    fake = FakeTTS(served="gpt-4o-tts-other")
    r = TMP / "runs" / "served"
    narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    check(any("served by gpt-4o-tts-other" in w for w in manifest_of(r)["warnings"]),
          "a different served model is a warning")
    fake = FakeTTS(usage=None)
    r = TMP / "runs" / "nousage"
    narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    nm = manifest_of(r)
    check(nm["clips"][0]["cost_usd"] is None and "no token usage" in " ".join(nm["clips"][0]["warnings"])
          and nm["totals"]["calls_cost_unknown"] == len(refs) and nm["totals"]["total_cost_usd"] == 0,
          "missing usage gives no cost and a warning, not a guess")

    # ── Reruns keep valid audio, even after a stop or a crash ────────
    def edited_lab(tag, clip_index=1):
        """The sample lab with one clip's narration changed (module 1, clip clip_index)."""
        d = json.loads(SAMPLE_LAB.read_text(encoding="utf-8"))
        d["modules"][0]["clips"][clip_index]["narration"] += " (edited)"
        path = fresh(tag) / SAMPLE_LAB.name
        path.write_text(json.dumps(d), encoding="utf-8")
        return path, d["modules"][0]["clips"][clip_index]["narration"].strip()

    def full_run(tag):
        r = TMP / "runs" / tag
        narrate(settings_for(SAMPLE_LAB), r, provider=FakeTTS(), log=quiet)
        return r

    r = full_run("fatal_keeps")
    edited, edited_text = edited_lab("fatal_keeps_input")
    fake = FakeTTS(script=lambda n, t: HTTPError(402) if t == edited_text else None)
    code = narrate(settings_for(edited), r, provider=fake, log=quiet)
    fk = manifest_of(r)
    check(code == 1 and fk["status"] == "failed" and len(fake.calls) == 1
          and fk["counts"]["skipped"] == len(refs) - 1 and fk["counts"]["failed"] == 1,
          "after a stop-the-run error, unchanged clips are still skipped, not marked 'not attempted'")
    fake = FakeTTS()
    narrate(settings_for(edited), r, provider=fake, log=quiet)
    check(fake.calls == [edited_text] and manifest_of(r)["status"] == "complete",
          "the next resume narrates only the clip that failed")

    r = full_run("crash_keeps")
    edited, edited_text = edited_lab("crash_keeps_input")
    fake = FakeTTS(script=lambda n, t: KeyboardInterrupt() if t == edited_text else None)
    try:
        narrate(settings_for(edited), r, provider=fake, log=quiet)
        crashed = False
    except KeyboardInterrupt:
        crashed = True
    ck = manifest_of(r)
    later = [c for c in ck["clips"] if c["module"] > 1]
    check(crashed and later and all(c["status"] in ("generated", "skipped") and c["source_hash"]
                                    for c in later),
          "a run killed mid-way keeps the earlier records of the clips it never reached")
    fake = FakeTTS()
    narrate(settings_for(edited), r, provider=fake, log=quiet)
    check(fake.calls == [edited_text] and manifest_of(r)["status"] == "complete",
          "after a crash, the next resume narrates only the unfinished clip")

    # ── Names that differ only in case ───────────────────────────────
    case_lab = json.loads(SAMPLE_LAB.read_text(encoding="utf-8"))
    cm = [m["clips"] for m in case_lab["modules"]]
    cm[0][0]["audioClip"] = "Intro"
    cm[0][1]["audioClip"] = "intro"                        # other words -> must not share a file
    cm[1][0].update(audioClip="INTRO", narration=cm[0][0]["narration"])   # same words -> shared
    cm[1][1].update(audioClip=None, clipName="Recap")
    cm[2][0]["audioClip"] = "recap"                        # a generated "Recap" would clash
    case_plans = {(q.module, q.clip): q for q in plan_clips(case_lab)}
    check(case_plans[(1, 2)].name == "intro_2" and case_plans[(1, 2)].name_change == "deduplicated",
          "a name that differs from an earlier one only in case gets a _2 suffix (Windows/macOS safe)")
    check(case_plans[(2, 1)].action == "shared" and case_plans[(2, 1)].name == "Intro",
          "same words under a different-case name share the first file, under its exact name")
    check(case_plans[(2, 2)].name == "m02_Recap",
          "a generated name that matches another clip's audio name in any case gets the module prefix")
    lowered = [q.name.casefold() for q in case_plans.values() if q.name and q.action == "narrate"]
    check(len(lowered) == len(set(lowered)), "no two narrated files differ only in case")
    case_path = fresh("case_input") / "case_json_lab_L4_x_output.json"
    case_path.write_text(json.dumps(case_lab), encoding="utf-8")
    r = TMP / "runs" / "case"
    code = narrate(settings_for(case_path), r, provider=FakeTTS(), log=quiet)
    check(code == 0 and validate_bundle(r / manifest_of(r)["bundle"]["zip_file"]).ok,
          "the case-clash lab narrates into a valid zip")

    # ── No audio in a reply ──────────────────────────────────────────
    class EmptyFirst(FakeTTS):
        def __init__(self, empty_for):
            super().__init__()
            self.empty_for = empty_for

        def synthesize(self, text):
            syn = super().synthesize(text)
            if self.empty_for(len(self.calls) - 1, text):
                syn.pcm = b""
            return syn

    fake = EmptyFirst(lambda n, t: n == 0)
    r = TMP / "runs" / "empty_once"
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    c0 = manifest_of(r)["clips"][0]
    check(code == 0 and c0["status"] == "generated" and c0["transient_retries"] == 1
          and c0["audio_seconds"] > 0 and "no audio" in c0["transient_errors"][0]["reason"],
          "a reply with no audio is retried, not saved as an empty clip")
    mine = [x for x in manifest_of(r)["calls"] if x["clip"] == c0["audio_name"]]
    check(len(mine) == 2 and mine[0]["error_type"] == "TTSNoAudio"
          and abs(mine[0]["cost_usd"] - COST_PER_CLIP) < 1e-12
          and abs(c0["spend_usd"] - 2 * COST_PER_CLIP) < 1e-12,
          "a reply that came back without audio was still billed: its cost is counted")
    first_text = lab["modules"][0]["clips"][0]["narration"].strip()
    fake = EmptyFirst(lambda n, t: t == first_text)
    r = TMP / "runs" / "empty_always"
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    em2 = manifest_of(r)
    check(code == 1 and em2["status"] == "incomplete" and em2["clips"][0]["status"] == "failed"
          and em2["counts"]["generated"] == len(refs) - 1,
          "a clip that never gets audio fails on its own (resumable); the run doesn't stop")

    # ── Spoken form: the word check ──────────────────────────────────
    import spoken_form  # noqa: E402
    from spoken_form import check_spoken
    allowed = [
        ("when one strand is moving 5'→3', the other must be moving 3'→5'.",
         "when one strand is moving five prime to three prime, the other must be moving three prime to five prime."),
        ("The arrows show how vx = v cosθ and vy = v sinθ.",
         "The arrows show how v x equals v cos theta and v y equals v sin theta."),
        ("the domain angles are about 109.5 degrees.", "the domain angles are about one hundred nine point five degrees."),
        ("Methane, CH₄, has carbon with four bonding pairs.", "Methane, C H four, has carbon with four bonding pairs."),
        ("binding proteins, or SSBs, attach behind it.", "binding proteins, or S S Bs, attach behind it."),
        ("the completed strand reads 3'-TACGGTCA-5', running", "the completed strand reads three prime, T, A, C, G, G, T, C, A, five prime, running"),
        ("Mars orbits at 1.52 AU.", "Mars orbits at one point five two A U."),
        ("the patient's own right/left.", "the patient's own right or left."),
        ("Press NEXT to advance.", "Press next to advance."),
    ]
    check(all(not check_spoken(o, sp)[0] for o, sp in allowed),
          "the word check allows symbols, numbers, formulas, acronyms and sequences to be spelled out: "
          + str([o[:30] for o, sp in allowed if check_spoken(o, sp)[0]]))
    flagged_cases = [
        ("Helicase separates the strands at the fork.", "Helicase splits the strands at the fork.", "changed"),
        ("Helicase separates the strands at the fork.", "Helicase separates the strands at the fork. Great job!", "added"),
        ("Helicase separates the strands at the fork.", "Helicase separates strands at the fork.", "changed"),
        ("Helicase separates the strands at the fork.", "", "empty"),
    ]
    check(all(any(k in f for f in check_spoken(o, sp)[0]) for o, sp, k in flagged_cases),
          "the word check flags a reworded, added, dropped or missing word")
    check(check_spoken("It is here.", "This is here.")[0] and check_spoken("In a cell.", "Inside a cell.")[0],
          "a changed two-letter capitalized word (It, In) is flagged, not taken for an acronym")
    check(check_spoken("Add a cell and a membrane.", "Add one cell and a membrane.")[0],
          "a dropped one-letter word is flagged even when the same letter appears nearby")
    check(not check_spoken("Xenon, Xe, has eight.", "Xenon, X e, has eight.")[0]
          and not check_spoken("The DNA's backbone.", "The D N A's backbone.")[0],
          "element symbols and possessive acronyms may still be spelled out")
    ch = check_spoken("Mars orbits at 1.52 AU.", "Mars orbits at one point five two A U.")[1]
    check(ch == [{"original": "1.52 AU", "spoken": "one point five two A U"}],
          "each change is recorded as original -> spoken")

    # ── Spoken form: LLM mode end to end with a fake model ──────────
    import re as _re

    class FakeSpoken:
        """Spells out DNA/SSBs, rewords one clip, and can leave a clip out or fail."""
        def __init__(self, reword=None, omit=None, fail=False):
            self.calls, self.reword, self.omit, self.fail = [], reword, omit, fail

        def __call__(self, prompt):
            clips = _re.findall(r"\[([^\]\n]+)\]\n(.+?)(?=\n\n\[|\Z)", prompt.split("Clips:\n\n", 1)[1], _re.S)
            self.calls.append([n for n, _ in clips])
            if self.fail:
                raise HTTPError(503)
            out = []
            for name, text in clips:
                if name == self.omit:
                    continue
                spoken = text.replace("DNA", "D N A").replace("SSBs", "S S Bs")
                if name == self.reword:
                    spoken = spoken.replace("two", "a pair of", 1) if "two" in spoken else spoken + " Nice work."
                out.append(spoken_form.ClipSpokenForm(audio_name=name, spoken_text=spoken))
            return spoken_form.LabSpokenForms(clips=out), {"input_tokens": 900, "output_tokens": 800,
                                                           "cost_usd": 0.0005, "served_models": ["fake-llm"]}

    def llm_settings(path=SAMPLE_LAB, model="claude-haiku-5.5"):
        st = settings_for(path)
        st.spoken_form, st.spoken_form_model = "llm", model
        return st

    names_in_order = [q.name for q in plan_clips(lab) if q.action == "narrate"]
    first, second, third = names_in_order[:3]
    fake_sp = FakeSpoken(reword=second, omit=third)
    tts_fake = FakeTTS()
    r = TMP / "runs" / "llm"
    code = narrate(llm_settings(), r, provider=tts_fake, spoken_call=fake_sp, log=quiet)
    lm_ = manifest_of(r)
    by_name = {c["audio_name"]: c for c in lm_["clips"]}
    check(code == 0 and lm_["status"] == "complete" and len(fake_sp.calls) == 1
          and len(fake_sp.calls[0]) == len(refs),
          "LLM mode: one call carries every narrated clip, and the run completes")
    check(by_name[first]["spoken_text"] == lab["modules"][0]["clips"][0]["narration"].strip().replace("DNA", "D N A")
          and by_name[first]["spoken_form"]["source"] == "llm" and not by_name[first]["spoken_form"]["flags"]
          and tts_fake.calls[0] == by_name[first]["spoken_text"],
          "a clip's spoken text comes back by audio name and is what the voice service receives")
    sent_second = fake_sp.calls[0].index(second)
    second_display = by_name[second]["display_text"].strip()
    check(by_name[second]["spoken_form"]["flags"] and by_name[second]["spoken_form"]["source"] == "llm"
          and by_name[second]["spoken_text"] != second_display.replace("DNA", "D N A").replace("SSBs", "S S Bs")
          and tts_fake.calls[sent_second] == by_name[second]["spoken_text"],
          "a reworded clip is flagged, not rejected, and the LLM's reworded text is what gets narrated")
    check(by_name[third]["spoken_form"]["source"] == "fallback"
          and by_name[third]["spoken_text"] == by_name[third]["display_text"].strip()
          and "didn't return" in by_name[third]["spoken_form"]["flags"][0],
          "a clip the model left out falls back to its narration as written, flagged")
    check(by_name[first]["display_text"] != by_name[first]["spoken_text"]
          and json.loads((r / rm["bundle"]["folder"] / bundle_names(lab["labId"]).json_file).read_text(encoding="utf-8"))
              ["modules"][0]["clips"][0]["narration"] == lab["modules"][0]["clips"][0]["narration"],
          "the bundled lab keeps the display narration unchanged")
    sfm = lm_["spoken_form"]
    check(sfm["mode"] == "llm" and sfm["model"] == "claude-haiku-5.5" and sfm["calls"] == 1
          and sfm["clips_sent"] == len(refs) and sfm["input_tokens"] == 900 and sfm["cost_usd"] == 0.0005
          and lm_["counts"]["spoken_form_flagged"] == 2 and lm_["totals"]["spoken_form_cost_usd"] == 0.0005
          and abs(lm_["totals"]["total_cost_usd"] - (lm_["totals"]["tts_cost_usd"] + 0.0005)) < 1e-12
          and [x["step"] for x in lm_["calls"]].count("spoken_form") == 1
          and lm_["calls"][0]["step"] == "spoken_form" and lm_["calls"][0]["clip"] is None,
          "the manifest records the spoken-form model, tokens, cost, flagged count and total cost")
    cache = json.loads((r / spoken_form.CACHE_FILE).read_text(encoding="utf-8"))["clips"]
    check(set(cache) == set(refs) and cache[second]["flags"] and cache[first]["prompt_version"] == spoken_form.PROMPT_VERSION,
          "spoken forms are cached in spoken_forms.json with their flags")

    fake_sp2, tts_fake = FakeSpoken(), FakeTTS()
    narrate(llm_settings(), r, provider=tts_fake, spoken_call=fake_sp2, log=quiet)
    lm2 = manifest_of(r)
    third_rec = {c["audio_name"]: c for c in lm2["clips"]}[third]
    check(fake_sp2.calls == [[third]] and third_rec["spoken_form"]["source"] == "llm"
          and lm2["spoken_form"]["clips_cached"] == len(refs) - 1
          and abs(lm2["totals"]["spoken_form_cost_usd"] - 0.0010) < 1e-12,
          "a rerun reuses every cached spoken form; only the clip that fell back is sent again"
          " (its audio is redone only if the model's text differs)")
    fake_sp2b, tts_fake = FakeSpoken(), FakeTTS()
    narrate(llm_settings(), r, provider=tts_fake, spoken_call=fake_sp2b, log=quiet)
    check(not fake_sp2b.calls and not tts_fake.calls and manifest_of(r)["counts"]["skipped"] == len(refs),
          "once every clip has the model's text, a rerun makes no LLM call and redoes no audio")

    data = json.loads((r / spoken_form.CACHE_FILE).read_text(encoding="utf-8"))
    data["clips"][first]["spoken_text"] = "Reviewer's fixed wording for D N A."
    (r / spoken_form.CACHE_FILE).write_text(json.dumps(data), encoding="utf-8")
    fake_sp3, tts_fake = FakeSpoken(), FakeTTS()
    narrate(llm_settings(), r, provider=tts_fake, spoken_call=fake_sp3, log=quiet)
    c_first = {c["audio_name"]: c for c in manifest_of(r)["clips"]}[first]
    check(not fake_sp3.calls and tts_fake.calls == ["Reviewer's fixed wording for D N A."]
          and c_first["spoken_form"]["source"] == "edited" and c_first["spoken_form"]["flags"],
          "a reviewer's edit in spoken_forms.json is kept and only that clip's audio is redone (still checked)")

    edited, edited_text = edited_lab("llm_edit_input")
    fake_sp4, tts_fake = FakeSpoken(), FakeTTS()
    narrate(llm_settings(edited), r, provider=tts_fake, spoken_call=fake_sp4, log=quiet)
    check(len(fake_sp4.calls) == 1 and len(fake_sp4.calls[0]) == 1 and len(tts_fake.calls) == 1,
          "after a narration edit, only the changed clip is sent to the LLM and re-narrated")

    fake_sp5 = FakeSpoken()
    narrate(llm_settings(model="gpt-6-luna"), TMP / "runs" / "llm", provider=FakeTTS(),
            spoken_call=fake_sp5, log=quiet)
    check(len(fake_sp5.calls) == 1 and len(fake_sp5.calls[0]) == len(refs),
          "switching the spoken-form model refreshes every cached spoken form")

    r = TMP / "runs" / "llm_fail"
    tts_fake = FakeTTS()
    code = narrate(llm_settings(), r, provider=tts_fake, spoken_call=FakeSpoken(fail=True), log=quiet)
    lf = manifest_of(r)
    check(code == 0 and lf["status"] == "complete" and "HTTPError" in (lf["spoken_form"]["error"] or "")
          and all(c["spoken_form"]["source"] == "fallback" for c in lf["clips"] if c["spoken_form"])
          and tts_fake.calls[0] == lab["modules"][0]["clips"][0]["narration"].strip(),
          "if the spoken-form call fails, every clip falls back to its narration, flagged, and the run completes")

    # A failed call's fallback is retried next time, not cached as the model's answer.
    r = TMP / "runs" / "llm_fail_then_ok"
    narrate(llm_settings(), r, provider=FakeTTS(), spoken_call=FakeSpoken(fail=True), log=quiet)
    retry_sp = FakeSpoken()
    narrate(llm_settings(), r, provider=FakeTTS(), spoken_call=retry_sp, log=quiet)
    lr = manifest_of(r)
    check(len(retry_sp.calls) == 1 and len(retry_sp.calls[0]) == len(refs)
          and all(c["spoken_form"]["source"] == "llm" for c in lr["clips"] if c["spoken_form"]),
          "after a failed spoken-form call, the next run sends every clip to the model again")
    r = TMP / "runs" / "llm_fallback_edit"
    narrate(llm_settings(), r, provider=FakeTTS(), spoken_call=FakeSpoken(omit=first), log=quiet)
    data = json.loads((r / spoken_form.CACHE_FILE).read_text(encoding="utf-8"))
    data["clips"][first]["spoken_text"] = "A reviewer's text."
    (r / spoken_form.CACHE_FILE).write_text(json.dumps(data), encoding="utf-8")
    edit_sp = FakeSpoken()
    narrate(llm_settings(), r, provider=FakeTTS(), spoken_call=edit_sp, log=quiet)
    c_first = {c["audio_name"]: c for c in manifest_of(r)["clips"]}[first]
    check(not edit_sp.calls and c_first["spoken_text"] == "A reviewer's text."
          and c_first["spoken_form"]["source"] == "edited",
          "a reviewer's edit of a fallback clip is kept and not sent to the model again")

    class OddReplies(FakeSpoken):
        """Echoes tags with brackets, spaces or other case, and returns one empty text."""
        def __call__(self, prompt):
            forms, usage = super().__call__(prompt)
            for i, c in enumerate(forms.clips):
                # clip 0: [bracketed], 1: exact tag but empty text, 2: spaces, 3: upper case
                c.audio_name = [f"[{c.audio_name}]", c.audio_name, f"  {c.audio_name} ",
                                c.audio_name.upper()][i % 4]
                if i == 1:
                    c.spoken_text = "   "
            return forms, usage

    r = TMP / "runs" / "llm_odd"
    tts_fake = FakeTTS()
    narrate(llm_settings(), r, provider=tts_fake, spoken_call=OddReplies(), log=quiet)
    lo = {c["audio_name"]: c for c in manifest_of(r)["clips"]}
    fourth = names_in_order[3]
    check(all(lo[n]["spoken_form"]["source"] == "llm" for n in (first, third, fourth)),
          "tags echoed with brackets, extra spaces or other case are still matched to their clips")
    check(lo[second]["spoken_form"]["source"] == "fallback"
          and lo[second]["spoken_text"] == lo[second]["display_text"].strip()
          and "" not in tts_fake.calls and lo[second]["spoken_form"]["flags"],
          "an empty spoken text falls back to the narration (flagged) instead of reaching the voice")

    r = TMP / "runs" / "none_mode"
    narrate(settings_for(SAMPLE_LAB), r, provider=FakeTTS(), log=quiet)
    nm_ = manifest_of(r)
    check(nm_["spoken_form"]["mode"] == "none" and nm_["spoken_form"]["calls"] == 0
          and not (r / spoken_form.CACHE_FILE).exists()
          and all(c["spoken_form"]["source"] == "none" for c in nm_["clips"] if c["spoken_form"]),
          "--spoken-form none makes no call and writes no cache")

    # ── Unsupported inputs and CLI flags ─────────────────────────────
    l1 = next(iter(sorted(glob(str(OUTPUTS / "*json_lab_L1_*_output.json")))), None)
    try:
        narrate_lab.load_lab(Path(l1))
        ok = False
    except narrate_lab.InputError as exc:
        ok = "not a full json_lab Lab" in str(exc)
    check(l1 is not None and ok, "an L1 outline is rejected with a clear message")

    def cli(*argv):
        out, errs = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errs):
            code = narrate_lab.main(list(argv))
        return code, out.getvalue() + errs.getvalue()

    code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--dry-run")
    check(code == 0 and "No requests sent" in text and "gemini-3.8-flash-tts / Charon" in text,
          "--dry-run prints the plan with the frozen Gemini defaults and sends nothing")
    code, text = cli(str(l1), "--provider", "openai", "--dry-run")
    check(code == 2 and "not a full json_lab Lab" in text, "an unsupported input exits 2")
    code, text = cli("--resume", str(real_run), "--provider", "gemini")
    check(code == 2 and "--provider" in text, "--resume refuses settings that the manifest owns")
    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY",)}
    try:
        import dotenv
        real_load = dotenv.load_dotenv
        dotenv.load_dotenv = lambda *a, **k: False      # ignore any local .env here
        code, text = cli(str(SAMPLE_LAB), "--provider", "openai")
    finally:
        dotenv.load_dotenv = real_load
        os.environ.update({k: v for k, v in saved.items() if v is not None})
    check(code == 2 and "OPENAI_API_KEY" in text, "a missing API key exits 2 before any request")
    code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--spoken-form", "llm", "--dry-run")
    check(code == 0 and "Spoken form: llm with claude-haiku-5.5" in text,
          "--spoken-form llm --dry-run names the default model and sends nothing")
    code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--spoken-form-model", "gpt-6-luna", "--dry-run")
    check(code == 2 and "only applies with --spoken-form llm" in text,
          "--spoken-form-model without --spoken-form llm is an error")
    code, text = cli("--resume", str(TMP / "runs" / "llm"), "--spoken-form-model", "gpt-6-luna")
    check(code == 2 and "--spoken-form-model" in text, "--resume refuses a spoken-form model override")
    saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")}
    try:
        try:
            narrate_lab.check_spoken_form_environment("claude-haiku-5.5")
            ok = False
        except narrate_lab.InputError as exc:
            ok = "ANTHROPIC_API_KEY" in str(exc)
    finally:
        os.environ.update({k: v for k, v in saved.items() if v is not None})
    check(ok, "a missing key for the spoken-form model is caught before any request")

    # ── Spoken form: rules mode ──────────────────────────────────────
    def rules_settings(path=SAMPLE_LAB, glossary=None):
        st = settings_for(path)
        st.spoken_form, st.glossary = "rules", glossary
        return st

    r = TMP / "runs" / "rules"
    tts_fake = FakeTTS()
    code = narrate(rules_settings(), r, provider=tts_fake, log=quiet)
    rl = manifest_of(r)
    sfr = rl["spoken_form"]
    by_name = {c["audio_name"]: c for c in rl["clips"]}
    strand = next(c for c in rl["clips"] if c["display_text"] and "5' to 3'" in c["display_text"])
    check(code == 0 and rl["status"] == "complete" and sfr["mode"] == "rules" and sfr["calls"] == 0
          and sfr["glossary_version"] and len(sfr["glossary_files"]) == 2
          and sfr["glossary_files"][1].endswith("topics/dna_structure_and_replication.yaml")
          and not [x for x in rl["calls"] if x["step"] == "spoken_form"]
          and rl["totals"]["spoken_form_cost_usd"] == 0,
          "rules mode: no LLM call, and the manifest records the glossary version and its layers")
    check("5 prime to 3 prime" in strand["spoken_text"] and strand["spoken_text"] in tts_fake.calls
          and strand["spoken_form"]["source"] == "rules"
          and any(ch["rule"] == "prime" for ch in strand["spoken_form"]["changes"])
          and strand["display_text"] != strand["spoken_text"],
          "rules mode: the voice receives the rules' text, and each clip records its rule changes")
    check(isinstance(sfr["unresolved_terms"], dict) and rl["counts"]["spoken_form_flagged"]
          == sum(1 for c in rl["clips"] if (c.get("spoken_form") or {}).get("flags")),
          "rules mode: undecided terms are listed for the lab and flagged per clip")

    with_ssb = [c["audio_name"] for c in rl["clips"] if c["display_text"] and "SSB" in c["display_text"]]
    (r / spoken_rules.RUN_GLOSSARY).write_text("terms:\n  SSB: single-strand binding protein\n",
                                               encoding="utf-8")
    tts_fake = FakeTTS()
    narrate(rules_settings(), r, provider=tts_fake, log=quiet)
    rl2 = manifest_of(r)
    ssb_clip = {c["audio_name"]: c for c in rl2["clips"]}[with_ssb[0]]
    check(with_ssb and len(tts_fake.calls) == len(with_ssb)
          and "single-strand binding proteins" in ssb_clip["spoken_text"]
          and rl2["spoken_form"]["glossary_version"] != sfr["glossary_version"]
          and rl2["spoken_form"]["glossary_files"][-1].endswith("glossary.yaml"),
          f"a term added in the run folder's glossary.yaml redoes only the {len(with_ssb)} clip(s) it changes")

    gdir = fresh("cli_glossary")
    good_g, bad_g = gdir / "extra.yaml", gdir / "bad.yaml"
    good_g.write_text("terms:\n  SSB: S S B\n", encoding="utf-8")
    bad_g.write_text("pins:\n  number_prime: furlongs\n", encoding="utf-8")
    r = TMP / "runs" / "rules_extra"
    narrate(rules_settings(glossary=[str(good_g)]), r, provider=FakeTTS(), log=quiet)
    rx_ = manifest_of(r)
    check(rx_["spoken_form"]["glossary_extra"] == [str(good_g)] and len(rx_["spoken_form"]["glossary_files"]) == 3
          and any("S S Bs" in (c["spoken_text"] or "") for c in rx_["clips"]),
          "--glossary FILE adds a layer, and the manifest keeps it for --resume")
    code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--spoken-form", "rules", "--dry-run")
    check(code == 0 and "Spoken form: rules, glossary" in text and "clips changed" in text,
          "--spoken-form rules --dry-run reports what the rules would change, sending nothing")
    code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--spoken-form", "none",
                     "--glossary", str(good_g), "--dry-run")
    check(code == 2 and "only applies with --spoken-form rules" in text, "--glossary without rules mode is an error")
    code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--dry-run")
    check(code == 0 and "Spoken form: rules, glossary" in text and narrate_lab.DEFAULT_SPOKEN_FORM == "rules"
          and spoken_form.DEFAULT_MODEL == "claude-haiku-5.5",
          "the default spoken form is rules, and the default LLM model is Claude Haiku 5.5")
    code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--spoken-form", "rules",
                     "--glossary", str(gdir / "missing.yaml"), "--dry-run")
    check(code == 2 and "no glossary file" in text, "a missing --glossary file exits 2")
    code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--spoken-form", "rules",
                     "--glossary", str(bad_g), "--dry-run")
    check(code == 2 and "furlongs" in text, "a bad glossary exits 2 with the reason, before any request")
    code, text = cli("--resume", str(TMP / "runs" / "rules"), "--glossary", str(good_g))
    check(code == 2 and "--glossary" in text,
          "--resume refuses --glossary (the run folder's glossary.yaml is the way to add terms)")

    # ── Text-only runs and the mode comparison (#94 measurement) ─────
    r = TMP / "runs" / "measure" / "rules_text"
    code = narrate(rules_settings(), r, provider=None, text_only=True, log=quiet)
    tm_ = manifest_of(r)
    check(code == 0 and tm_["status"] == "text_only" and tm_["counts"]["text_only"] == len(refs)
          and not list(r.rglob("*.wav")) and not list(r.glob("*.zip"))
          and all(c["spoken_text"] for c in tm_["clips"] if c["status"] == "text_only")
          and tm_["runs"][0]["text_only"] and tm_["runs"][0]["spoken_form_seconds"] is not None
          and not tm_["calls"],
          "--text-only records every clip's spoken text with no voice service, audio or zip")
    tts_fake = FakeTTS()
    code = narrate(rules_settings(), r, provider=tts_fake, log=quiet)
    check(code == 0 and manifest_of(r)["status"] == "complete" and len(tts_fake.calls) == len(refs),
          "running a text-only folder again narrates it")

    r_llm = TMP / "runs" / "measure" / "llm_text"
    sp_fake = FakeSpoken()
    narrate(llm_settings(), r_llm, provider=None, spoken_call=sp_fake, text_only=True, log=quiet)
    tts_fake = FakeTTS()
    narrate(llm_settings(), r_llm, provider=tts_fake, spoken_call=sp_fake, log=quiet)
    lt = manifest_of(r_llm)
    check(len(sp_fake.calls) == 1 and len(tts_fake.calls) == len(refs) and lt["status"] == "complete"
          and [x["run"] for x in lt["calls"] if x["step"] == "spoken_form"] == [1],
          "narrating a text-only LLM folder reuses its spoken forms: the model is paid once")
    narrate(settings_for(SAMPLE_LAB), TMP / "runs" / "measure" / "none_text", provider=None,
            text_only=True, log=quiet)

    import compare_runs  # noqa: E402
    out_md = TMP / "runs" / "measure_report" / "report.md"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = compare_runs.main([str(TMP / "runs" / "measure"), "--out", str(out_md)])
    md = out_md.read_text(encoding="utf-8") if out_md.exists() else ""
    rows = list(csv.DictReader(open(out_md.with_suffix(".csv"), encoding="utf-8"))) if out_md.exists() else []
    check(code == 0 and {r_["mode"] for r_ in rows} == {"none", "rules", "llm claude-haiku-5.5"}
          and "| rules | 1 |" in md and "| llm claude-haiku-5.5 | 1 |" in md and "$0.0005" in md,
          "compare_runs.py tabulates each mode's clips changed, flags, spoken-form cost and time")
    check("## Clips whose spoken text differs" in md and "**5 prime" in md and "🔊" in md
          and "*none*" in md and "*Written:*" in md,
          "compare_runs.py lists each clip whose spoken text differs, with every mode's text and changes in bold")
    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")}
    real_root = narrate_lab.NARRATION_ROOT
    narrate_lab.NARRATION_ROOT = TMP / "cli_root"
    try:
        import dotenv
        real_load = dotenv.load_dotenv
        dotenv.load_dotenv = lambda *a, **k: False
        code, text = cli(str(SAMPLE_LAB), "--provider", "gemini", "--spoken-form", "rules",
                         "--text-only", "--run-group", "g")
    finally:
        dotenv.load_dotenv = real_load
        narrate_lab.NARRATION_ROOT = real_root
        os.environ.update({k: v for k, v in saved.items() if v is not None})
    made = [p.name for p in (TMP / "cli_root" / "g").glob("*")] if (TMP / "cli_root" / "g").exists() else []
    check(code == 0 and len(made) == 1 and "_gemini_rules_" in made[0],
          "--text-only needs no voice-service key, and the run folder is named after the mode")

# ══ Part 3: real clients against local mock servers ═══════════════════

if HAVE_NARRATION:
    print("\n-- Part 3: OpenAI and Gemini clients against mock servers")
    import base64
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Mock:
        """A local threaded HTTP server: ``handler(path, body_json) -> (status, headers,
        data)``, where data is bytes, or a list of ``(delay_seconds, bytes)`` pieces sent
        one by one (to stall or trickle). A handler may also sleep before answering."""
        def __init__(self, handler):
            self.requests = []
            mock = self

            class H(BaseHTTPRequestHandler):
                def do_POST(self):
                    n = int(self.headers.get("content-length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    mock.requests.append((self.path, dict(self.headers), body))
                    status, headers, data = handler(self.path, body)
                    try:
                        self.send_response(status)
                        for k, v in headers.items():
                            self.send_header(k, v)
                        if isinstance(data, (bytes, bytearray)):
                            self.send_header("content-length", str(len(data)))
                            self.end_headers()
                            self.wfile.write(data)
                            return
                        self.send_header("connection", "close")   # body ends at close
                        self.end_headers()
                        for delay, piece in data:
                            time.sleep(delay)
                            self.wfile.write(piece)
                            self.wfile.flush()
                    except ConnectionError:   # broken pipe, reset, or Windows' "aborted"
                        pass   # the client gave up (time limit): expected in these tests

                def log_message(self, *a):
                    pass

            self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
            self.server.daemon_threads = True
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.url = f"http://127.0.0.1:{self.server.server_port}"

        def close(self):
            self.server.shutdown()

    class _SkipGemini(Exception):
        pass

    half = len(PCM) // 2 // 2 * 2

    def sse(*events):
        return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)

    openai_state = {"fail": []}

    def openai_handler(path, body):
        if openai_state["fail"]:
            code = openai_state["fail"].pop(0)
            return code, {"content-type": "application/json"}, json.dumps(
                {"error": {"message": f"mock {code}", "type": "x"}}).encode()
        return 200, {"content-type": "text/event-stream", "openai-model": body.get("model", ""),
                     "x-request-id": "req_mock123"}, sse(
            {"type": "speech.audio.delta", "audio": base64.b64encode(PCM[:half]).decode()},
            {"type": "speech.audio.delta", "audio": base64.b64encode(PCM[half:]).decode()},
            {"type": "speech.audio.done", "usage": {"input_tokens": 14, "output_tokens": 101,
                                                    "total_tokens": 115}},
        )

    om = Mock(openai_handler)
    try:
        prov = tts.make_provider("openai", api_key="sk-test", base_url=om.url + "/v1")
        syn = prov.synthesize("Hello there.")
        path, headers, req = om.requests[-1]
        check(path == "/v1/audio/speech" and req == {
                  "model": "gpt-4o-mini-tts-2025-12-15", "voice": "marin", "input": "Hello there.",
                  "instructions": tts.STYLE_INSTRUCTION, "response_format": "pcm",
                  "stream_format": "sse"},
              "OpenAI request: pinned snapshot, marin, style as instructions, PCM over SSE")
        check(syn.pcm == PCM and (syn.input_tokens, syn.output_tokens) == (14, 101)
              and syn.served_model == "gpt-4o-mini-tts-2025-12-15" and syn.request_id == "req_mock123",
              "OpenAI SSE: audio deltas joined, usage from speech.audio.done, served model and request id from headers")

        openai_state["fail"] = [503]
        r = TMP / "runs" / "openai_mock"
        n_before = len(om.requests)
        code = narrate(settings_for(SAMPLE_LAB), r, provider=prov, log=quiet)
        om_m = manifest_of(r)
        check(code == 0 and om_m["status"] == "complete" and om_m["totals"]["transient_retries"] == 1
              and len(om.requests) - n_before == len(refs) + 1,
              "end to end over HTTP: a 503 is retried once by our retryer (the SDK makes no retries)")
        check(abs(om_m["totals"]["kept_audio_cost_usd"] - len(refs) * (14 * 0.60 + 101 * 12.00) / 1e6) < 1e-9
              and om_m["totals"]["total_cost_usd"] == om_m["totals"]["kept_audio_cost_usd"]
              and om_m["totals"]["calls"] == len(refs) + 1 and om_m["totals"]["calls_failed"] == 1
              and all(x["request_id"] == "req_mock123" for x in om_m["calls"] if x["status"] == "ok"),
              "OpenAI cost from the reported usage; the 503 is a logged request that cost $0")

        openai_state["fail"] = [401]
        r = TMP / "runs" / "openai_401"
        n_before = len(om.requests)
        code = narrate(settings_for(SAMPLE_LAB), r, provider=prov, log=quiet)
        check(code == 1 and len(om.requests) - n_before == 1 and manifest_of(r)["status"] == "failed",
              "an OpenAI 401 stops the run after one request")
        check(syn.first_audio_seconds is not None and syn.max_gap_seconds is not None,
              "OpenAI: time to first audio and the longest gap between pieces are measured")
        check(isinstance(prov.client.timeout, object) and prov.time_limit == tts.REQUEST_TIME_LIMIT == 90.0,
              "the default per-request time limit is 90 s")
    finally:
        om.close()

    # ── The per-request time limit (short limits so the checks run fast) ─
    def delta(chunk):
        return b"data: " + json.dumps({"type": "speech.audio.delta",
                                       "audio": base64.b64encode(chunk).decode()}).encode() + b"\n\n"
    done = b"data: " + json.dumps({"type": "speech.audio.done", "usage": {
        "input_tokens": 14, "output_tokens": 101, "total_tokens": 115}}).encode() + b"\n\n"
    stall = {"mode": None, "count": 0}

    def stalling_handler(path, body):
        stall["count"] += 1
        # The first request stalls; with "every", each attempt for one clip's text does.
        mode = stall["mode"] if (stall["count"] == 1 or (
            stall.get("every") and body.get("input") == stall.get("text"))) else None
        head = {"content-type": "text/event-stream", "x-request-id": f"req_{stall['count']}"}
        if mode == "before_headers":
            time.sleep(3)
            return 200, head, delta(PCM) + done
        if mode == "silent":      # some audio, then nothing
            return 200, head, [(0, delta(PCM[:half])), (5, delta(PCM[half:]) + done)]
        if mode == "trickle":     # keeps sending, slowly, past the limit
            return 200, head, [(0.25, delta(PCM[:2])) for _ in range(16)] + [(0, done)]
        return 200, head, [(0, delta(PCM[:half])), (0.05, delta(PCM[half:])), (0, done)]

    sm = Mock(stalling_handler)
    try:
        fast = tts.make_provider("openai", api_key="sk-test", base_url=sm.url + "/v1", time_limit=1.0)
        for mode in ("silent", "trickle"):
            stall.update(mode=mode, count=0)
            t0 = time.monotonic()
            try:
                fast.synthesize("x")
                outcome = "no error"
            except tts.TTSTimeout as exc:
                outcome = str(exc)
            took = time.monotonic() - t0
            check(outcome.startswith("no complete reply within 1 s") and took < 2.5,
                  f"a stream that goes {mode} past the limit is abandoned at the limit "
                  f"({took:.1f} s; {outcome})")
        # Windows' coarse timers can fire the watchdog after httpx's read timeout (the
        # same limit, counted from the last chunk); that must still be a TTSTimeout.
        real_timer = threading.Timer

        class LateTimer(real_timer):
            def __init__(self, interval, function, *a, **kw):
                super().__init__(interval + 0.3, function, *a, **kw)
        stall.update(mode="silent", count=0)
        threading.Timer = LateTimer
        try:
            fast.synthesize("x")
            outcome = "no error"
        except Exception as exc:  # noqa: BLE001
            outcome = type(exc).__name__
        finally:
            threading.Timer = real_timer
        check(outcome == "TTSTimeout",
              f"a stall caught by httpx's read timeout before the watchdog is still the time limit ({outcome})")
        stall.update(mode="before_headers", count=0)
        t0 = time.monotonic()
        try:
            fast.synthesize("x")
            outcome = None
        except Exception as exc:  # noqa: BLE001
            outcome = exc
        took = time.monotonic() - t0
        check(outcome is not None and "Timeout" in type(outcome).__name__ and took < 2.5,
              f"a reply that doesn't start within the limit times out ({type(outcome).__name__}, {took:.1f} s)")

        stall.update(mode="silent", count=0)
        r = TMP / "runs" / "openai_time_limit"
        code = narrate(settings_for(SAMPLE_LAB), r, provider=fast, log=quiet)
        tl = manifest_of(r)
        c0 = tl["clips"][0]
        check(code == 0 and tl["status"] == "complete" and c0["transient_retries"] == 1
              and c0["transient_errors"][0]["type"] == "TTSTimeout"
              and "time limit" in c0["transient_errors"][0]["reason"]
              and tl["voice_service"]["request_time_limit_seconds"] == 1.0,
              "end to end: a stalled request is abandoned, retried and recorded with its reason")
        check(c0["first_audio_seconds"] is not None and c0["max_gap_seconds"] is not None
              and c0["request_id"] == "req_2",
              "the manifest records time to first audio, the longest gap and the request id")
        stall.update(mode="trickle", count=0, every=True,
                     text=lab["modules"][0]["clips"][0]["narration"].strip())
        r = TMP / "runs" / "openai_time_limit_exhausted"
        code = narrate(settings_for(SAMPLE_LAB), r, provider=fast, log=quiet)
        c0 = manifest_of(r)["clips"][0]
        check(code == 1 and c0["status"] == "failed" and c0["transient_retries"] == 2
              and "time limit" in c0["error_reason"],
              "a request that hits the limit on every attempt fails after 3 attempts (resumable)")
        stall.pop("every")
    finally:
        sm.close()

    # A watchdog that fires in the instant between the last line and cancel(): the
    # complete reply must be kept, not thrown away as a timeout.
    class LateTimer:
        def __init__(self, interval, fn):
            self.fn = fn
            self.daemon = True

        def start(self):
            pass

        def cancel(self):
            self.fn()

    rm_ = Mock(openai_handler)
    real_timer = tts.threading.Timer
    try:
        tts.threading.Timer = LateTimer
        racy = tts.make_provider("openai", api_key="sk-test", base_url=rm_.url + "/v1")
        try:
            got = racy.synthesize("x")
            race_ok = got.pcm == PCM
        except tts.TTSTimeout:
            race_ok = False
        check(race_ok, "a reply that completed as the time limit hit is kept, not retried and re-billed")
    finally:
        tts.threading.Timer = real_timer
        rm_.close()

    gem_state = {"mime": "audio/L16;codec=pcm;rate=24000", "data": PCM, "fail": []}

    def gemini_handler(path, body):
        if gem_state.get("no_audio"):
            gem_state["no_audio"] -= 1
            return 200, {"content-type": "application/json"}, json.dumps({
                "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "OTHER"}],
                "usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 12},
                "modelVersion": "gemini-3.8-flash-tts"}).encode()
        if gem_state["fail"]:
            code = gem_state["fail"].pop(0)
            code, details = code if isinstance(code, tuple) else (code, [])
            return code, {"content-type": "application/json"}, json.dumps(
                {"error": {"code": code, "message": f"mock {code}", "status": "X",
                           "details": details}}).encode()
        return 200, {"content-type": "application/json"}, json.dumps({
            "candidates": [{"content": {"role": "model", "parts": [{"inlineData": {
                "mimeType": gem_state["mime"], "data": base64.b64encode(gem_state["data"]).decode()}}]},
                "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 125, "totalTokenCount": 137},
            "modelVersion": "gemini-3.8-flash-tts",
            "responseId": "resp_mock456",
        }).encode()

    import dataclasses
    future = dataclasses.replace(tts.PROVIDERS["gemini"], min_sdk_version="999.0.0")
    problem = tts.sdk_problem(future)
    check(problem is not None and 'pip install -U "google-genai>=999.0.0"' in problem,
          "an SDK older than a provider's minimum is reported with the upgrade command")
    try:
        narrate_lab.check_environment(future)
        ok = False
    except narrate_lab.InputError as exc:
        ok = "too old" in str(exc)
    check(ok, "the CLI's environment check stops on an SDK that is too old")

    gemini_sdk_problem = tts.sdk_problem(tts.PROVIDERS["gemini"])
    check(gemini_sdk_problem is None,
          f"google-genai is new enough for Gemini 3.8 TTS: {gemini_sdk_problem or 'ok'}")
    gm = Mock(gemini_handler)
    try:
        if gemini_sdk_problem:
            raise _SkipGemini()
        prov = tts.make_provider("gemini", api_key="g-test", base_url=gm.url)
        syn = prov.synthesize("Hello there.")
        path, headers, req = gm.requests[-1]
        part = req["contents"][0]["parts"][0]
        gen = req.get("generationConfig", {})
        check(path.endswith("/models/gemini-3.8-flash-tts:generateContent")
              and part == {"text": "Hello there.", "speechMetadata": {"style": tts.STYLE_INSTRUCTION}}
              and gen.get("responseModalities") == ["AUDIO"]
              # google-genai sends voice_config in snake_case; Google's JSON APIs accept both.
              and (gen.get("speechConfig") or {}).get("voiceConfig",
                   (gen.get("speechConfig") or {}).get("voice_config")) == {"voice": "Charon"},
              "Gemini request: gemini-3.8-flash-tts, Charon, style in speechMetadata (not in the text)")
        check(syn.pcm == PCM and (syn.input_tokens, syn.output_tokens) == (12, 125)
              and syn.served_model == "gemini-3.8-flash-tts" and syn.request_id == "resp_mock456",
              "Gemini raw PCM: audio, usage, model version and response id parsed")

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(PCM)
        gem_state.update(mime="audio/wav", data=wav_buf.getvalue())
        check(prov.synthesize("x").pcm == PCM, "Gemini WAV inline audio: PCM extracted from the WAV")
        gem_state.update(mime="audio/L16;codec=pcm;rate=16000", data=PCM)
        try:
            prov.synthesize("x")
            ok = False
        except tts.TTSFormatError:
            ok = True
        check(ok, "Gemini audio at another sample rate is a format error (never resampled)")
        gem_state.update(mime="audio/L16;codec=pcm;rate=24000", data=PCM)

        gem_state["fail"] = [500]
        r = TMP / "runs" / "gemini_mock"
        n_before = len(gm.requests)
        code = narrate(settings_for(SAMPLE_LAB, voice="Charon", provider="gemini"), r,
                       provider=prov, log=quiet)
        gmm = manifest_of(r)
        check(code == 0 and gmm["status"] == "complete" and gmm["totals"]["transient_retries"] == 1
              and len(gm.requests) - n_before == len(refs) + 1
              and validate_bundle(r / gmm["bundle"]["zip_file"]).ok,
              "Gemini end to end: a 500 is retried once (no SDK retries) and the zip validates")
        check(abs(gmm["totals"]["kept_audio_cost_usd"] - len(refs) * (12 * 1.00 + 125 * 18.00) / 1e6) < 1e-9,
              "Gemini cost from the reported usage at the gemini-3.8-flash-tts rates")
        gem_state["fail"] = [403]
        r = TMP / "runs" / "gemini_403"
        n_before = len(gm.requests)
        code = narrate(settings_for(SAMPLE_LAB, voice="Charon", provider="gemini"), r,
                       provider=prov, log=quiet)
        check(code == 1 and len(gm.requests) - n_before == 1, "a Gemini 403 stops the run after one request")
        def quota(quota_id, delay):
            return [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaId": quota_id, "quotaValue": "100"}]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]
        gem_state["fail"] = [(429, quota("GenerateRequestsPerDayPerProjectPerModel", "40s"))]
        r = TMP / "runs" / "gemini_daily_quota"
        n_before = len(gm.requests)
        code = narrate(settings_for(SAMPLE_LAB, voice="Charon", provider="gemini"), r,
                       provider=prov, log=quiet)
        check(code == 1 and len(gm.requests) - n_before == 1
              and "daily request quota" in manifest_of(r)["clips"][0]["error_reason"],
              "the real Gemini client's 429 for a used-up daily quota stops the run after one request")
        gem_state["fail"] = [(429, quota("GenerateRequestsPerMinutePerProjectPerModel", "0.4s"))]
        r = TMP / "runs" / "gemini_per_minute"
        t0 = time.monotonic()
        code = narrate(settings_for(SAMPLE_LAB, voice="Charon", provider="gemini"), r,
                       provider=prov, log=quiet)
        gpm = manifest_of(r)
        check(code == 0 and gpm["clips"][0]["transient_retries"] == 1 and time.monotonic() - t0 >= 0.4
              and "asked to wait 0.4 s" in gpm["clips"][0]["transient_errors"][0]["reason"],
              "the real Gemini client's per-minute 429 waits the RetryInfo delay, then succeeds")
        gem_state["no_audio"] = 1
        r = TMP / "runs" / "gemini_no_audio"
        code = narrate(settings_for(SAMPLE_LAB, voice="Charon", provider="gemini"), r,
                       provider=prov, log=quiet)
        gna = manifest_of(r)
        c0 = gna["clips"][0]
        check(code == 0 and gna["status"] == "complete" and c0["transient_retries"] == 1
              and "no audio" in c0["transient_errors"][0]["reason"]
              and "OTHER" in c0["transient_errors"][0]["detail"],
              "a Gemini reply with no audio (finish reason OTHER) is retried; it doesn't stop the run")
        na = next(x for x in gna["calls"] if x["status"] == "error")
        check(na["error_type"] == "TTSNoAudio" and na["input_tokens"] == 12 and na["output_tokens"] == 0
              and abs(na["cost_usd"] - 12 * 1.00 / 1e6) < 1e-15
              and na["served_model"] == "gemini-3.8-flash-tts"
              and abs(gna["totals"]["total_cost_usd"] - gna["totals"]["kept_audio_cost_usd"] - 12e-6) < 1e-15,
              "the billed tokens of a Gemini reply with no audio are logged and counted in the total")

        slow = {"sleep": 0}
        real_handler = gemini_handler

        def slow_gemini(path, body):
            delay, slow["sleep"] = slow["sleep"], 0   # only the first request is slow
            if delay:
                time.sleep(delay)
            return real_handler(path, body)

        gs = Mock(slow_gemini)
        try:
            gfast = tts.make_provider("gemini", api_key="g-test", base_url=gs.url, time_limit=1.0)
            slow["sleep"] = 3
            t0 = time.monotonic()
            try:
                gfast.synthesize("x")
                outcome = "no error"
            except tts.TTSTimeout as exc:
                outcome = str(exc)
            took = time.monotonic() - t0
            check(outcome == "no complete reply within 1 s" and took < 2.5,
                  f"a Gemini reply slower than the limit is abandoned ({took:.1f} s)")
            check(gs.requests[-1][1].get("X-Server-Timeout") == "1",
                  "Gemini also sends the limit to the server as a deadline (X-Server-Timeout)")
            slow["sleep"] = 3
            r = TMP / "runs" / "gemini_time_limit"
            code = narrate(settings_for(SAMPLE_LAB, voice="Charon", provider="gemini"), r,
                           provider=gfast, log=quiet)
            c0 = manifest_of(r)["clips"][0]
            check(code == 0 and c0["transient_retries"] == 1 and c0["transient_errors"][0]["type"] == "TTSTimeout"
                  and c0["first_audio_seconds"] is not None and c0["request_id"] == "resp_mock456",
                  "Gemini end to end: a slow reply is abandoned and retried; timing and response id recorded")
        finally:
            gs.close()
    except _SkipGemini:
        print("     (Gemini mock checks skipped until google-genai is upgraded)")
    finally:
        gm.close()


# The spoken-form LLM call, for real (guarded client, tracker, provider kwargs), against
# a mock Anthropic server.
if HAVE_NARRATION:
    anth_reqs = []
    anth_state = {"bad": False, "fail": []}

    def anthropic_handler(path, body):
        anth_reqs.append(body)
        if anth_state["fail"]:
            code = anth_state["fail"].pop(0)
            return code, {"content-type": "application/json"}, json.dumps(
                {"type": "error", "error": {"type": "overloaded_error", "message": "mock"}}).encode()
        if anth_state["bad"]:   # a reply that never fits LabSpokenForms
            tool = body["tools"][0]["name"]
            return 200, {"content-type": "application/json"}, json.dumps({
                "id": "msg_mock", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "tool_use", "id": "toolu_bad", "name": tool, "input": {"oops": 1}}],
                "stop_reason": "tool_use", "stop_sequence": None,
                "usage": {"input_tokens": 1200, "output_tokens": 1000}}).encode()
        prompt = body["messages"][0]["content"]
        if isinstance(prompt, list):
            prompt = " ".join(block.get("text", "") for block in prompt)
        clips = _re.findall(r"\[([^\]\n]+)\]\n(.+?)(?=\n\n\[|\Z)", prompt.split("Clips:\n\n", 1)[1], _re.S)
        tool = body["tools"][0]["name"]
        return 200, {"content-type": "application/json"}, json.dumps({
            "id": "msg_mock", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "tool_use", "id": "toolu_mock", "name": tool, "input": {"clips": [
                {"audio_name": n, "spoken_text": t.replace("DNA", "D N A")} for n, t in clips]}}],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 1000},
        }).encode()

    am = Mock(anthropic_handler)
    saved_env = {k: os.environ.get(k) for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY")}
    os.environ["ANTHROPIC_BASE_URL"], os.environ["ANTHROPIC_API_KEY"] = am.url, "test"
    try:
        r = TMP / "runs" / "llm_anthropic"
        st = settings_for(SAMPLE_LAB)
        st.spoken_form, st.spoken_form_model = "llm", "claude-haiku-5.5"
        code = narrate(st, r, provider=FakeTTS(), log=quiet)
        la = manifest_of(r)
        req = anth_reqs[-1] if anth_reqs else {}
        check(code == 0 and len(anth_reqs) == 1 and req.get("model") == "claude-haiku-5-5"
              and req.get("tool_choice") == {"type": "auto", "disable_parallel_tool_use": True}
              and "exactly once" in str(req.get("system")) and req["tools"][0]["name"] == "LabSpokenForms",
              "the real spoken-form call goes through the guarded Anthropic path (tool_choice auto, one call)")
        check(la["spoken_form"]["input_tokens"] == 1200 and la["spoken_form"]["output_tokens"] == 1000
              and abs(la["spoken_form"]["cost_usd"] - (1200 * 0.10 + 1000 * 0.50) / 1e6) < 1e-12
              and la["spoken_form"]["served_models"] == ["claude-haiku-5-5"]
              and la["spoken_form"]["model_id"] == "claude-haiku-5-5",
              "its tokens, cost (Claude Haiku 5.5 rates) and served model are tracked")
        sfc = [x for x in la["calls"] if x["step"] == "spoken_form"]
        check(len(sfc) == 1 and sfc[0]["status"] == "ok" and sfc[0]["kind"] == "first"
              and sfc[0]["seconds"] is not None and sfc[0]["seconds"] >= 0
              and sfc[0]["input_tokens"] == 1200 and sfc[0]["served_model"] == "claude-haiku-5-5"
              and abs(sfc[0]["cost_usd"] - la["spoken_form"]["cost_usd"]) < 1e-15
              and abs(la["totals"]["spoken_form_cost_usd"] - la["spoken_form"]["cost_usd"]) < 1e-15,
              "the spoken-form request is in the call log with its own cost and time")
        check(any("D N A" in c["spoken_text"] for c in la["clips"] if c.get("spoken_text")),
              "the model's spoken text reaches the clips")
        check(req.get("output_config") == {"effort": "low"} and la["spoken_form"]["reasoning_effort"] == "low",
              "the spoken-form call asks for low reasoning effort, and the manifest records it")
        anth_state["bad"] = True
        r = TMP / "runs" / "llm_anthropic_bad"
        n_before = len(anth_reqs)
        code = narrate(st, r, provider=FakeTTS(), log=quiet)
        lb = manifest_of(r)
        attempts = len(anth_reqs) - n_before
        check(code == 0 and attempts == spoken_form.MAX_RETRIES + 1 and lb["spoken_form"]["error"]
              and lb["spoken_form"]["input_tokens"] == 1200 * attempts
              and abs(lb["spoken_form"]["cost_usd"] - attempts * (1200 * 0.10 + 1000 * 0.50) / 1e6) < 1e-12,
              "a spoken-form call that fails after its reasks still records the tokens and cost it used")
        sfc = [x for x in lb["calls"] if x["step"] == "spoken_form"]
        one = (1200 * 0.10 + 1000 * 0.50) / 1e6
        check([x["kind"] for x in sfc] == ["first"] + ["reask"] * spoken_form.MAX_RETRIES
              and all(x["status"] == "error" and x["error_type"] and abs(x["cost_usd"] - one) < 1e-15
                      for x in sfc)
              and abs(lb["totals"]["spoken_form_cost_usd"] - attempts * one) < 1e-12
              and abs(lb["totals"]["total_cost_usd"]
                      - lb["totals"]["tts_cost_usd"] - attempts * one) < 1e-12,
              "each reask is its own logged request with its cost, and all of them count in the total")
        anth_state["bad"] = False
        anth_state["fail"] = [529]
        r = TMP / "runs" / "llm_anthropic_529"
        code = narrate(st, r, provider=FakeTTS(), log=quiet)
        l5 = manifest_of(r)
        sfc = [x for x in l5["calls"] if x["step"] == "spoken_form"]
        check(code == 0 and not l5["spoken_form"]["error"] and l5["spoken_form"]["transient_retries"] == 1
              and [(x["kind"], x["status"]) for x in sfc] == [("first", "error"), ("transient re-send", "ok")]
              and sfc[0]["cost_usd"] == 0.0 and abs(sfc[1]["cost_usd"] - one) < 1e-15,
              "a spoken-form request rejected with HTTP 529 is logged at $0, then the re-send with its cost")
    finally:
        am.close()
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not FAILED else "\nSOME FAILED")
sys.exit(1 if FAILED else 0)
