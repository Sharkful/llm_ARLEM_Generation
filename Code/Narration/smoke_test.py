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
    data = json.loads(p.read_text())
    edit(data)
    p.write_text(json.dumps(data))
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

# ══ Part 2: the narration CLI with a fake voice service (#93) ═════════

print("\n── Part 2: narration CLI")
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
        return json.loads((Path(run_dir) / "narration_manifest.json").read_text())

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
    edge_path = fresh("edge_input") / "edge_json_lab_L4_x_output.json"
    edge_path.write_text(json.dumps(edge, indent=2), encoding="utf-8")
    plans = plan_clips(edge)
    by_pos = {(p.module, p.clip): p for p in plans}
    check(by_pos[(1, 2)].name == f"m01_{m1[1]['clipName']}" and by_pos[(1, 2)].name_change == "generated",
          "a clip with narration but no audioClip gets m{module:02d}_{clipName}")
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
    check(json.loads(edge_path.read_text()) == edge, "the input lab file is never modified")
    check(em["counts"]["names"] == {"generated": 1, "sanitized": 1, "deduplicated": 1, "dropped": 1}
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
    check(abs(rm["totals"]["cost_usd"] - len(refs) * COST_PER_CLIP) < 1e-9
          and rm["totals"]["input_tokens"] == 50 * len(refs),
          "cost comes from reported token usage at the tracking/pricing.py rates")
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
          and rm2["totals"]["cost_usd_this_run"] == 0
          and abs(rm2["totals"]["cost_usd"] - rm["totals"]["cost_usd"]) < 1e-9
          and len(rm2["runs"]) == 2,
          "a rerun skips every unchanged clip, keeps the lab's cost, and logs a second run")
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

    # ── Retries, failures, resume ────────────────────────────────────
    fake = FakeTTS(script=lambda n, t: HTTPError(503) if n == 0 else None)
    r = TMP / "runs" / "transient"
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    tm = manifest_of(r)
    check(code == 0 and tm["clips"][0]["transient_retries"] == 1 and tm["totals"]["transient_retries"] == 1,
          "a 503 is retried and counted as a transient retry")

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
    check(not stale.exists() and fm["bundle"]["validation"] is None,
          "an incomplete run builds no zip and removes a stale one")
    fake = FakeTTS()
    code = narrate(settings_for(SAMPLE_LAB), r, provider=fake, log=quiet)
    check(code == 0 and fake.calls == [target_text] and manifest_of(r)["status"] == "complete",
          "rerunning the same folder narrates only the failed clip and completes")

    for status in (401, 403, 404):
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
    check(nm["clips"][0]["cost_usd"] is None and "no token usage" in " ".join(nm["clips"][0]["warnings"]),
          "missing usage gives no cost and a warning, not a guess")

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

# ══ Part 3: real clients against local mock servers ═══════════════════

if HAVE_NARRATION:
    print("\n── Part 3: OpenAI and Gemini clients against mock servers")
    import base64
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Mock:
        """A local HTTP server: ``handler(path, body_json) -> (status, headers, bytes)``."""
        def __init__(self, handler):
            self.requests = []
            mock = self

            class H(BaseHTTPRequestHandler):
                def do_POST(self):
                    n = int(self.headers.get("content-length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    mock.requests.append((self.path, dict(self.headers), body))
                    status, headers, data = handler(self.path, body)
                    self.send_response(status)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def log_message(self, *a):
                    pass

            self.server = HTTPServer(("127.0.0.1", 0), H)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.url = f"http://127.0.0.1:{self.server.server_port}"

        def close(self):
            self.server.shutdown()

    half = len(PCM) // 2 // 2 * 2

    def sse(*events):
        return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)

    openai_state = {"fail": []}

    def openai_handler(path, body):
        if openai_state["fail"]:
            code = openai_state["fail"].pop(0)
            return code, {"content-type": "application/json"}, json.dumps(
                {"error": {"message": f"mock {code}", "type": "x"}}).encode()
        return 200, {"content-type": "text/event-stream", "openai-model": body.get("model", "")}, sse(
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
              and syn.served_model == "gpt-4o-mini-tts-2025-12-15",
              "OpenAI SSE: audio deltas joined, usage from speech.audio.done, served model from the header")

        openai_state["fail"] = [503]
        r = TMP / "runs" / "openai_mock"
        n_before = len(om.requests)
        code = narrate(settings_for(SAMPLE_LAB), r, provider=prov, log=quiet)
        om_m = manifest_of(r)
        check(code == 0 and om_m["status"] == "complete" and om_m["totals"]["transient_retries"] == 1
              and len(om.requests) - n_before == len(refs) + 1,
              "end to end over HTTP: a 503 is retried once by our retryer (the SDK makes no retries)")
        check(abs(om_m["totals"]["cost_usd"] - len(refs) * (14 * 0.60 + 101 * 12.00) / 1e6) < 1e-9,
              "OpenAI cost from the reported usage")

        openai_state["fail"] = [401]
        r = TMP / "runs" / "openai_401"
        n_before = len(om.requests)
        code = narrate(settings_for(SAMPLE_LAB), r, provider=prov, log=quiet)
        check(code == 1 and len(om.requests) - n_before == 1 and manifest_of(r)["status"] == "failed",
              "an OpenAI 401 stops the run after one request")
    finally:
        om.close()

    gem_state = {"mime": "audio/L16;codec=pcm;rate=24000", "data": PCM, "fail": []}

    def gemini_handler(path, body):
        if gem_state["fail"]:
            code = gem_state["fail"].pop(0)
            return code, {"content-type": "application/json"}, json.dumps(
                {"error": {"code": code, "message": f"mock {code}", "status": "X"}}).encode()
        return 200, {"content-type": "application/json"}, json.dumps({
            "candidates": [{"content": {"role": "model", "parts": [{"inlineData": {
                "mimeType": gem_state["mime"], "data": base64.b64encode(gem_state["data"]).decode()}}]},
                "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 125, "totalTokenCount": 137},
            "modelVersion": "gemini-3.8-flash-tts",
        }).encode()

    gm = Mock(gemini_handler)
    try:
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
              and syn.served_model == "gemini-3.8-flash-tts",
              "Gemini raw PCM: audio, usage and model version parsed")

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
        check(abs(gmm["totals"]["cost_usd"] - len(refs) * (12 * 1.00 + 125 * 18.00) / 1e6) < 1e-9,
              "Gemini cost from the reported usage at the gemini-3.8-flash-tts rates")
        gem_state["fail"] = [403]
        r = TMP / "runs" / "gemini_403"
        n_before = len(gm.requests)
        code = narrate(settings_for(SAMPLE_LAB, voice="Charon", provider="gemini"), r,
                       provider=prov, log=quiet)
        check(code == 1 and len(gm.requests) - n_before == 1, "a Gemini 403 stops the run after one request")
    finally:
        gm.close()

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not FAILED else "\nSOME FAILED")
sys.exit(1 if FAILED else 0)
