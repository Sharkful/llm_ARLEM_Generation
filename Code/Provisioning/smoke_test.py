"""
Offline smoke test for provisioning: no API keys, no network, no paid calls.

    python "Code/Provisioning/smoke_test.py"

Part 1 (the committed seed list, #107): it parses, every download is https and
pinned, ids and names are unique, built-ins match the moon-lab sets in
``lab_metrics`` and ``Code/Hierarchical/registry.py``, and its licenses are allowed.

Part 2 (the asset library, #107), against a local mock download server and a
throwaway library (never the real one):
- seeding: built-ins, pinning, a transient 503 retried, a 404 not retried, a
  non-JPEG rejected and never pinned, disallowed and uncredited licenses refused,
  a cut-off download retried and then failed alone, an unexpected error failing only
  its entry, a killed run keeping the pins it got, an unpinned file never
  downloaded, an entry retried after another run wrote meanwhile, a pin that can't
  be saved reported, and a built-in whose download fails keeping an intact file and
  dropping a damaged one
- network failures: proxy refusals, unknown hosts and bad certificates fail at once;
  temporary failures are retried; no benchmark or instructor import
- reruns download nothing; a damaged local file is re-downloaded; a changed
  source file is refused; ``--pin`` never overwrites an existing pin
- two libraries seeded from one list share a content hash, not a library id
- lookups by name and alias ignore case; alias clashes are refused
- a write is refused when another process wrote in between, and nothing is lost;
  two simultaneous writers never both succeed; a held lock makes a writer give up;
  a run that dies holding the lock doesn't block the next
- ``put`` refuses an entry without a license or size and leaves the library as it was
- the validator catches missing, altered and wrong-format files, duplicate names,
  blank aliases and unlisted files; sizes must be real sizes; files of two roles
  can't collide; library.json is 0644
- credit lines, the GLB check, library path resolution and the CLI exit codes

Parts 3 and 4: the audit, describe and route, the reuse judge and the CLI, with a fake
LLM. Part 5: the same CLI on the real client (create_instructor_client, guarded_create,
the tracker) against a local mock Anthropic server: per-step calls, tokens, cost and
served models. Part 6: the molecule builder (#111) against a mock PubChem server
(3D records, idealized VSEPR geometry, lone pairs, formula lookup, failures), the
builder route in describe and route, and ``build_assets.py`` (build, save, alias).

Part 7: search (#110) against mock Poly Haven, GitHub, NASA, ambientCG, Poly Pizza and
Smithsonian servers with a fake
vision ranker: collecting, the license / format / budget filters, the text pass, the
ranking checks, auto-pick, glTF to GLB, texture resizing, a Git LFS pointer, no match,
text-only ranking, a reviewer's choice, resume, a model over budget, and --save.

Requires the project venv (pydantic, Pillow; Part 5 also anthropic and instructor;
Parts 6 and 7 trimesh and matplotlib).
Exit code 1 on any failure.
"""

from __future__ import annotations

import contextlib
import copy
import http.server
import io
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Code" / "Benchmark"))

import library as lib  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        FAILURES.append(what)


def jpeg_bytes(w: int = 64, h: int = 32, color=(200, 80, 40)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="JPEG")
    return buf.getvalue()


def png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ── Mock download server ──────────────────────────────────────────────


class Server:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.fail_once: dict[str, int] = {}  # path -> status returned on the first request
        self.cut_off: set[str] = set()  # paths whose reply stops short of its Content-Length
        self.hits: dict[str, int] = {}
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                server.hits[self.path] = server.hits.get(self.path, 0) + 1
                if self.path in server.fail_once:
                    self.send_response(server.fail_once.pop(self.path))
                    self.end_headers()
                    return
                body = server.files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                promised = len(body) + (1000 if self.path in server.cut_off else 0)
                self.send_header("Content-Length", str(promised))
                self.end_headers()
                self.wfile.write(body)  # then the connection closes (HTTP/1.0)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, path: str) -> str:
        return self.base + path

    def close(self):
        self.httpd.shutdown()


# ── Part 1 ────────────────────────────────────────────────────────────


def part1() -> None:
    print("Part 1: the committed seed list")
    seed = lib.load_seed()
    check(seed.version == 1, "seed list parses (version 1)")
    downloads = [e for e in seed.entries if e.download]
    check(all(e.download.url.startswith("https://") for e in downloads), "every download is https")
    check(all(e.download.sha256 and e.download.bytes for e in downloads),
          f"every download is pinned ({len(downloads)} downloads)")
    names = [lib.name_key(e.name) for e in seed.entries]
    check(len(names) == len(set(names)), "names are unique ignoring case")
    check(all(lib.ASSET_ID.match(e.asset_id) for e in seed.entries), "asset ids are well formed")
    check(all(e.provenance is None or e.provenance.license in lib.ALLOWED_LICENSES
              for e in seed.entries), "every license is allowed")
    check(all(e.download is None or e.provenance is not None for e in seed.entries),
          "every download has provenance")

    from lab_metrics import KNOWN_PREFABS, KNOWN_TEXTURES
    builtin_models = {lib.name_key(e.name) for e in seed.entries if e.builtin and e.kind == "model"}
    builtin_textures = {lib.name_key(e.name) for e in seed.entries if e.builtin and e.kind == "texture"}
    check(builtin_models == KNOWN_PREFABS, "built-in models == lab_metrics.KNOWN_PREFABS")
    check(builtin_textures == KNOWN_TEXTURES, "built-in textures == lab_metrics.KNOWN_TEXTURES")
    registry_src = (ROOT / "Code" / "Hierarchical" / "registry.py").read_text(encoding="utf-8")
    exact = {e.name for e in seed.entries if e.builtin}
    check(all(f'"{n}"' in registry_src for n in exact - {"demoPrefab"}),
          "built-in names are spelled as in Code/Hierarchical/registry.py")
    demo = next(e for e in seed.entries if e.name == "demoPrefab")
    check(not demo.reusable, "demoPrefab is not reusable")
    check(not any(e.name.endswith("_fictional") for e in seed.entries),
          "no fictional dwarf-planet maps")


# ── Part 2 ────────────────────────────────────────────────────────────


def seed_doc(srv: Server) -> dict:
    prov = {"source_site": "example.org", "page_url": "https://example.org/a", "author": "A. Author",
            "title": "Test map", "license": "CC-BY-4.0"}

    def tex(asset_id, path, **extra):
        e = {"asset_id": asset_id, "kind": "texture", "name": asset_id, "mapping": "equirectangular",
             "description": f"test texture {asset_id}",
             "download": {"url": srv.url(path)},
             "provenance": dict(prov)}
        e.update(extra)
        return e

    no_author = tex("no_author", "/b.jpg")
    no_author["provenance"].pop("author")
    nc = tex("nc_map", "/b.jpg")
    nc["provenance"]["license"] = "CC-BY-NC-4.0"
    return {"version": 1, "entries": [
        {"asset_id": "box_builtin", "kind": "model", "name": "boxPrefab", "builtin": True,
         "description": "a built-in box", "form": "primitive"},
        tex("sky_builtin", "/missing_builtin.jpg", builtin=True),
        tex("map_a", "/a.jpg"),
        tex("cut_off", "/short.jpg"),
        tex("flaky", "/flaky.jpg"),
        tex("missing", "/missing.jpg"),
        tex("not_jpeg", "/png.jpg"),
        nc,
        no_author,
    ]}


def run_seed(store, seed_path, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rep = lib.seed_library(store, seed_path, **kw)
    return {r.asset_id: r for r in rep.results}, rep


def part2(tmp: Path) -> None:
    print("Part 2: the asset library")
    import tenacity
    lib.TRANSIENT_WAIT = tenacity.wait_none()  # the shared retry policy, no real back-off
    attempts = lib._policy().ATTEMPTS
    srv = Server()
    try:
        srv.files = {"/a.jpg": jpeg_bytes(), "/b.jpg": jpeg_bytes(color=(5, 6, 7)),
                     "/flaky.jpg": jpeg_bytes(color=(1, 2, 3)), "/png.jpg": png_bytes(),
                     "/short.jpg": jpeg_bytes(color=(7, 7, 7))}
        srv.cut_off = {"/short.jpg"}
        seed_path = tmp / "seed.json"
        lib.write_json_atomic(seed_path, seed_doc(srv))

        # Seeding
        store = lib.LocalFolderStore(tmp / "libA", create=True)
        check(store.identity()["revision"] == 0 and store.entries() == [], "a new library is empty")
        res, rep = run_seed(store, seed_path)
        check(res["map_a"].status == "failed" and "not pinned" in res["map_a"].detail,
              "an unpinned download is refused without --pin")
        check(store.get("map_a") is None and srv.hits.get("/a.jpg", 0) == 0,
              "...before downloading anything, and nothing is added")
        check(res["box_builtin"].status == "added", "a built-in without a file is added")
        srv.fail_once = {"/flaky.jpg": 503}
        res, rep = run_seed(store, seed_path, pin=True)
        check(res["box_builtin"].status == "unchanged", "...and is unchanged on the next run")
        check(res["map_a"].status == "added", "a pinned download is added")
        check(res["flaky"].status == "added" and res["flaky"].retries == 1,
              "a 503 is retried once and then succeeds")
        check(res["missing"].status == "failed" and "404" in res["missing"].detail
              and srv.hits.get("/missing.jpg") == 1, "a 404 fails without retrying")
        check(res["not_jpeg"].status == "failed" and "not JPEG" in res["not_jpeg"].detail,
              "a PNG served as .jpg is refused")
        check(res["nc_map"].status == "failed" and "not allowed" in res["nc_map"].detail,
              "a CC-BY-NC license is refused")
        check(res["no_author"].status == "failed" and "needs credit" in res["no_author"].detail,
              "a CC-BY asset without an author is refused")
        check(res["cut_off"].status == "failed" and "IncompleteRead" in res["cut_off"].detail
              and res["cut_off"].retries == attempts - 1,
              "a cut-off download is retried as transient, then fails only its own entry")
        check(res["flaky"].status == "added", "...and the entries after it still run")
        sky = store.get("sky_builtin")
        check(res["sky_builtin"].status == "partial" and sky is not None and sky.builtin
              and sky.files == [] and sky.pixels is None,
              "a built-in whose download fails is still recorded, without its file")
        check(any(r.asset_id == "sky_builtin" for r in rep.failed), "...and is reported as needing attention")
        if os.name != "nt":
            check((tmp / "libA" / "library.json").stat().st_mode & 0o777 == 0o644,
                  "library.json is readable by everyone (0644)")
        pinned = {e["asset_id"]: e["download"].get("sha256")
                  for e in json.loads(seed_path.read_text(encoding="utf-8"))["entries"] if "download" in e}
        check(bool(pinned["map_a"]) and bool(pinned["flaky"]), "accepted downloads are pinned")
        check(not pinned["not_jpeg"] and not pinned["nc_map"] and not pinned["missing"],
              "refused downloads are never pinned")
        a = store.get("map_a")
        check(a.pixels == [64, 32] and a.file("thumbnail") is not None, "pixels and thumbnail recorded")
        check(a.provenance.license_url == lib.LICENSE_URLS["CC-BY-4.0"]
              and a.provenance.source_url == srv.url("/a.jpg"), "provenance filled in")
        check((tmp / "libA" / "map_a" / "texture.jpg").is_file(), "files land in <library>/<asset_id>/")
        check([p for p in store.validate() if p.level == "error"] == [], "the library validates")

        # Reruns
        hits = srv.hits.get("/a.jpg", 0)
        rev = store.identity()["revision"]
        res, _ = run_seed(store, seed_path)
        check(res["map_a"].status == "unchanged" and srv.hits.get("/a.jpg", 0) == hits,
              "a rerun downloads nothing and changes nothing")
        check(store.identity()["revision"] == rev, "...and the revision stays the same")
        (tmp / "libA" / "map_a" / "texture.jpg").write_bytes(b"damaged")
        check(any("checksum" in p.message for p in store.validate()), "validate catches an altered file")
        res, _ = run_seed(store, seed_path)
        check(res["map_a"].status == "updated" and srv.hits.get("/a.jpg") == hits + 1,
              "a damaged local file is downloaded again")
        check([p for p in store.validate() if p.level == "error"] == [], "...and the library validates again")
        srv.files["/a.jpg"] = jpeg_bytes(color=(9, 9, 9))
        (tmp / "libA" / "map_a" / "texture.jpg").unlink()
        res, _ = run_seed(store, seed_path, pin=True)
        check(res["map_a"].status == "failed" and "source file changed" in res["map_a"].detail,
              "a changed source file is refused, even with --pin")
        check(next(e for e in json.loads(seed_path.read_text(encoding="utf-8"))["entries"]
                   if e["asset_id"] == "map_a")["download"]["sha256"] == pinned["map_a"],
              "...and its pin is kept")
        srv.files["/a.jpg"] = jpeg_bytes()
        run_seed(store, seed_path)

        # Identity across libraries
        other = lib.LocalFolderStore(tmp / "libB", create=True)
        run_seed(other, seed_path)
        ia, ib = store.identity(), other.identity()
        check(ia["content_hash"] == ib["content_hash"], "two libraries from one seed list share a content hash")
        check(ia["library_id"] != ib["library_id"], "...but not a library id")

        # Lookups and aliases
        check(store.find("  MAP_A ") is not None, "find ignores case and spaces")
        store.add_alias("map_a", "jupiterTexture")
        check(store.find("JUPITERTEXTURE").asset_id == "map_a", "an alias is found ignoring case")
        check(ia["content_hash"] != store.identity()["content_hash"], "an alias changes the content hash")
        try:
            store.add_alias("flaky", "jupitertexture")
            check(False, "an alias already used elsewhere is refused")
        except lib.LibraryError:
            check(True, "an alias already used elsewhere is refused")

        # One entry failing for an unexpected reason, or the run being killed, loses nothing
        doc = json.loads(seed_path.read_text(encoding="utf-8"))
        doc["entries"] = [dict(e, asset_id=f"extra_{i}", name=f"extra_{i}",
                               download={"url": srv.url(f"/extra_{i}.jpg")})
                          for i, e in enumerate(doc["entries"][2:5])]
        for i in range(3):
            srv.files[f"/extra_{i}.jpg"] = jpeg_bytes(color=(i, 40, 80))
        extra_seed = tmp / "extra_seed.json"
        lib.write_json_atomic(extra_seed, doc)
        real_thumbnail = lib.make_thumbnail
        calls = []

        def flaky_thumbnail(src, dest):
            calls.append(src)
            if len(calls) == 2:
                raise OSError("No space left on device")
            if len(calls) == 3:
                raise KeyboardInterrupt
            real_thumbnail(src, dest)

        lib.make_thumbnail = flaky_thumbnail
        try:
            run_seed(store, extra_seed, pin=True)
            check(False, "a killed seed run stops")
        except KeyboardInterrupt:
            check(True, "a killed seed run stops")
        finally:
            lib.make_thumbnail = real_thumbnail
        pins = [e["download"].get("sha256") for e in json.loads(extra_seed.read_text(encoding="utf-8"))["entries"]]
        check(bool(pins[0]) and not pins[1] and not pins[2],
              "pins of entries accepted before the run was killed are already saved")
        store = lib.LocalFolderStore(tmp / "libA")
        res, _ = run_seed(store, extra_seed, pin=True)
        check(res["extra_0"].status == "unchanged" and res["extra_1"].status == "added"
              and res["extra_2"].status == "added", "...and a rerun picks up where it stopped")
        lib.make_thumbnail = lambda src, dest: (_ for _ in ()).throw(OSError("No space left on device"))
        try:
            doc["entries"] = [dict(doc["entries"][0], asset_id="extra_3", name="extra_3",
                                   download={"url": srv.url("/extra_0.jpg")})]
            lib.write_json_atomic(extra_seed, doc)
            res, _ = run_seed(store, extra_seed, pin=True)
        finally:
            lib.make_thumbnail = real_thumbnail
        check(res["extra_3"].status == "failed" and "OSError" in res["extra_3"].detail,
              "an unexpected error fails only its own entry, with its type named")

        # Another run writing during a seed: the refused entry is retried and succeeds
        doc["entries"] = [dict(doc["entries"][0], asset_id="extra_4", name="extra_4",
                               download={"url": srv.url("/extra_0.jpg")})]
        lib.write_json_atomic(extra_seed, doc)
        other = lib.LocalFolderStore(tmp / "libA")
        store = lib.LocalFolderStore(tmp / "libA")
        logged = []

        def meddle(msg):
            logged.append(msg)
            if len(logged) == 1:
                other.add_alias("map_a", "addedMeanwhile")

        rep4 = lib.seed_library(store, extra_seed, pin=True, log=meddle)
        r4 = rep4.results[0]
        fresh = lib.LocalFolderStore(tmp / "libA")
        check(r4.status == "added" and fresh.get("extra_4") is not None
              and fresh.find("addedMeanwhile") is not None,
              "an entry refused because another run wrote meanwhile is retried, keeping both")

        # A pin that can't be saved is reported, not hidden behind 'failed'
        doc["entries"] = [dict(doc["entries"][0], asset_id="extra_5", name="extra_5",
                               download={"url": srv.url("/extra_0.jpg")})]
        lib.write_json_atomic(extra_seed, doc)
        real_write = lib.write_json_atomic

        def no_seed_write(path, data):
            if Path(path) == extra_seed:
                raise PermissionError(13, "Permission denied", str(path))
            real_write(path, data)

        lib.write_json_atomic = no_seed_write
        try:
            rep5 = lib.seed_library(lib.LocalFolderStore(tmp / "libA"), extra_seed, pin=True,
                                    log=lambda m: None)
        finally:
            lib.write_json_atomic = real_write
        r5 = rep5.results[0]
        check(r5.status == "added" and "pin was not saved" in r5.detail
              and r5 in rep5.failed and lib.LocalFolderStore(tmp / "libA").get("extra_5") is not None,
              "a pin that can't be saved: the asset is reported as added, and as needing attention")

        # A built-in whose new download fails keeps a good file, and drops only a bad one
        srv.files["/moon.jpg"] = jpeg_bytes(color=(90, 90, 90))
        moon = {"asset_id": "moon_builtin", "kind": "texture", "name": "moonBuiltin", "builtin": True,
                "mapping": "equirectangular", "description": "a built-in moon",
                "download": {"url": srv.url("/moon.jpg")},
                "provenance": {"page_url": "https://example.org/m", "author": "A", "title": "Moon",
                               "license": "CC-BY-4.0"}}
        moon_seed = tmp / "moon_seed.json"
        lib.write_json_atomic(moon_seed, {"version": 1, "entries": [moon]})
        store = lib.LocalFolderStore(tmp / "libA")
        res, _ = run_seed(store, moon_seed, pin=True)
        good = store.get("moon_builtin").file("texture").sha256
        repinned = json.loads(moon_seed.read_text(encoding="utf-8"))
        repinned["entries"][0]["download"]["sha256"] = "0" * 64  # pin updated upstream
        lib.write_json_atomic(moon_seed, repinned)
        del srv.files["/moon.jpg"]  # ...and the new file can't be fetched here
        res, _ = run_seed(store, moon_seed)
        m = store.get("moon_builtin")
        check(res["moon_builtin"].status == "partial" and "kept its current file" in res["moon_builtin"].detail
              and m.file("texture") is not None and m.file("texture").sha256 == good
              and store.path_of("moon_builtin", "texture").is_file(),
              "a built-in whose new download fails keeps its intact file")
        store.path_of("moon_builtin", "texture").write_bytes(b"damaged")
        res, _ = run_seed(store, moon_seed)
        m = store.get("moon_builtin")
        check(res["moon_builtin"].status == "partial" and m.file("texture") is None and m.pixels is None
              and m.file("thumbnail") is not None,
              "...but drops a damaged one (keeping its intact thumbnail)")
        check(not [p for p in store.validate() if p.level == "error"], "...and the library validates")

        # Network failures: permanent ones fail at once, transient ones are retried
        def classify(reason):
            return lib._network_failure("u", lib.urllib.error.URLError(reason))

        e = classify(OSError("Tunnel connection failed: 403 Forbidden"))
        check(e.status == 403 and not e.transient and e.status not in lib._policy().STATUS,
              "a proxy refusing the host (403) is permanent")
        check(classify(OSError("Tunnel connection failed: 503 Service Unavailable")).status
              in lib._policy().STATUS, "a proxy answering 503 is transient")
        check(not classify(socket.gaierror(socket.EAI_NONAME, "Name or service not known")).transient,
              "a host that doesn't exist is permanent")
        check(classify(socket.gaierror(socket.EAI_AGAIN, "Temporary failure")).transient,
              "a temporary DNS failure is transient")
        check(not classify(ssl.SSLCertVerificationError("certificate verify failed")).transient,
              "a certificate that fails verification is permanent")
        check(classify(ConnectionRefusedError(111, "refused")).transient
              and classify(TimeoutError("timed out")).transient, "refused and timed-out connections are transient")
        check(not classify("unknown url type: ftp").transient, "an unsupported URL is permanent")
        real_fetch, tries = lib._fetch_once, []

        def refused(url, dest, headers=None):
            tries.append(url)
            raise classify(OSError("Tunnel connection failed: 403 Forbidden"))

        lib._fetch_once = refused
        try:
            lib.download("https://blocked.example/x.jpg", tmp / "x.jpg")
            check(False, "a proxy refusal is tried once")
        except lib.DownloadError:
            check(len(tries) == 1, "a proxy refusal is tried once, not retried")
        finally:
            lib._fetch_once = real_fetch

        # Concurrent writers
        s1 = lib.LocalFolderStore(tmp / "libA")
        s2 = lib.LocalFolderStore(tmp / "libA")
        s1.add_alias("flaky", "flakyOne")
        try:
            s2.add_alias("flaky", "flakyTwo")
            check(False, "a write after another process's write is refused")
        except lib.ConcurrentWriteError:
            check(True, "a write after another process's write is refused")
        check("flakyTwo" not in s2.get("flaky").aliases, "...the refused change is not kept in memory")
        s2.add_alias("flaky", "flakyTwo")
        check({"flakyOne", "flakyTwo"} <= set(lib.LocalFolderStore(tmp / "libA").get("flaky").aliases),
              "...and a retry keeps both writers' changes")

        # put refusals leave the library as it was
        store = lib.LocalFolderStore(tmp / "libA")
        rev = store.identity()["revision"]
        bad = lib.LibraryEntry(asset_id="no_license", kind="texture", name="noLicense",
                               description="x", mapping="uv")
        img = tmp / "t.jpg"
        img.write_bytes(jpeg_bytes())
        for what, entry in [("no provenance", bad),
                            ("no pixels", bad.model_copy(update={"provenance": lib.Provenance(license="CC0-1.0")}))]:
            try:
                store.put(entry, {"texture": img})
                check(False, f"put refuses a texture with {what}")
            except lib.LibraryError:
                check(True, f"put refuses a texture with {what}")
        check(store.identity()["revision"] == rev and not (tmp / "libA" / "no_license").exists()
              and not [p for p in (tmp / "libA").iterdir() if p.name.startswith(".")],
              "...and leaves no entry, folder or staging files")
        good = bad.model_copy(update={"provenance": lib.Provenance(license="CC0-1.0"), "pixels": [64, 32]})
        store.put(good, {"texture": img})
        check(store.get("no_license").file("texture").sha256 == lib.sha256_of(img), "put accepts a valid texture")
        try:
            store.put(good.model_copy(update={"asset_id": "dup_name"}), {"texture": img})
            check(False, "put refuses a name another entry already uses")
        except lib.LibraryError:
            check(True, "put refuses a name another entry already uses")

        # Two writers at the same moment: one succeeds, the other is refused, nothing is lost
        s3 = lib.LocalFolderStore(tmp / "libA")
        s4 = lib.LocalFolderStore(tmp / "libA")
        barrier = threading.Barrier(2)
        outcome = {}

        def writer(st, aid):
            e = good.model_copy(update={"asset_id": aid, "name": aid})
            barrier.wait()
            try:
                st.put(e, {"texture": img})
                outcome[aid] = "ok"
            except lib.ConcurrentWriteError:
                outcome[aid] = "refused"

        threads = [threading.Thread(target=writer, args=(st, aid))
                   for st, aid in ((s3, "race_one"), (s4, "race_two"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        fresh = lib.LocalFolderStore(tmp / "libA")
        winner = next((a for a, o in outcome.items() if o == "ok"), None)
        loser = next((a for a, o in outcome.items() if o == "refused"), None)
        check(winner is not None and loser is not None and fresh.get(winner) is not None
              and fresh.get(loser) is None and not (tmp / "libA" / loser).exists(),
              "two simultaneous writers: one is recorded, the other refused, none silently lost")
        lock = tmp / "libA" / lib.LOCK_NAME
        with open(lock, "a+b") as f:
            check(lib._try_lock(f), "the lock is released after writing")
            lib._unlock(f)

        # The lock: a writer waits for another run's lock, then gives up
        store = lib.LocalFolderStore(tmp / "libA")
        saved_timeout, lib.LOCK_TIMEOUT = lib.LOCK_TIMEOUT, 0.2
        with open(lock, "a+b") as held:
            lib._try_lock(held)
            try:
                store.add_alias("map_a", "lockedAlias")
                check(False, "a writer gives up on a lock held by another run")
            except lib.LibraryError as exc:
                check("another run is writing" in str(exc),
                      "a writer gives up on a lock held by another run")
            finally:
                lib.LOCK_TIMEOUT = saved_timeout
                lib._unlock(held)
        # A run that dies holding the lock: the OS releases it, nothing to clean up
        code = ("import os, sys; sys.path.insert(0, sys.argv[1]); import library as lib; "
                "f = open(sys.argv[2], 'a+b'); assert lib._try_lock(f); os._exit(0)")
        died = subprocess.run([sys.executable, "-c", code, str(HERE), str(lock)], timeout=60)
        store.add_alias("map_a", "afterCrashAlias")
        check(died.returncode == 0 and store.find("afterCrashAlias") is not None,
              "a lock held by a run that died is released by the OS")
        probe = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import library as lib; "
             "lib._policy(); print('benchmark' in sys.modules, 'instructor' in sys.modules)", str(HERE)],
            capture_output=True, text=True, timeout=60)
        check(probe.stdout.strip() == "False False",
              "downloads use the shared retry policy without importing benchmark or instructor")

        # Sizes must be real sizes; aliases can't be blank; files can't collide
        for what, kw in [("one-number bounds", {"bounds": [0.4]}), ("a zero bound", {"bounds": [1, 0, 1]}),
                         ("negative triangles", {"triangles": -5}), ("zero pixels", {"pixels": [0, 0]}),
                         ("one-number pixels", {"pixels": [64]})]:
            try:
                lib.LibraryEntry(asset_id="sz", kind="model", name="sz", description="x", **kw)
                check(False, f"an entry with {what} is refused")
            except lib.ValidationError:
                check(True, f"an entry with {what} is refused")
        try:
            store.get("map_a").pixels = [64]
            check(False, "assigning a bad size is refused too")
        except lib.ValidationError:
            check(True, "assigning a bad size is refused too")
        store = lib.LocalFolderStore(tmp / "libA")
        try:
            store.add_alias("map_a", "   ")
            check(False, "a blank alias is refused")
        except lib.LibraryError:
            check(True, "a blank alias is refused")
        check(any("alias is empty" in p.message for p in lib.check_entry(
            store.get("map_a").model_copy(update={"aliases": [" "]}))), "check_entry flags a blank alias")
        before = store.get("flaky").file("texture").sha256
        odd = tmp / "odd"
        odd.mkdir()
        shutil.copyfile(tmp / "libA" / "flaky" / "thumbnail.png", odd / "texture.jpg")
        store.put(store.get("flaky"), {"thumbnail": odd / "texture.jpg"})
        f = store.get("flaky")
        check(f.file("texture").sha256 == before and f.file("thumbnail").path == "thumbnail.png"
              and [x.path for x in f.files] == ["texture.jpg", "thumbnail.png"],
              "a new thumbnail named texture.jpg is stored as thumbnail.png, the texture untouched")
        try:
            store.put(store.get("flaky"), {"preview": img})
            check(False, "an unknown file role is refused")
        except lib.LibraryError:
            check(True, "an unknown file role is refused")

        # Models: the GLB check
        glb = tmp / "model.glb"
        glb.write_bytes(b"glTF" + (2).to_bytes(4, "little") + (12).to_bytes(4, "little"))
        model = lib.LibraryEntry(asset_id="cube", kind="model", name="cubeModel", description="a cube",
                                 bounds=[1, 1, 1], triangles=12, provenance=lib.Provenance(license="CC0-1.0", method="kit"))
        store.put(model, {"model": glb})
        check(store.path_of("cube", "model").is_file(), "a GLB model is accepted")
        notglb = tmp / "fake.glb"
        notglb.write_bytes(b"not a model at all")
        try:
            store.put(model.model_copy(update={"asset_id": "cube2", "name": "cube2"}), {"model": notglb})
            check(False, "a model that isn't a GLB is refused")
        except lib.LibraryError:
            check(True, "a model that isn't a GLB is refused")
        try:
            store.put(model.model_copy(update={"asset_id": "cube3", "name": "cube3", "bounds": None}),
                      {"model": glb})
            check(False, "a model without a known size is refused")
        except lib.LibraryError:
            check(True, "a model without a known size is refused")

        # Validator on hand-made problems
        index = copy.deepcopy(store.index)
        twin = index.entries[0].model_copy(update={"asset_id": "zz_twin"})
        index.entries.append(twin)
        check(any("already used" in p.message for p in lib.check_index(index)),
              "check_index catches a duplicate name")
        (tmp / "libA" / "map_a" / "stray.txt").write_text("x")
        (tmp / "libA" / "orphan").mkdir()
        msgs = [p.message for p in store.validate()]
        check(any("unlisted file stray.txt" in m for m in msgs), "validate warns about an unlisted file")
        check(any("orphan/ has no library entry" in m for m in msgs), "validate warns about an orphan folder")
        shutil.copyfile(tmp / "libA" / "map_a" / "thumbnail.png", tmp / "libA" / "map_a" / "texture.jpg")
        e = store.get("map_a")
        e.files = [f.model_copy(update={"sha256": lib.sha256_of(tmp / "libA" / "map_a" / f.path),
                                        "bytes": (tmp / "libA" / "map_a" / f.path).stat().st_size})
                   for f in e.files]
        check(any("not JPEG" in p.message for p in lib.check_entry(e, tmp / "libA" / "map_a")),
              "validate catches a texture that isn't JPEG")
        (tmp / "libA" / "map_a" / "stray.txt").unlink()
        (tmp / "libA" / "orphan").rmdir()

        # Credits
        line = lib.credit_line(store.get("flaky"))
        check(line == '"Test map" by A. Author, https://example.org/a, CC-BY-4.0 '
                      '(https://creativecommons.org/licenses/by/4.0/), Changes: none.',
              "a CC-BY asset gets a complete credit line")
        check(lib.credit_line(store.get("no_license")) is None, "a CC0 asset needs no credit line")

        # Paths and the CLI
        os.environ[lib.LIBRARY_ENV] = str(tmp / "libB")
        check(lib.resolve_library_path() == (tmp / "libB").resolve(), f"${lib.LIBRARY_ENV} picks the library")
        check(lib.resolve_library_path(tmp / "libA") == (tmp / "libA").resolve(), "--library wins over the variable")
        del os.environ[lib.LIBRARY_ENV]
        check(lib.resolve_library_path() == lib.DEFAULT_LIBRARY, "the default is Artifacts/Assets/Library")

        def cli(*args):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return lib.main(list(args))

        check(cli("validate", "--library", str(tmp / "nothing")) == 2, "CLI: no library -> exit 2")
        check(not (tmp / "nothing").exists(), "CLI: validate never creates a library")
        check(cli("validate", "--library", str(tmp / "libB")) == 0, "CLI: a clean library -> exit 0")
        check(cli("info", "--library", str(tmp / "libB")) == 0, "CLI: info -> exit 0")
        check(cli("find", "missingName", "--library", str(tmp / "libB")) == 1, "CLI: an unknown name -> exit 1")
        (tmp / "libB" / "map_a" / "texture.jpg").write_bytes(b"x")
        check(cli("validate", "--library", str(tmp / "libB")) == 1, "CLI: a damaged library -> exit 1")
        check(cli("seed", "--library", str(tmp / "libC"), "--seed-file", str(seed_path)) == 1,
              "CLI: seed with failing entries -> exit 1")
        check(cli("seed", "--library", str(tmp / "libD"), "--seed-file", str(seed_path),
                  "--only", "map_a", "flaky", "box_builtin") == 0, "CLI: seed --only -> exit 0")
        check(cli("seed", "--library", str(tmp / "libE"), "--seed-file", str(seed_path),
                  "--only", "unknown_id") == 2, "CLI: seed --only with an unknown id -> exit 2")
        check(cli("seed", "--library", str(tmp / "libF"), "--seed-file", str(tmp / "nope.json")) == 2,
              "CLI: an unreadable seed list -> exit 2")
        check(not (tmp / "libE").exists() and not (tmp / "libF").exists(),
              "CLI: a seed that fails its argument checks creates no library")
    finally:
        srv.close()


# ── Part 3 ────────────────────────────────────────────────────────────


def tiny_lab() -> dict:
    """A small valid lab touching every audit case."""
    obj = lambda **kw: {"position": [0, 0, 0], "scale": [1, 1, 1], **kw}  # noqa: E731
    return {
        "labId": "tiny_lab", "courseName": "Test", "objectives": ["see things"],
        "modules": [
            {"moduleName": "One", "description": "first", "educationalObjectives": ["a"],
             "objects": [
                 obj(prefab="dnaHelixPrefab", name="dnaHelix", color=[1, 0, 0, 1]),
                 obj(prefab="DNAHELIXPREFAB", name="dnaHelix2"),
                 obj(prefab="moveableSphere", name="ball", texture="2k_mars"),
                 obj(prefab="summaryPanelPrefab", name="summaryPanel",
                     components=[{"type": "TextMeshProComponent", "text": "Key ideas"}]),
                 obj(prefab="textPrefab", name="title",
                     components=[{"type": "TextMeshProComponent", "text": "Hello"}]),
             ],
             "clips": [
                 {"clipName": "c1", "autoAdvance": False, "changeMeaning": "Introduces the helix and the summary panel.",
                  "narration": "Here is the DNA helix. It twists. The summary panel lists ideas.",
                  "changes": [{"target": "summaryPanel",
                               "components": [{"type": "TextMeshProComponent", "text": "Updated"}]}]},
             ]},
            {"moduleName": "Two", "description": "second", "educationalObjectives": ["b"],
             "objects": [obj(prefab="dnaDoubleHelixPrefab", name="helixAgain",
                             texture="heart_muscle_texture")],
             "clips": [{"clipName": "c2", "autoAdvance": False, "changeMeaning": "Shows the helix again in the second module.",
                        "narration": "Nothing relevant."}]},
        ],
    }


def fake_router(behaviour: dict):
    """A call_fn for both LLM steps, answering whatever the prompt lists (validated under
    the step's scope, as instructor would). The reuse judge decides "none" unless
    ``behaviour["reuse"][name]`` says otherwise; describe and route plans "search" unless
    ``behaviour[name]`` overrides fields. ``behaviour["raise"]`` makes the step named in
    ``behaviour["raise_in"]`` ("reuse" or "route", default both) fail."""

    def call(response_model, messages, max_retries):
        prompt = messages[0]["content"]
        listed = re.findall(r"^\[(model|texture)\] (\S+)$", prompt, flags=re.M)
        step = "reuse" if response_model.__name__ == "ReusePlan" else "route"
        if behaviour.get("raise") and behaviour.get("raise_in", step) == step:
            raise RuntimeError(behaviour["raise"])
        if step == "reuse":
            behaviour["reuse_calls"] = behaviour.get("reuse_calls", 0) + 1
            behaviour["reuse_prompt"] = prompt
            decisions = []
            for kind, name in listed:
                d = {"name": name, "kind": kind, "decision": "none", "reason": "test"}
                d.update(behaviour.get("reuse", {}).get(name, {}))
                decisions.append(d)
            return response_model.model_validate({"decisions": decisions})
        behaviour.setdefault("calls", 0)
        behaviour["calls"] += 1
        behaviour["prompt"] = prompt
        assets = []
        for kind, name in listed:
            a = {"name": name.upper() if behaviour.get("shout") else name, "kind": kind,
                 "description": f"a {name}", "search_keywords": [name, f"{name} model"],
                 "form": "object" if kind == "model" else "real_surface",
                 "route": "search", "reason": "test"}
            a.update(behaviour.get(name, {}))
            assets.append(a)
        return response_model.model_validate({"assets": assets})

    return call


def part3(tmp: Path) -> None:
    print("Part 3: audit, describe and route, provision_lab.py --dry-run (#106)")
    import audit
    import builders
    import provision_lab
    import routing
    from lab_metrics import analyze_assets, KNOWN_PREFABS, KNOWN_TEXTURES

    # The audit agrees with lab_metrics on every wave-2 lab
    labs = mismatches = 0
    for f in sorted((ROOT / "Artifacts" / "Data" / "Benchmark" / "Outputs").glob("*json_lab*_output.json")):
        lab = json.loads(f.read_text(encoding="utf-8"))
        if "modules" not in lab:
            continue
        labs += 1
        reqs = audit.audit_lab(audit.load_lab(f))
        a = analyze_assets(lab)
        novel_p = {r.key for r in reqs if r.kind == "model" and r.key not in KNOWN_PREFABS}
        novel_t = {r.key for r in reqs if r.kind == "texture" and r.key not in KNOWN_TEXTURES}
        if novel_p != {x.lower() for x in a["novel_prefab_names"]} or \
                novel_t != {x.lower() for x in a["novel_texture_names"]}:
            mismatches += 1
    check(labs >= 100 and mismatches == 0,
          f"the audit finds the same invented assets as lab_metrics in all {labs} wave-2 labs")

    # The audit on a small lab
    lab = tiny_lab()
    lab_path = tmp / "tiny_lab_output.json"
    lab_path.write_text(json.dumps(lab), encoding="utf-8")
    reqs = {(r.kind, r.key): r for r in audit.audit_lab(audit.load_lab(lab_path), {"dnahelixprefab": "shows DNA"})}
    helix = reqs[("model", "dnahelixprefab")]
    check(helix.spellings == ["dnaHelixPrefab", "DNAHELIXPREFAB"] and helix.use_count == 2,
          "names are grouped ignoring case, keeping each spelling")
    check(("model", "demoprefab") not in reqs, "demoPrefab (the module's system object) is not an asset")
    check(helix.narration == ["Here is the DNA helix."], "narration lines that mention the object are kept")
    check(helix.purpose == "shows DNA", "a hierarchical purpose is attached")
    panel = reqs[("model", "summarypanelprefab")]
    check(panel.uses[0].text == ["Key ideas", "Updated"], "text on the object, also from clip changes, is kept")
    mars = reqs[("texture", "2k_mars")]
    check(mars.uses[0].paired_with == "moveableSphere", "a texture records the model it is on")
    check(audit.name_words("dnaHelixPrefab") == "dna helix"
          and audit.name_words("heart_cutaway_texture") == "heart cutaway", "name words for narration matching")
    check(audit.purposes_from_metrics({"registry": {"invented_prefabs": {"heartPrefab": {"module": 1, "purpose": "pumps"}},
                                                     "invented_textures": {}}}) == {"heartprefab": "pumps"},
          "purposes come from a hierarchical metrics registry")
    for bad, what in [({"objects": []}, "a module-only output"), ([1, 2], "not an object")]:
        p = tmp / "bad.json"
        p.write_text(json.dumps(bad))
        try:
            audit.load_lab(p)
            check(False, f"{what} is refused")
        except audit.InputError:
            check(True, f"{what} is refused")

    # The plan's checks
    asked = [helix, reqs[("model", "dnadoublehelixprefab")], mars]
    good = {"assets": [
        {"name": "dnahelixprefab", "kind": "model", "description": "a helix", "search_keywords": ["dna", "double helix"],
         "form": "structure", "route": "composite", "reason": "r"},
        {"name": "dnaDoubleHelixPrefab", "kind": "model", "description": "a helix", "search_keywords": ["dna", "double helix"],
         "form": "structure", "route": "composite", "reason": "r", "same_as": "DNAHELIXPREFAB"},
        {"name": "2k_mars", "kind": "texture", "description": "mars", "search_keywords": ["mars map", "mars texture"],
         "form": "real_surface", "route": "search", "reason": "r"}]}
    with routing.routing_scope(asked):
        plan = routing.LabAssetPlan.model_validate(good)
    check(plan.assets[0].name == "dnaHelixPrefab" and plan.assets[1].same_as == "dnaHelixPrefab",
          "names and same_as come back in the lab's own spelling")
    broken = copy.deepcopy(good)
    broken["assets"][0]["route"] = "draw"                      # a texture route on a model
    broken["assets"][1]["same_as"] = "dnaDoubleHelixPrefab"    # itself
    broken["assets"][2]["kind"] = "model"                      # wrong kind
    broken["assets"].append(dict(good["assets"][0], name="ghostPrefab"))
    try:
        with routing.routing_scope(asked):
            routing.LabAssetPlan.model_validate(broken)
        check(False, "a bad plan is refused")
    except lib.ValidationError as exc:
        msg = str(exc)
        check(all(s in msg for s in ("route 'draw'", "same_as can't name the asset itself",
                                     "listed as a texture, not a model", "'ghostPrefab' is not one of the listed")),
              "every problem in a plan is reported in one error (one reask)")
    lenient = copy.deepcopy(good)
    lenient["assets"][0]["kind"] = "prefab"            # a field error...
    del lenient["assets"][0]["search_keywords"]
    lenient["assets"] = lenient["assets"][:2]          # ...and a missing asset
    try:
        with routing.routing_scope(asked):
            routing.LabAssetPlan.model_validate(lenient)
        check(False, "field problems and plan problems are refused")
    except lib.ValidationError as exc:
        msg = str(exc)
        check("not a prefab asset" in msg and "missing: 2k_mars" in msg and "1 validation error" in msg,
              "a bad field and a missing asset come back in the same single error")
    for n, ok in ((1, False), (2, True), (6, True), (7, False)):
        few = copy.deepcopy(good)
        few["assets"][2]["search_keywords"] = [f"k{i}" for i in range(n)]
        try:
            with routing.routing_scope(asked):
                routing.LabAssetPlan.model_validate(few)
            accepted = True
        except lib.ValidationError:
            accepted = False
        check(accepted == ok, f"{n} search keyword(s) {'accepted' if ok else 'refused'} (2 to 6)")
    twin = audit.AssetRequest(name="earth", kind="texture")
    twin_model = audit.AssetRequest(name="Earth", kind="model")
    with routing.routing_scope([twin_model, twin]):
        both = routing.LabAssetPlan.model_validate({"assets": [
            {"name": "earth", "kind": "model", "description": "a globe", "search_keywords": ["earth", "globe"],
             "form": "textured_primitive", "route": "shape", "reason": "r"},
            {"name": "earth", "kind": "texture", "description": "a map", "search_keywords": ["earth", "map"],
             "form": "real_surface", "route": "search", "reason": "r"}]})
    check([(a.kind, a.name) for a in both.assets] == [("model", "Earth"), ("texture", "earth")],
          "a model and a texture with the same name are two assets, each planned once")
    try:
        with routing.routing_scope(asked):
            routing.LabAssetPlan.model_validate({"assets": good["assets"][:2]})
        check(False, "a plan missing an asset is refused")
    except lib.ValidationError as exc:
        check("missing: 2k_mars" in str(exc), "a plan missing an asset is refused, naming it")
    chain = copy.deepcopy(good)
    chain["assets"][0]["same_as"] = "dnaDoubleHelixPrefab"
    try:
        with routing.routing_scope(asked):
            routing.LabAssetPlan.model_validate(chain)
        check(False, "merges into a merged asset are refused")
    except Exception as exc:
        check("itself merged" in str(exc), "merges into a merged asset are refused")
    check(routing.LabAssetPlan.model_validate(broken) is not None, "outside a scope nothing is checked")
    info = builders.register(builders.BuilderInfo("testmol", "model", "any molecule", "its name"))
    try:
        check("builder:testmol" in routing.routes_for("model") and "builder:testmol" not in routing.routes_for("texture")
              and "builder:testmol" in routing.build_prompt(lab, asked),
              "a registered builder becomes a route, offered by its description")
    finally:
        builders.REGISTRY.remove(info)

    # The CLI, with a library holding the built-ins (seeded without network) and 2k_mars
    p3lib = lib.LocalFolderStore(tmp / "libP3", create=True)
    no_download = {e.asset_id for e in lib.load_seed().entries if e.builtin and not e.download}
    run_seed(p3lib, lib.SEED_FILE, only=no_download)
    jpg = tmp / "mars.jpg"
    jpg.write_bytes(jpeg_bytes())
    p3lib.put(lib.LibraryEntry(asset_id="2k_mars", kind="texture", name="2k_mars", mapping="equirectangular",
                               description="Mars", pixels=[64, 32],
                               provenance=lib.Provenance(license="CC0-1.0")), {"texture": jpg})
    out = tmp / "prov_out"

    def run(*args, call_fn=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = provision_lab.main(list(args), call_fn=call_fn, output_root=out)
        return code, buf.getvalue()

    def last_manifest():
        return json.loads(max(out.rglob("provisioning_manifest.json"), key=os.path.getmtime).read_text(encoding="utf-8"))

    check(run(str(lab_path))[0] == 2, "CLI: without --dry-run -> exit 2 (the full run is #113)")
    check(run(str(lab_path), "--dry-run", "--run-group", "bad name")[0] == 2, "CLI: a bad run group -> exit 2")
    check(run(str(tmp / "bad.json"), "--dry-run", "--no-llm")[0] == 2, "CLI: not a lab -> exit 2")
    saved_default, lib.DEFAULT_LIBRARY = lib.DEFAULT_LIBRARY, tmp / "no_library"  # empty default
    try:
        code, text = run(str(lab_path), "--dry-run", "--no-llm")
    finally:
        lib.DEFAULT_LIBRARY = saved_default
    m = last_manifest()
    by = {a["name"]: a for a in m["assets"]}
    check(code == 0 and m["status"] == "audited" and "no asset library" in text
          and by["moveableSphere"]["status"] == "available" and by["textPrefab"]["library"]["builtin"]
          and by["dnaHelixPrefab"]["status"] == "unplanned",
          "CLI --no-llm: without a library, the seed list's built-ins are known; the rest is unplanned")
    behaviour = {"dnaDoubleHelixPrefab": {"same_as": "dnaHelixPrefab"},
                 "summaryPanelPrefab": {"route": "ui", "form": "ui"},
                 "heart_muscle_texture": {"route": "generate", "form": "material"},
                 "dnaHelixPrefab": {"route": "component", "form": "effect",
                                    "reason": "it grows as the strands separate"}}
    code, text = run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"),
                     call_fn=fake_router(behaviour))
    m = last_manifest()
    by = {a["name"]: a for a in m["assets"]}
    check(code == 0 and m["status"] == "planned" and behaviour["calls"] == 1
          and behaviour["reuse_calls"] == 1, "CLI: one reuse call and one routing call plan the lab")
    check("[model] moveableSphere" not in behaviour["prompt"] and "[model] dnaHelixPrefab" in behaviour["prompt"]
          and "[texture] 2k_mars" not in behaviour["prompt"],
          "assets the library has (by name) are not sent to the LLM")
    check(by["2k_mars"]["status"] == "available" and not by["2k_mars"]["library"]["builtin"],
          "a library texture is available")
    check(by["dnaDoubleHelixPrefab"]["plan"]["same_as"] == "dnaHelixPrefab" and by["summaryPanelPrefab"]["plan"]["route"] == "ui"
          and m["counts"]["by_route"] == {"component": 1, "generate": 1, "ui": 1},
          "planned, merged and per-route counts are recorded")
    check(by["dnaHelixPrefab"]["status"] == "flagged"
          and m["review_items"][0] == {"name": "dnaHelixPrefab", "kind": "model", "route": "component",
                                       "reason": "it grows as the strands separate", "description": "a dnaHelixPrefab",
                                       "objects": ["dnaHelix", "dnaHelix2"]}
          and "flagged for a reviewer" in text,
          "behaviour routed to 'component' is flagged for a reviewer, with its objects and reason")
    check(by["dnaDoubleHelixPrefab"]["status"] == "flagged"
          and m["review_items"][1]["same_as"] == "dnaHelixPrefab"
          and m["review_items"][1]["reason"] == "it grows as the strands separate"
          and m["review_items"][1]["objects"] == ["helixAgain"] and m["review_items"][1]["route"] == "component",
          "an asset merged into a flagged one is flagged too, so its objects reach the reviewer")
    check(m["routing"]["reask_restarts"] == 0 and m["routing"]["transient_retries"] == 0,
          "the routing record carries transient retries and reask restarts")
    check({"shape", "component", "impossible"} <= set(routing.routes_for("model"))
          and {"component", "impossible"} <= set(routing.routes_for("texture"))
          and not {"kit", "textured_primitive"} & set(routing.routes_for("model")),
          "routes: one 'shape' route (kit and textured_primitive merged), and the review routes")
    check(m["library"]["library_id"] == lib.LocalFolderStore(tmp / "libP3").identity()["library_id"]
          and m["source"]["labId"] == "tiny_lab" and m["provenance"]["git_commit"],
          "the manifest records the library identity, the source and provenance")
    lib.LocalFolderStore(tmp / "libP3").add_alias("2k_mars", "heart_muscle_texture")
    run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"), call_fn=fake_router({}))
    a = {x["name"]: x for x in last_manifest()["assets"]}["heart_muscle_texture"]
    check(a["status"] == "available" and a["library"]["matched"] == "alias", "an alias hit is available")
    code, _ = run(str(lab_path), "--dry-run", "--library", str(tmp / "libB"), "--no-reuse",
                  call_fn=fake_router({"raise": "the provider is down"}))
    m = last_manifest()
    check(code == 1 and m["status"] == "failed" and "provider is down" in m["routing"]["error"],
          "CLI: a failed call -> exit 1, recorded in the manifest")
    run(str(lab_path), "--dry-run", "--no-llm", "--library", str(tmp / "libB"))
    run(str(lab_path), "--dry-run", "--no-llm", "--library", str(tmp / "libB"))
    check(len(list(out.rglob("provisioning_manifest.json"))) == 6, "two runs in the same second get separate folders")
    # Libraries that can't be used are errors, not "no library"
    code, text = run(str(lab_path), "--dry-run", "--no-llm", "--library", str(tmp / "typo_library"))
    check(code == 2 and "no asset library there" in text, "CLI: a --library path with no library -> exit 2")
    broken_lib = tmp / "broken_library"
    broken_lib.mkdir()
    (broken_lib / lib.INDEX_NAME).write_text('{"schema_version": 99}', encoding="utf-8")
    code, text = run(str(lab_path), "--dry-run", "--no-llm", "--library", str(broken_lib))
    check(code == 2 and "not a valid library index" in text, "CLI: a broken library -> exit 2 with the real error")
    os.environ[lib.LIBRARY_ENV] = str(broken_lib)
    try:
        code, text = run(str(lab_path), "--dry-run", "--no-llm")
    finally:
        del os.environ[lib.LIBRARY_ENV]
    check(code == 2, "CLI: a broken library at the default location is an error too")
    os.environ[lib.LIBRARY_ENV] = str(tmp / "typo_library")
    try:
        code, text = run(str(lab_path), "--dry-run", "--no-llm")
    finally:
        del os.environ[lib.LIBRARY_ENV]
    check(code == 2 and f"${lib.LIBRARY_ENV}=" in text and "no asset library there" in text,
          f"CLI: ${lib.LIBRARY_ENV} pointing at a folder with no library -> exit 2")

    # Metrics files
    (tmp / "list.json").write_text("[1, 2]", encoding="utf-8")
    code, text = run(str(lab_path), "--dry-run", "--no-llm", "--library", str(tmp / "libP3"),
                     "--metrics", str(tmp / "list.json"))
    check(code == 2 and "not a metrics record" in text, "CLI: --metrics that isn't a JSON object -> exit 2")
    pair = tmp / "pair" / "Outputs"
    pair.mkdir(parents=True)
    (tmp / "pair" / "Metrics").mkdir()
    paired_lab = pair / "x_json_lab_L4_1_output.json"
    paired_lab.write_text(json.dumps(lab), encoding="utf-8")
    (pair / "x_json_lab_L4_1_metrics.json").write_text("{truncated", encoding="utf-8")
    (tmp / "pair" / "Metrics" / "x_json_lab_L4_1_metrics.json").write_text(
        json.dumps({"model": "m1", "level": "L4", "registry": {"invented_prefabs": {
            "summaryPanelPrefab": {"module": 1, "purpose": "recap"}}}}), encoding="utf-8")
    problems: list = []
    found = audit.paired_metrics(paired_lab, problems)
    check(found is not None and found["model"] == "m1" and len(problems) == 1 and "skipped" in problems[0],
          "an unreadable metrics file beside the output is skipped, and the one in Metrics/ is used")
    code, text = run(str(paired_lab), "--dry-run", "--no-llm", "--library", str(tmp / "libP3"))
    m = last_manifest()
    check(code == 0 and m["source"]["level"] == "L4" and m["source"]["metrics_problems"]
          and "skipped" in text, "CLI: the skipped metrics file is noted and recorded")

    import llm
    llm._benchmark()  # loads .env, as a real run would
    saved = {k: os.environ.pop(k) for ks in llm.API_KEY_ENV.values() for k in ks if k in os.environ}
    saved_default, lib.DEFAULT_LIBRARY = lib.DEFAULT_LIBRARY, tmp / "no_library"
    try:
        code, text = run(str(lab_path), "--dry-run", "--library", str(tmp / "libB"))
        check(code == 2 and "no API key" in text, "CLI: no API key -> exit 2, before any call")
        code, text = run(str(lab_path), "--dry-run", "--model", "no-such-model")
        check(code == 2 and "unknown model" in text, "CLI: an unknown model -> exit 2")
        only_builtins = tiny_lab()
        for module in only_builtins["modules"]:
            for o in module["objects"]:
                o["prefab"] = "tinySphere"
                o.pop("texture", None)
        nothing = tmp / "only_builtins_output.json"
        nothing.write_text(json.dumps(only_builtins), encoding="utf-8")
        code, text = run(str(nothing), "--dry-run", "--library", str(tmp / "libP3"))
        m = last_manifest()
        check(code == 0 and m["status"] == "planned" and m["routing"]["calls"] == 0,
              "CLI: a lab whose assets are all in the library needs no API key and makes no call")
        code, text = run(str(nothing), "--dry-run", "--library", str(tmp / "libP3"), "--model", "no-such-model")
        check(code == 2 and "unknown model" in text,
              "CLI: an unknown model -> exit 2, even when nothing needs planning")
    finally:
        os.environ.update(saved)
        lib.DEFAULT_LIBRARY = saved_default


# ── Part 4 ────────────────────────────────────────────────────────────


def part4(tmp: Path) -> None:
    print("Part 4: the reuse judge (#108)")
    import audit
    import provision_lab
    import reuse
    import routing

    lab = tiny_lab()
    lab_path = tmp / "tiny_lab_output.json"   # written by Part 3
    reqs = {(r.kind, r.key): r for r in audit.audit_lab(audit.load_lab(lab_path))}
    helix, panel = reqs[("model", "dnahelixprefab")], reqs[("model", "summarypanelprefab")]
    muscle = reqs[("texture", "heart_muscle_texture")]
    store = lib.LocalFolderStore(tmp / "libP3")
    pool = reuse.candidates(store.entries())
    check(all(e.reusable for e in pool) and not any(e.name == "demoPrefab" for e in pool),
          "candidates leave out system objects (demoPrefab)")

    # The judge's checks
    asked = [helix, panel, muscle]
    good = {"decisions": [
        {"name": "dnaHelixPrefab", "kind": "model", "decision": "textured_shape", "shape": "sphere",
         "texture_id": "2k_mars", "reason": "r"},
        {"name": "SUMMARYPANELPREFAB", "kind": "model", "decision": "reuse", "asset_id": "text_prefab",
         "adapt": "", "reason": "r"},
        {"name": "heart_muscle_texture", "kind": "texture", "decision": "none", "asset_id": "2k_mars",
         "reason": "r"}]}
    with reuse.reuse_scope(asked, pool):
        plan = reuse.ReusePlan.model_validate(good)
    by = {d.name: d for d in plan.decisions}
    check(by["summaryPanelPrefab"].asset_id == "text_prefab" and by["heart_muscle_texture"].asset_id == "",
          "decisions come back in the lab's spelling; 'none' clears ids")
    bad = copy.deepcopy(good)
    bad["decisions"][0].update(shape="pyramid")                       # not a shape
    bad["decisions"][1].update(asset_id="2k_mars")                    # a texture for a model
    bad["decisions"][2].update(decision="textured_shape", shape="sphere", texture_id="2k_mars")  # texture
    bad["decisions"].append({"name": "ghostPrefab", "kind": "model", "decision": "none", "reason": "r"})
    try:
        with reuse.reuse_scope(asked, pool):
            reuse.ReusePlan.model_validate(bad)
        check(False, "a bad reuse plan is refused")
    except lib.ValidationError as exc:
        msg = str(exc)
        check(all(x in msg for x in ("shape 'pyramid'", "'2k_mars' is a texture, not a model",
                                     "textured_shape is for models", "'ghostPrefab' is not one of")),
              "every problem in a reuse plan is reported in one error (one reask)")
    for change, what in [({"decision": "reuse", "asset_id": "no_such_id"}, "not a library id"),
                         ({"decision": "reuse", "asset_id": ""}, "needs the asset_id"),
                         ({"decision": "textured_shape", "shape": "sphere", "texture_id": ""}, "needs the texture_id"),
                         ({"decision": "reuse", "asset_id": "tiny_sphere", "texture_id": "text_prefab"},
                          "is a model, not a texture"),
                         ({"decision": "maybe"}, "is not one of reuse")]:
        one = copy.deepcopy(good)
        one["decisions"][0] = {"name": "dnaHelixPrefab", "kind": "model", "reason": "r", **change}
        try:
            with reuse.reuse_scope(asked, pool):
                reuse.ReusePlan.model_validate(one)
            check(False, f"refused: {what}")
        except lib.ValidationError as exc:
            check(what in str(exc), f"refused: {what}")
    texture_reuse = copy.deepcopy(good)
    texture_reuse["decisions"][2] = {"name": "heart_muscle_texture", "kind": "texture", "decision": "reuse",
                                     "asset_id": "2k_mars", "texture_id": "2k_mars", "reason": "r"}
    try:
        with reuse.reuse_scope(asked, pool):
            reuse.ReusePlan.model_validate(texture_reuse)
        check(False, "a reused texture can't take a texture_id")
    except lib.ValidationError as exc:
        check("only a reused model can take a texture_id" in str(exc), "a reused texture can't take a texture_id")
    schema = reuse.ReusePlan.model_json_schema()["$defs"]["ReuseDecision"]
    check(schema["required"] == ["name", "kind", "decision", "reason"]
          and "default" not in schema["properties"]["decision"],
          "the reuse tool schema stays strict while validation is lenient")

    # Shortlisting a big library
    many = [lib.LibraryEntry(asset_id=f"filler_{i}", kind="model", name=f"filler{i}", description="unrelated thing",
                             builtin=True) for i in range(30)]
    target = lib.LibraryEntry(asset_id="dna_double_helix", kind="model", name="dnaDoubleHelix",
                              description="a DNA double helix", tags=["dna", "helix"], builtin=True)
    saved_limit, saved_size = reuse.SHORTLIST_LIMIT, reuse.SHORTLIST_SIZE
    reuse.SHORTLIST_LIMIT, reuse.SHORTLIST_SIZE = 10, 3
    try:
        short = reuse.shortlist([helix], many + [target])
    finally:
        reuse.SHORTLIST_LIMIT, reuse.SHORTLIST_SIZE = saved_limit, saved_size
    check(len(short) == 3 and target in short, "a big library is shortlisted to each asset's best word matches")
    check(len(reuse.shortlist([helix], many)) == 30, "a small library is sent whole")
    # What students do with an asset's objects reaches both judges
    check(audit._behaviours([{"componentType": "pointerReceiver", "draggable": True},
                             {"type": "SimpleOrbitComponent"}, {"componentType": "newscript", "scriptName": "Spin"},
                             {"componentType": "textMeshPro", "text": "x"}])
          == ["pointerReceiver (draggable)", "simpleOrbit", "Spin"],
          "object components are named (saved or schema form, new scripts by name, draggable marked), text left out")
    ball_lab = {"modules": [{"moduleName": "M", "objects": [
        {"name": "marker", "prefab": "domainPrefab"},
        {"name": "dragMe", "prefab": "domainPrefab",
         "components": [{"componentType": "pointerReceiver", "draggable": True}]},
        {"name": "later", "prefab": "domainPrefab"}],
        "clips": [{"narration": "", "changes": [
            {"target": "later", "components": [{"componentType": "simpleOrbit"}]},
            {"target": "dragMe", "components": [{"componentType": "pointerReceiver", "draggable": False}]}]}]}]}
    dom = audit.audit_lab(ball_lab)[0]
    block = routing.asset_block(dom)
    check(dom.components == {"pointerReceiver (draggable)": 1, "simpleOrbit": 1}
          and dom.uses[1].components == ["pointerReceiver (draggable)"]
          and "components on its objects: pointerReceiver (draggable) on 1 of 3, simpleOrbit on 1 of 3" in block
          and 'object "dragMe" in module 1 "M", components pointerReceiver (draggable)' in block,
          "the asset block shows each object's components and a count over all its objects (clip changes too)")
    check(dom.uses[1].components == ["pointerReceiver (draggable)"],
          "an object draggable in any clip counts as draggable once (a change turning drag off isn't a second entry)")
    check(reuse.SHAPES == routing.HEADSET_PRIMITIVES and "torus" not in reuse.SHAPES,
          "a textured shape is one the headset makes today (Unity's primitives)")
    prompt = reuse.build_prompt(lab, asked, pool)
    check("- 2k_mars [texture] 2k_mars:" in prompt and "[model] dnaHelixPrefab" in prompt
          and "text_prefab [model, built-in]" in prompt, "the prompt lists library entries and the assets")

    # The CLI
    out = tmp / "prov_out4"

    def run(*args, call_fn=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = provision_lab.main(list(args), call_fn=call_fn, output_root=out)
        return code, buf.getvalue()

    def last_manifest():
        return json.loads(max(out.rglob("provisioning_manifest.json"), key=os.path.getmtime)
                          .read_text(encoding="utf-8"))

    glb = tmp / "board.glb"
    glb.write_bytes(b"glTF" + (2).to_bytes(4, "little") + (12).to_bytes(4, "little"))
    lib.LocalFolderStore(tmp / "libP3").put(lib.LibraryEntry(
        asset_id="summary_board", kind="model", name="summaryBoard", description="a board listing key ideas",
        bounds=[1, 0.6, 0.05], triangles=12, provenance=lib.Provenance(license="CC0-1.0", method="kit")),
        {"model": glb})
    builtin_reuse = reuse.ReusePlan(decisions=[reuse.ReuseDecision(
        name="marsPrefab", kind="model", decision="reuse", asset_id="tiny_sphere", reason="a red ball")])
    check(provision_lab.alias_proposals(builtin_reuse, lib.LocalFolderStore(tmp / "libP3").entries()) == [],
          "a reuse of a built-in (a stand-in) is not proposed as an alias")
    behaviour = {"reuse": {"summaryPanelPrefab": {"decision": "reuse", "asset_id": "summary_board",
                                                  "reason": "a panel of text is the summary board"},
                           "dnaHelixPrefab": {"decision": "textured_shape", "shape": "sphere",
                                              "texture_id": "2k_mars", "reason": "test"}}}
    code, text = run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"), call_fn=fake_router(behaviour))
    m = last_manifest()
    by = {a["name"]: a for a in m["assets"]}
    check(code == 0 and behaviour["reuse_calls"] == 1 and behaviour["calls"] == 1,
          "CLI: one reuse call, then one routing call")
    check("[model] moveableSphere" not in behaviour["reuse_prompt"]
          and "[model] summaryPanelPrefab" in behaviour["reuse_prompt"],
          "assets found by name are not sent to the reuse judge")
    check("[model] summaryPanelPrefab" not in behaviour["prompt"] and "[model] dnaHelixPrefab" not in behaviour["prompt"]
          and "[model] dnaDoubleHelixPrefab" in behaviour["prompt"],
          "reused assets are not sent to describe and route")
    check(by["summaryPanelPrefab"]["status"] == "reused" and by["summaryPanelPrefab"]["reuse"] ==
          {"decision": "reuse", "asset_id": "summary_board", "reason": "a panel of text is the summary board"}
          and by["dnaHelixPrefab"]["reuse"]["texture_id"] == "2k_mars"
          and m["counts"]["by_reuse"] == {"reuse": 1, "textured_shape": 1},
          "reused assets and their decisions are recorded")
    check(m["aliases"] == {"saved": False, "proposed": [{"name": "summaryPanelPrefab", "asset_id": "summary_board"}]}
          and "1 proposed" in text and m["reuse"]["shortlist"] > 0 and "reuse: claude-haiku-5.5" in text,
          "a single-entry reuse is proposed as an alias, not saved by default")
    check(by["dnaDoubleHelixPrefab"]["reuse"] == {"decision": "none", "reason": "test"}
          and by["dnaDoubleHelixPrefab"]["status"] == "planned",
          "the judge's 'none' and its reason are kept for a reviewer")
    check(by["dnaHelixPrefab"]["components"] == {} and "components" in by["summaryPanelPrefab"],
          "the manifest records each asset's component counts")
    merge = {"reuse": {"dnaHelixPrefab": behaviour["reuse"]["dnaHelixPrefab"]},
             "dnaDoubleHelixPrefab": {"same_as": "dnaHelixPrefab"}}
    code, text = run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"), call_fn=fake_router(merge))
    m = last_manifest()
    twin = {a["name"]: a for a in m["assets"]}["dnaDoubleHelixPrefab"]
    check(code == 0 and "ALREADY IN THE LIBRARY" in merge["prompt"] and "- dnaHelixPrefab (model)" in merge["prompt"]
          and twin["status"] == "reused" and twin["reuse"]["same_as"] == "dnaHelixPrefab"
          and twin["reuse"]["texture_id"] == "2k_mars" and "(same as dnaHelixPrefab)" in text,
          "a duplicate of a reused asset can be same_as it, and follows the library match")
    check("- dnaHelixPrefab (model): matched to sphere + 2k_mars; objects \"dnaHelix\", \"dnaHelix2\"" in merge["prompt"]
          and "Saturn is not Jupiter" in merge["prompt"],
          "describe and route sees what each reused asset was matched to, and its objects")
    check("plan" not in twin and twin["reuse"]["judge"] == {"decision": "none", "reason": "test"}
          and twin["reuse"]["routing_reason"] == "test"
          and m["counts"]["by_reuse"] == {"textured_shape": 1} and m["counts"]["by_form"] == {"object": 1},  # only the one planned asset
          "a duplicate following a reused asset has no plan to provide, keeps the judge's own verdict, "
          "and is counted once")
    flag = {"reuse": {"dnaHelixPrefab": behaviour["reuse"]["dnaHelixPrefab"]},
            "dnaDoubleHelixPrefab": {"same_as": "dnaHelixPrefab", "route": "component", "form": "effect"}}
    run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"), call_fn=fake_router(flag))
    m = last_manifest()
    twin = {a["name"]: a for a in m["assets"]}["dnaDoubleHelixPrefab"]
    check(twin["status"] == "flagged" and [i["name"] for i in m["review_items"]] == ["dnaDoubleHelixPrefab"],
          "behaviour merged into a reused asset is still flagged for a reviewer")
    try:
        with routing.routing_scope([reqs[("model", "dnadoublehelixprefab")]]):
            routing.LabAssetPlan.model_validate({"assets": [{
                "name": "dnaDoubleHelixPrefab", "kind": "model", "description": "d", "search_keywords": ["a", "b"],
                "form": "object", "route": "search", "reason": "r", "same_as": "dnaHelixPrefab"}]})
        check(False, "same_as a reused asset is refused when it isn't passed as reused")
    except lib.ValidationError as exc:
        check("not a listed or reused model" in str(exc),
              "same_as a reused asset is refused when it isn't passed as reused")
    rev = lib.LocalFolderStore(tmp / "libP3").identity()["revision"]
    code, text = run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"), "--save-aliases",
                     call_fn=fake_router(dict(behaviour)))
    m = last_manifest()
    store = lib.LocalFolderStore(tmp / "libP3")
    check(code == 0 and m["aliases"]["saved"] and m["aliases"]["proposed"][0]["saved"]
          and store.find("summaryPanelPrefab").asset_id == "summary_board"
          and store.identity()["revision"] == rev + 1 and m["aliases"]["library_after"]["revision"] == rev + 1,
          "--save-aliases saves the match as an alias and records the library after")
    again_saved = provision_lab.save_aliases(store, [{"name": "summaryPanelPrefab", "asset_id": "summary_board"}])
    check(again_saved == [{"name": "summaryPanelPrefab", "asset_id": "summary_board", "saved": True,
                           "already": True}]
          and store.identity()["revision"] == rev + 1,
          "an alias already there (another run saved it) counts as saved, with no write")
    again = {"reuse": {"dnaHelixPrefab": behaviour["reuse"]["dnaHelixPrefab"]}}
    code, text = run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"), call_fn=fake_router(again))
    a = {x["name"]: x for x in last_manifest()["assets"]}["summaryPanelPrefab"]
    check(code == 0 and a["status"] == "available" and a["library"]["matched"] == "alias"
          and "summaryPanelPrefab" not in again["reuse_prompt"],
          "on the next run the saved alias is found by name, with no judge call for it")
    nr = {}
    run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"), "--no-reuse", call_fn=fake_router(nr))
    check("reuse_calls" not in nr and nr["calls"] == 1 and last_manifest()["reuse"] is None,
          "--no-reuse skips the judge")
    code, text = run(str(lab_path), "--dry-run", "--no-llm", "--save-aliases", "--library", str(tmp / "libP3"))
    check(code == 2 and "needs the reuse judge" in text, "--save-aliases without the judge -> exit 2")
    saved_default, lib.DEFAULT_LIBRARY = lib.DEFAULT_LIBRARY, tmp / "no_library"
    try:
        code, text = run(str(lab_path), "--dry-run", "--save-aliases", call_fn=fake_router({}))
    finally:
        lib.DEFAULT_LIBRARY = saved_default
    check(code == 2 and "needs an asset library" in text, "--save-aliases without a library -> exit 2")
    down = {"raise": "the provider is down", "raise_in": "reuse"}
    code, text = run(str(lab_path), "--dry-run", "--library", str(tmp / "libP3"), call_fn=fake_router(down))
    m = last_manifest()
    check(code == 1 and m["status"] == "failed" and "provider is down" in m["reuse"]["error"]
          and m["routing"] is None and "calls" not in down and "the reuse judge failed" in text,
          "a failed reuse call -> exit 1, and nothing is routed")


# ── Part 5 ────────────────────────────────────────────────────────────


def part5(tmp: Path) -> None:
    print("Part 5: the real client path (mock Anthropic server)")
    try:
        import anthropic  # noqa: F401
        import instructor  # noqa: F401
    except ImportError:
        check(False, "Part 5 needs the project venv (anthropic, instructor)")
        return
    import provision_lab
    import reuse
    import routing

    answer = fake_router({"reuse": {"dnaHelixPrefab": {"decision": "textured_shape", "shape": "sphere",
                                                       "texture_id": "2k_mars", "reason": "test"}}})
    sent: list[dict] = []

    class MockAnthropic(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sent.append(body)
            tool = body["tools"][0]["name"]
            content = body["messages"][0]["content"]
            prompt = content if isinstance(content, str) else "".join(b.get("text", "") for b in content)
            model = reuse.ReusePlan if tool == "ReusePlan" else routing.LabAssetPlan
            data = answer(model, [{"role": "user", "content": prompt}], 1).model_dump()
            payload = json.dumps({
                "id": "msg_mock", "type": "message", "role": "assistant",
                # the reuse step is "served" by another model, to tell the steps apart
                "model": body["model"] + ("-other" if tool == "ReusePlan" else ""),
                "content": [{"type": "tool_use", "id": "toolu_mock", "name": tool, "input": data}],
                "stop_reason": "tool_use", "stop_sequence": None,
                "usage": {"input_tokens": 1000, "output_tokens": 200}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = http.server.HTTPServer(("127.0.0.1", 0), MockAnthropic)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved = {k: os.environ.get(k) for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY")}
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{srv.server_port}"
    os.environ["ANTHROPIC_API_KEY"] = "test"
    out = tmp / "prov_out5"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = provision_lab.main([str(tmp / "tiny_lab_output.json"), "--dry-run", "--library",
                                       str(tmp / "libP3"), "--model", "claude-haiku-5.5"], output_root=out)
    finally:
        srv.shutdown()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    m = json.loads(next(out.rglob("provisioning_manifest.json")).read_text(encoding="utf-8"))
    model_id = sent[0]["model"] if sent else ""
    check(code == 0 and [b["tools"][0]["name"] for b in sent] == ["ReusePlan", "LabAssetPlan"]
          and sent[0].get("tool_choice") == {"type": "auto", "disable_parallel_tool_use": True},
          "CLI on the real client: one reuse request, then one routing request, with the Anthropic guard")
    r, t = m["reuse"], m["routing"]
    check(r["calls"] == 1 and t["calls"] == 1 and r["prompt_tokens"] == t["prompt_tokens"] == 1000
          and r["completion_tokens"] == t["completion_tokens"] == 200,
          "each step records only its own calls and tokens")
    check(r["cost_usd"] > 0 and abs(r["cost_usd"] + t["cost_usd"] - m["llm"]["cost"]["total_usd"]) < 1e-6
          and m["llm"]["total_calls"] == 2,
          "the steps' costs add up to the run's tracker total")
    check(r["served_models"] == [model_id + "-other"] and t["served_models"] == [model_id],
          "each step records the models that served its own calls")


# ── Part 6 ────────────────────────────────────────────────────────────


def pc_record(elements, bonds, coords=None, charges=None) -> dict:
    """A PubChem PUG REST record (the fields the builder reads)."""
    aids = list(range(1, len(elements) + 1))
    comp = {"atoms": {"aid": aids, "element": elements},
            "bonds": {"aid1": [a + 1 for a, _, _ in bonds], "aid2": [b + 1 for _, b, _ in bonds],
                      "order": [o for _, _, o in bonds]}}
    if charges:
        comp["atoms"]["charge"] = [{"aid": a + 1, "value": v} for a, v in charges]
    if coords:
        comp["coords"] = [{"type": [2, 5, 255], "aid": aids, "conformers": [
            {"x": [c[0] for c in coords], "y": [c[1] for c in coords], "z": [c[2] for c in coords]}]}]
    return {"PC_Compounds": [comp]}


def part6(tmp: Path) -> None:
    print("Part 6: the molecule builder (#111), against a mock PubChem")
    try:
        import numpy as np
        import trimesh
    except ImportError:
        check(False, "Part 6 needs the project venv (trimesh)")
        return
    import audit
    import build_assets
    import builders
    import molecule as M
    import provision_lab
    import routing
    from urllib.parse import quote

    srv = Server()
    saved = (M.PUBCHEM, M.USE_RDKIT)
    M.PUBCHEM, M.USE_RDKIT = srv.base + "/rest/pug", False

    def serve(name, cid, title, formula, rec3d=None, rec2d=None, by_formula=None):
        js = lambda d: json.dumps(d).encode()  # noqa: E731
        key = f"/rest/pug/compound/fastformula/{by_formula}/cids/JSON" if by_formula \
            else f"/rest/pug/compound/name/{quote(name, safe='')}/cids/JSON"
        srv.files[key] = js({"IdentifierList": {"CID": [cid]}})
        srv.files[f"/rest/pug/compound/cid/{cid}/property/Title,MolecularFormula/JSON"] = js(
            {"PropertyTable": {"Properties": [{"CID": cid, "Title": title, "MolecularFormula": formula}]}})
        if rec3d:
            srv.files[f"/rest/pug/compound/cid/{cid}/record/JSON?record_type=3d"] = js(rec3d)
        srv.files[f"/rest/pug/compound/cid/{cid}/record/JSON?record_type=2d"] = js(rec2d or rec3d)

    t = 1.09 / 3 ** 0.5
    four = [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)]
    serve("methane", 297, "Methane", "CH4",
          rec3d=pc_record([6, 1, 1, 1, 1], four, [(0, 0, 0), (t, t, t), (t, -t, -t), (-t, t, -t), (-t, -t, t)]))
    serve("water", 962, "Water", "H2O",
          rec3d=pc_record([8, 1, 1], [(0, 1, 1), (0, 2, 1)], [(0, 0, 0), (0.757, 0.586, 0), (-0.757, 0.586, 0)]))
    serve("xenon tetrafluoride", 123324, "Xenon tetrafluoride", "F4Xe", rec2d=pc_record([54, 9, 9, 9, 9], four))
    serve("carbon dioxide", 280, "Carbon Dioxide", "CO2", rec2d=pc_record([6, 8, 8], [(0, 1, 2), (0, 2, 2)]))
    serve("ammonium", 223, "Ammonium", "H4N+", rec2d=pc_record([7, 1, 1, 1, 1], four, charges=[(0, 1)]))
    serve("SF6", 17358, "Sulfur Hexafluoride", "F6S", by_formula="SF6",
          rec2d=pc_record([16, 9, 9, 9, 9, 9, 9], [(0, k, 1) for k in range(1, 7)]))
    serve("argon", 23968, "Argon", "Ar", rec2d=pc_record([18], []))
    serve("boron tetrafluoride test", 9001, "Test dative", "BF4", rec2d=pc_record(
        [5, 9, 9, 9, 9], [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 5)]))
    serve("ammonia borane", 6328, "Ammonia borane", "BH6N", rec3d=pc_record(
        [5, 7, 1, 1, 1, 1, 1, 1], [(1, 0, 5), (0, 2, 1), (0, 3, 1), (0, 4, 1), (1, 5, 1), (1, 6, 1), (1, 7, 1)],
        [(0, 0, 0), (0, 0, 1.6), (1.1, 0, -0.4), (-0.55, 0.95, -0.4), (-0.55, -0.95, -0.4),
         (0.95, 0, 2.0), (-0.48, 0.82, 2.0), (-0.48, -0.82, 2.0)]))
    serve("ethane", 6324, "Ethane", "C2H6", rec2d=pc_record(
        [6, 6, 1, 1, 1, 1, 1, 1], [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (1, 5, 1), (1, 6, 1), (1, 7, 1)]))

    def build(name, *options, folder=None):
        req = builders.BuildRequest(name=name, kind="model", input=name, options=tuple(options))
        return M.build(req, tmp / "mol" / (folder or name.replace(" ", "_")))

    def angle(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))

    try:
        srv.fail_once["/rest/pug/compound/name/methane/cids/JSON"] = 503
        r = build("methane")
        check(r.ok and r.details["geometry"] == "pubchem_3d" and r.details["shape"] == "tetrahedral"
              and r.details["cid"] == 297 and srv.hits["/rest/pug/compound/name/methane/cids/JSON"] == 2,
              "methane builds from PubChem's 3D record (a 503 from PubChem is retried)")
        glb = trimesh.load(r.files["model"], file_type="glb")
        colors = {n: tuple(int(x) for x in g.visual.material.baseColorFactor[:3]) for n, g in glb.geometry.items()}
        check(colors == {"C": (144, 144, 144), "H": (255, 255, 255)}
              and abs(max(glb.extents) - 1.0) < 1e-6 and np.allclose(glb.bounds.mean(axis=0), 0, atol=1e-6)
              and sum(len(g.faces) for g in glb.geometry.values()) == r.details["triangles"],
              "the GLB has one CPK-colored material per element, fits a unit cube and is centered")
        again = build("methane", folder="methane_again")
        check(again.files["model"].read_bytes() == r.files["model"].read_bytes()
              and again.files["thumbnail"].read_bytes() == r.files["thumbnail"].read_bytes(),
              "the same PubChem data gives byte-identical files")
        check(srv.hits["/rest/pug/compound/name/methane/cids/JSON"] == 2,
              "PubChem replies are cached: building methane again asks nothing")
        rng = np.random.default_rng(7)
        clouds = [rng.normal(size=(9, 3)) * [3, 2, 1] for _ in range(20)]
        check(all(np.linalg.det(M.main_axes(c)) > 0 and np.linalg.det(M.main_axes(c * [-1, 1, 1])) > 0
                  for c in clouds),
              "the thumbnail's view frame is always right-handed (never a mirror image)")
        standard = r.details["triangles"]
        saved_budget = lib.MAX_MODEL_TRIANGLES
        try:
            lib.MAX_MODEL_TRIANGLES = standard - 1
            small = build("methane", folder="methane_small")
            lib.MAX_MODEL_TRIANGLES = 10
            none = build("methane", folder="methane_none")
        finally:
            lib.MAX_MODEL_TRIANGLES = saved_budget
        check(small.ok and small.details["detail"] == "reduced" and small.details["triangles"] < standard
              and not none.ok and "even at reduced detail" in none.reason
              and r.details["detail"] == "standard",
              "over the triangle budget: rebuilt coarser, refused if still over")
        mol_lib = lib.LocalFolderStore(tmp / "libMol", create=True)
        mol_lib.put(lib.LibraryEntry(**r.entry), r.files)
        stored = mol_lib.get("pubchem_297")
        check(stored.provenance.license == "public-domain" and stored.provenance.record_id == "297"
              and stored.name == "methane_molecule" and not [p for p in lib.check_index(mol_lib.index, mol_lib.root)
                                                              if p.level == "error"],
              "the result is a valid library entry with PubChem provenance and a thumbnail")

        r = build("xenon tetrafluoride", "lone_pairs")
        mol, has_3d = M.fetch(123324)
        pairs, shape = M.idealize(mol)
        xe, fs = np.asarray(mol.coords[0]), [np.asarray(c) for c in mol.coords[1:]]
        angles = sorted(round(angle(a - xe, b - xe)) for i, a in enumerate(fs) for b in fs[i + 1:])
        normal = max((np.cross(fs[0] - xe, f - xe) for f in fs[1:]), key=np.linalg.norm)
        check(r.ok and not has_3d and r.details["geometry"] == "idealized_vsepr" and shape == "square planar"
              and angles == [90, 90, 90, 90, 180, 180] and r.details["lone_pairs"] == {"Xe1": 2}
              and all(abs(abs(np.dot(d, normal / np.linalg.norm(normal))) - 1) < 1e-9 for d in pairs[0]),
              "XeF4 (no 3D record) gets an idealized square-planar geometry, its two lone pairs on the axis")
        glb = trimesh.load(r.files["model"], file_type="glb")
        check("lone_pair" in glb.geometry and glb.geometry["lone_pair"].visual.material.alphaMode == "BLEND",
              "lone pairs are translucent lobes")

        r = build("water", "lone_pairs")
        mol, _ = M.fetch(962)
        o = np.asarray(mol.coords[0])
        bonds = [np.asarray(c) - o for c in mol.coords[1:]]
        lps = M.free_directions(bonds, 2)
        check(r.ok and r.details["shape"] == "bent" and r.details["lone_pairs"] == {"O1": 2}
              and 95 < angle(*lps) < 135 and all(np.dot(d, sum(bonds)) < 0 for d in lps),
              "water's two lone pairs point away from its hydrogens, roughly tetrahedrally")
        check(not build("water", folder="water_plain").details["lone_pairs_drawn"],
              "lone pairs are drawn only when asked")

        r = build("carbon dioxide", "lone_pairs")
        check(r.ok and r.details["shape"] == "linear" and r.details["bond_orders"] == [2, 2]
              and r.details["lone_pairs"] == {},
              "CO2: linear, double bonds, and no lobes on the terminal oxygens")
        r = build("ammonium", "lone_pairs")
        check(r.ok and r.details["shape"] == "tetrahedral" and r.details["lone_pairs"] == {},
              "a charge counts toward lone pairs (NH4+ has none)")
        r = build("SF6")
        check(r.ok and r.details["matched_by"] == "formula" and r.details["shape"] == "octahedral",
              "a formula is looked up when no compound has that name")
        r = build("ethane")
        check(not r.ok and "more than one central atom" in r.reason and not (tmp / "mol" / "ethane").exists(),
              "no 3D record, several centers and no RDKit -> can't build, with the reason, and no files")
        try:
            import rdkit  # noqa: F401
            have_rdkit = True
        except ImportError:
            have_rdkit = False
        if have_rdkit:
            M.USE_RDKIT = True
            try:
                r = build("ethane", folder="ethane_rdkit")
                mol, _ = M.fetch(6324)
                M._rdkit_coords(mol)
                cc = float(np.linalg.norm(np.subtract(mol.coords[0], mol.coords[1])))
            finally:
                M.USE_RDKIT = False
            check(r.ok and r.details["geometry"] == "rdkit" and 1.45 < cc < 1.6,
                  "with RDKit installed, a multi-center molecule with no 3D record is embedded (C-C ~1.53 A)")
        else:
            print("  skip RDKit isn't installed: its fallback wasn't exercised")
        r = build("argon")
        check(r.ok and r.details["geometry"] == "single_atom" and r.details["atoms"] == 1,
              "a single atom is one ball")
        r = build("boron tetrafluoride test")
        check(not r.ok and "dative bonds" in r.reason and "the RDKit fallback is turned off" in r.reason,
              "an idealized geometry refuses a dative bond, and the reason says why RDKit didn't help")
        if have_rdkit:
            M.USE_RDKIT = True
            try:
                r = build("boron tetrafluoride test", folder="bf4_rdkit")
            finally:
                M.USE_RDKIT = False
            check(not r.ok and "RDKit couldn't embed it" in r.reason and "isn't installed" not in r.reason,
                  "an RDKit failure is reported as one, not as RDKit missing")
        octahedron = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
        tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
        line = np.array([[0, 0, 0], [0, 0, 1.2], [0, 0, -1.2]], float)
        check(np.allclose(np.abs(M.main_axes(octahedron)), np.eye(3))
              and np.allclose(np.abs(M.main_axes(tetra)), np.eye(3))
              and np.allclose(np.abs(M.main_axes(line)[0]), [0, 0, 1])
              and np.allclose(np.abs(M.main_axes(line)[1:]), np.eye(3)[:2]),
              "equal spreads take the world axes, so every machine picks the same thumbnail view")
        r = build("ammonia borane", "lone_pairs")
        check(r.ok and r.details["unusual_bonds"] == ["dative"] and r.details["lone_pairs"] == {}
              and r.details["bond_orders"] == [1] * 7,
              "a dative bond (3D record) is one stick, reported, and its atoms get no lone pairs")
        r = build("unobtainium")
        check(not r.ok and "no compound named" in r.reason, "an unknown compound -> can't build")

        # The builder route
        mol_route = routing.routes_for("model").get("builder:molecule", "")
        check("PubChem" in mol_route and "Options: lone_pairs" in mol_route
              and "builder:molecule" not in routing.routes_for("texture") and builders.get("builder:molecule"),
              "the molecule builder is offered as a model route, with its options")
        lab = tiny_lab()
        lab_path = tmp / "tiny_lab_output.json"
        reqs = {(r.kind, r.key): r for r in audit.audit_lab(audit.load_lab(lab_path))}
        helix = reqs[("model", "dnahelixprefab")]
        base = {"name": "dnaHelixPrefab", "kind": "model", "description": "d", "search_keywords": ["a", "b"],
                "form": "structure", "route": "builder:molecule", "reason": "r"}
        try:
            with routing.routing_scope([helix]):
                routing.LabAssetPlan.model_validate({"assets": [{**base, "builder_options": ["wireframe"]}]})
            check(False, "a builder route without its input, or with an unknown option, is refused")
        except lib.ValidationError as exc:
            check("needs builder_input" in str(exc) and "no option 'wireframe'" in str(exc),
                  "a builder route without its input, or with an unknown option, is refused")
        with routing.routing_scope([helix]):
            plain = routing.LabAssetPlan.model_validate({"assets": [{
                **base, "route": "search", "form": "object", "builder_input": "x", "builder_options": ["y"]}]})
        check(plain.assets[0].builder_input is None and plain.assets[0].builder_options == [],
              "builder fields are dropped from a route that isn't a builder")

        # build_assets.py: plan -> build -> save -> found by name next time
        out = tmp / "prov_out6"
        library_path = str(tmp / "libMol")

        def run(mod, *args, call_fn=None):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if mod is provision_lab:
                    code = mod.main(list(args), call_fn=call_fn, output_root=out)
                else:
                    code = mod.main(list(args))
            return code, buf.getvalue()

        plan = {"dnaHelixPrefab": {"route": "builder:molecule", "form": "structure", "builder_input": "methane",
                                   "builder_options": ["lone_pairs"]},
                "dnaDoubleHelixPrefab": {"same_as": "dnaHelixPrefab"},
                "summaryPanelPrefab": {"route": "builder:molecule", "form": "structure",
                                       "builder_input": "unobtainium"}}
        code, text = run(provision_lab, str(lab_path), "--dry-run", "--no-reuse", "--library", library_path,
                         call_fn=fake_router(plan))
        run_dir = max(out.glob("*"), key=os.path.getmtime)
        check(code == 0 and "builder:molecule" in text and "'methane' [lone_pairs]" in text,
              "a dry run routes an asset to the builder and prints its input and options")
        code, text = run(build_assets, str(run_dir), "--only", "nothere")
        check(code == 2, "build_assets: --only with an asset not routed to a builder -> exit 2")
        code, text = run(build_assets, str(run_dir), "--save", "--library", library_path)
        res = json.loads((run_dir / "build_results.json").read_text(encoding="utf-8"))
        by = {a["name"]: a for a in res["assets"]}
        check(code == 1 and res["counts"] == {"assets": 2, "built": 1, "failed": 1, "saved": 1}
              and (run_dir / "built" / "dnaHelixPrefab" / "model.glb").is_file()
              and by["dnaHelixPrefab"]["files"]["model"] == "built/dnaHelixPrefab/model.glb"
              and "FAILED" in text and "no compound named" in by["summaryPanelPrefab"]["reason"],
              "build_assets builds what the plan routed to a builder, records failures, exit 1 if any")
        entry = lib.LocalFolderStore(tmp / "libMol").get("pubchem_297")
        check(by["dnaHelixPrefab"]["saved"] == {"asset_id": "pubchem_297", "added": False, "replaced": False,
                                                "aliases": ["dnaHelixPrefab", "dnaDoubleHelixPrefab"]}
              and {"dnaHelixPrefab", "dnaDoubleHelixPrefab"} <= set(entry.aliases),
              "--save adds the lab's names (and the merged one) as aliases of the entry already there")
        store6 = lib.LocalFolderStore(tmp / "libMol")
        def set_entry(**fields):
            st = lib.LocalFolderStore(tmp / "libMol")
            e = st.get("pubchem_297")
            prov = {k: v for k, v in fields.items() if k in ("builder", "builder_version", "notes")}
            rest = {k: v for k, v in fields.items() if k not in prov}
            st.put(e.model_copy(update={"provenance": e.provenance.model_copy(update=prov), **rest}))

        def save_again():
            code, _ = run(build_assets, str(run_dir), "--save", "--library", library_path, "--only", "dnaHelixPrefab")
            res = json.loads((run_dir / "build_results.json").read_text(encoding="utf-8"))
            return code, res["assets"][0], lib.LocalFolderStore(tmp / "libMol").get("pubchem_297")

        set_entry(builder_version=1, name="methane_old_name")
        code, a, entry = save_again()
        check(code == 0 and a["saved"]["replaced"] and entry.provenance.builder_version == M.VERSION
              and entry.provenance.builder == "molecule" and entry.name == "methane_molecule"
              and {"dnaHelixPrefab", "dnaDoubleHelixPrefab", "methane_old_name"} <= set(entry.aliases),
              "an entry made by an older builder version is replaced, keeping its aliases and old name")
        set_entry(builder=None, builder_version=None)
        code, a, entry = save_again()
        check(a["saved"]["replaced"] and entry.provenance.builder_version == M.VERSION,
              "an entry saved before versions were recorded counts as older")
        set_entry(builder_version=M.VERSION + 1)
        code, a, entry = save_again()
        check(not a["saved"]["replaced"] and entry.provenance.builder_version == M.VERSION + 1
              and "a newer builder version" in " ".join(a["save_notes"]),
              "an entry from a newer builder version is kept, with a note")
        set_entry(builder_version=1, review="reviewed")
        code, a, entry = save_again()
        check(not a["saved"]["replaced"] and entry.review == "reviewed" and entry.provenance.builder_version == 1
              and "a reviewer reviewed it" in " ".join(a["save_notes"]),
              "a reviewed entry is never replaced, with a note")
        set_entry(builder_version=M.VERSION, review="auto")
        code, a, entry = save_again()
        check(not a["saved"]["replaced"] and "save_notes" not in a,
              "an entry from this builder version just gains aliases, silently")
        twin_manifest = tmp / "twins" / MANIFEST_NAME
        twin_manifest.parent.mkdir()
        twin_plan = {"route": "builder:molecule", "builder_input": "methane", "builder_options": []}
        twin_manifest.write_text(json.dumps({"tool": "provision_lab", "assets": [
            {"name": n, "kind": "model", "status": "planned", "plan": twin_plan}
            for n in ("CO2 Prefab", "co2_prefab")]}), encoding="utf-8")
        code, _ = run(build_assets, str(twin_manifest.parent))
        twins = json.loads((twin_manifest.parent / "build_results.json").read_text(encoding="utf-8"))
        check(code == 0 and [a["files"]["model"] for a in twins["assets"]]
              == ["built/CO2_Prefab/model.glb", "built/co2_prefab_2/model.glb"]
              and all((twin_manifest.parent / a["files"]["model"]).is_file() for a in twins["assets"]),
              "two names that make the same folder (ignoring case) get folders of their own")
        stale_dir = run_dir / "built" / "summaryPanelPrefab"
        stale_dir.mkdir(parents=True, exist_ok=True)
        (stale_dir / "model.glb").write_bytes(b"old")
        failed = build_assets.build_one({"name": "summaryPanelPrefab", "kind": "model", "merged_names": [],
                                         "plan": {"route": "builder:molecule", "builder_input": "unobtainium"}},
                                        run_dir / "built")
        check(not failed["ok"] and not stale_dir.exists(),
              "a failed rebuild leaves no earlier files behind")
        saved_defaults = builders.DEFAULT_BUILDERS
        builders._defaults_loaded, builders.DEFAULT_BUILDERS = False, ("no_such_builder_module",)
        try:
            builders.registry()
            check(False, "a default builder that fails to import is retried on the next call")
        except ImportError:
            builders.DEFAULT_BUILDERS = saved_defaults
            check(builders.get("molecule") is not None and builders._defaults_loaded,
                  "a default builder that fails to import is retried on the next call")
        finally:
            builders.DEFAULT_BUILDERS = saved_defaults
        code, text = run(provision_lab, str(lab_path), "--dry-run", "--no-reuse", "--library", library_path,
                         call_fn=fake_router({}))
        m = json.loads(next(max(out.glob("*"), key=os.path.getmtime).glob(MANIFEST_NAME)).read_text(encoding="utf-8"))
        a = {x["name"]: x for x in m["assets"]}
        check(a["dnaHelixPrefab"]["status"] == "available" and a["dnaHelixPrefab"]["library"]["asset_id"]
              == "pubchem_297" and a["dnaDoubleHelixPrefab"]["status"] == "available",
              "the next dry run finds the built molecule by the lab's names")
    finally:
        srv.close()
        M.PUBCHEM, M.USE_RDKIT = saved


MANIFEST_NAME = "provisioning_manifest.json"


# ── Part 7 ────────────────────────────────────────────────────────────


class Routes:
    """A mock HTTP server answering by path and query: ``routes[(path, frozenset(query))]``
    or ``routes[path]`` (any query) -> (status, bytes)."""

    def __init__(self):
        self.routes: dict = {}
        self.hits: list[str] = []
        self.headers: list[tuple[str, dict]] = []   # (path, request headers)
        outer = self
        from urllib.parse import parse_qsl, urlsplit

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.append(self.path)
                outer.headers.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
                u = urlsplit(self.path)
                from urllib.parse import unquote
                path = unquote(u.path)
                q = frozenset(parse_qsl(u.query))
                hit = outer.routes.get((path, q), outer.routes.get(path))
                status, body = hit if hit else (404, b"{}")
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def json(self, path, data, query=None, status=200):
        key = (path, frozenset(query.items())) if query is not None else path
        self.routes[key] = (status, json.dumps(data).encode())

    def file(self, path, body, status=200):
        self.routes[path] = (status, body)

    def close(self):
        self.httpd.shutdown()


def part7(tmp: Path) -> None:
    print("Part 7: search (#110), against mock Poly Haven, GitHub, NASA, ambientCG, Poly Pizza, Smithsonian")
    try:
        import trimesh
    except ImportError:
        check(False, "Part 7 needs the project venv (trimesh)")
        return
    import search_assets as S
    import sources as src

    srv = Routes()
    saved = (src.POLYHAVEN_API, src.GITHUB_API, src.GITHUB_RAW, src.GITHUB_MEDIA, src.AMBIENTCG_API,
             src.POLYPIZZA_API, src.POLYPIZZA_HOSTS, src.SMITHSONIAN_API, lib.MAX_MODEL_TRIANGLES)
    saved_key = os.environ.pop(src.POLYPIZZA_KEY_ENV, None)
    saved_si_key = os.environ.pop(src.SMITHSONIAN_KEY_ENV, None)
    src.SMITHSONIAN_API = srv.base + "/si"
    src.POLYHAVEN_API, src.GITHUB_API = srv.base + "/ph", srv.base + "/gh"
    src.AMBIENTCG_API = srv.base + "/acg"
    src.POLYPIZZA_API, src.POLYPIZZA_HOSTS = srv.base + "/pp", ("127.0.0.1",)
    src.GITHUB_RAW, src.GITHUB_MEDIA = srv.base + "/raw", srv.base + "/media"
    src.SOURCES["polyhaven"].__init__()
    src.SOURCES["nasa"].__init__()

    def glb(extents, triangles_x=1):
        mesh = trimesh.creation.box(extents=extents)
        if triangles_x > 1:
            mesh = trimesh.creation.icosphere(subdivisions=triangles_x)
        return trimesh.Scene(mesh).export(file_type="glb")

    # Poly Haven: two models and two textures
    heart = trimesh.Scene(trimesh.creation.icosphere(subdivisions=2, radius=3.0))
    gltf = heart.export(file_type="gltf")   # model.gltf + gltf_buffer_*.bin
    for name, body in gltf.items():
        srv.file(f"/cdn/human_heart/{name}", body)
    srv.json("/ph/assets", {
        "human_heart": {"name": "Human Heart", "tags": ["anatomy", "heart", "organ"], "categories": ["props"],
                        "authors": {"Jane Doe": "All"}, "thumbnail_url": srv.base + "/thumb/human_heart.png"},
        "heart_shaped_box": {"name": "Heart Shaped Box", "tags": ["box", "valentine"], "authors": {"A": "All"},
                             "thumbnail_url": srv.base + "/thumb/heart_shaped_box.png"},
        "office_chair": {"name": "Office Chair", "tags": ["chair", "furniture"], "authors": {"B": "All"},
                         "thumbnail_url": srv.base + "/thumb/office_chair.png"}},
        query={"type": "models", "t": "models"})
    srv.json("/ph/assets", {
        "red_brick": {"name": "Red Brick Wall", "tags": ["brick", "wall", "red"], "authors": {"C": "All"},
                      "thumbnail_url": srv.base + "/thumb/red_brick.png"}},
        query={"type": "textures", "t": "textures"})
    srv.json("/ph/search", {"results": [{"slug": "human_heart", "score": 0.91},
                                        {"slug": "heart_shaped_box", "score": 0.62},
                                        {"slug": "office_chair", "score": 0.2}], "total": 3},
             query={"q": "human heart anatomy", "t": "models", "limit": "15"})
    srv.json("/ph/files/human_heart", {"gltf": {"1k": {"gltf": {
        "url": srv.base + "/cdn/human_heart/model.gltf",
        "include": {n: {"url": srv.base + f"/cdn/human_heart/{n}"} for n in gltf if n != "model.gltf"}}}}})
    srv.file("/cdn/box.glb", glb([1, 1, 1]))
    srv.json("/ph/files/heart_shaped_box", {"gltf": {"1k": {"gltf": {"url": srv.base + "/cdn/box.glb",
                                                                    "include": {}}}}})
    srv.file("/cdn/brick.jpg", jpeg_bytes(3000, 1500, (180, 40, 30)))
    srv.json("/ph/files/red_brick", {"Diffuse": {"4k": {"jpg": {"url": srv.base + "/cdn/brick.jpg"}}},
                                     "nor_gl": {"4k": {"jpg": {"url": srv.base + "/cdn/none.jpg"}}}})
    for slug in ("human_heart", "heart_shaped_box", "office_chair", "red_brick"):
        srv.file(f"/thumb/{slug}.png", png_bytes())
    # NASA: the tree, a GLB stored with Git LFS, and its picture
    srv.json("/gh/repos/nasa/NASA-3D-Resources/git/trees/master", {"tree": [
        {"path": "3D Models/Saturn V/Saturn V.glb", "type": "blob", "size": 2000},
        {"path": "3D Models/Saturn V/Saturn V.png", "type": "blob"},
        {"path": "3D Models/Mars Rover/rover.obj", "type": "blob"},
        {"path": "README.md", "type": "blob"}], "truncated": False}, query={"recursive": "1"})
    srv.file("/raw/nasa/NASA-3D-Resources/master/3D Models/Saturn V/Saturn V.glb",
             b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 2000\n")
    srv.file("/media/nasa/NASA-3D-Resources/master/3D Models/Saturn V/Saturn V.glb", glb([0.4, 0.4, 4.0]))
    srv.file("/raw/nasa/NASA-3D-Resources/master/3D Models/Saturn V/Saturn V.png", png_bytes())

    # ambientCG: the reply shape of /api/v3/assets (2026-11-19), and a zip of maps
    import zipfile as _zip
    buf = io.BytesIO()
    with _zip.ZipFile(buf, "w") as z:
        z.writestr("Bricks090_2K-JPG_NormalGL.jpg", jpeg_bytes(64, 64, (128, 128, 255)))
        z.writestr("Bricks090_2K-JPG_Color.jpg", jpeg_bytes(96, 48, (170, 60, 40)))
        z.writestr("../evil_Color.txt", b"not read")
    srv.file("/acg/get", buf.getvalue())
    srv.file("/thumb/Bricks090.png", png_bytes())
    srv.json("/acg/assets", {"totalResults": 2, "nextPageHttp": None, "assets": [
        {"id": "Bricks090", "title": "Bricks 090", "tags": ["brick", "red", "wall"],
         "url": "https://ambientcg.com/a/Bricks090",
         "downloads": [{"attributes": "1K-JPG", "extension": "zip", "url": srv.base + "/acg/get?file=x1", "size": 9},
                       {"attributes": "2K-JPG", "extension": "zip", "url": srv.base + "/acg/get?file=x2", "size": 34},
                       {"attributes": "8K-PNG", "extension": "zip", "url": srv.base + "/acg/huge", "size": 10 ** 9}],
         "thumbnails": {"64-PNG": "x", "256-PNG": srv.base + "/thumb/Bricks090.png"},
         "maps": ["color", "displacement", "normal", "roughness"], "dimensions": {"width": 280, "height": 280}},
        {"id": "BrickDecal001", "title": "Brick Decal", "tags": ["brick"], "url": "u",
         "downloads": [{"attributes": "1K-PNG", "extension": "zip", "url": "u", "size": 1}],
         "thumbnails": {}, "maps": ["opacity"]}]})

    # Poly Pizza: the /search/{keyword} reply shape of the v1.1 spec
    srv.file("/static/heart.glb", glb([1, 2, 1]))
    srv.file("/static/heart.webp", png_bytes())
    pp_model = lambda **kw: {"ID": "x", "Title": "t", "Attribution": "a", "Thumbnail": srv.base + "/static/heart.webp",  # noqa: E731
                             "Download": srv.base + "/static/heart.glb", "Tri Count": 900,
                             "Creator": {"Username": "drx", "DPURL": ""}, "Category": "Objects",
                             "Tags": ["heart"], "Licence": "CC-BY 3.0", "Animated": False, **kw}
    srv.json("/pp/search/heart", {"total": 3, "results": [
        pp_model(ID="H1", Title="Anatomical Heart", Tags=["heart", "anatomy", "organ"],
                 Attribution='"Anatomical Heart" by drx, https://poly.pizza/m/H1. Licence at https://creativecommons.org/licenses/by/3.0'),
        pp_model(ID="H2", Title="Heart Emoji", Licence="CC0 1.0", **{"Tri Count": 10 ** 6}),
        pp_model(ID="H3", Title="Heart Pillow", Licence="CC-BY-NC 4.0")]})

    # Smithsonian: the /search reply shape (2026-11-23 sample): 3d_voyager media with resources
    def si_media(vid, files, access="CC0"):
        return {"id": f"3d_package:{vid}", "type": "3d_voyager", "voyagerId": f"3d_package:{vid}",
                "usage": {"access": access, "text": "Thanks to a sponsor."},
                "thumbnail": srv.base + f"/3d/{vid}/thumb.jpg",
                "resources": [{"url": srv.base + f"/3d/{vid}/{name}", "filename": name, "category": "Low Resolution",
                               "attributes": [{"MODEL_FILE_TYPE": kind, "DRACO_COMPRESSED": draco}, {"FILE_SIZE": size}]}
                              for name, kind, draco, size in files]}
    def si_row(title, media):
        return {"id": "ld1-" + title, "title": title, "unitCode": "NMNH", "type": "edanmdm",
                "content": {"freetext": {"notes": [{"label": "Description", "content": f"A scan of a {title}."}]},
                            "indexedStructured": {"topic": ["Anatomy"], "object_type": ["specimen"],
                                                  "online_media_type": ["Images"]},
                            "descriptiveNonRepeating": {"record_link": f"https://collections.si.edu/{title}",
                                                        "data_source": "NMNH - Anthropology",
                                                        "online_media": {"media": [
                                                            {"type": "Images", "usage": {"access": "CC0"}}, media]}}}}
    srv.json("/si/search", {"status": 200, "response": {"rowCount": 3, "rows": [
        si_row("Heart Specimen", si_media("big", [("heart-150k-4096-gltf_std.zip", "GLTF", False, 6),
                                                  ("heart-100k-2048_std_draco.glb", "GLB", True, 1),
                                                  ("heart-150k-4096_std.glb", "GLB", False, 8)])),
        si_row("Human Heart Model", si_media("small", [("model-40k-2048_std.glb", "GLB", False, 3),
                                                       ("model-40k-2048_std_draco.glb", "GLB", True, 1)])),
        si_row("Ritual ewer (<em>gong</em>) &amp; heart", si_media("ewer", [("e-10k-1024_std.glb", "GLB", False, 1)])),
        si_row("Heart Drawing", si_media("nc", [("d-10k-1024_std.glb", "GLB", False, 1)],
                                         access="Usage conditions apply"))]}})
    srv.file("/3d/small/model-40k-2048_std.glb", glb([1, 1, 2]))
    for vid in ("big", "small", "nc", "ewer"):
        srv.file(f"/3d/{vid}/thumb.jpg", jpeg_bytes(32, 32))
    srv.routes["/si-broken/search"] = (500, b"{}")

    # A plan with four search assets (one merged duplicate)
    run_dir = tmp / "search_run"
    run_dir.mkdir()
    def planned(name, kind, keywords, description, form="object", status="planned", same_as=None):
        plan = {"route": "search", "form": form, "description": description, "search_keywords": keywords,
                "must_be_right": ["four chambers"] if name == "heartPrefab" else []}
        if same_as:
            plan["same_as"] = same_as
        return {"name": name, "kind": kind, "status": status, "plan": plan}
    (run_dir / MANIFEST_NAME).write_text(json.dumps({"tool": "provision_lab", "assets": [
        planned("heartPrefab", "model", ["human heart anatomy"], "A human heart seen from the front", "organism"),
        planned("heartModel2", "model", ["heart"], "same", status="merged", same_as="heartPrefab"),
        planned("brickTexture", "texture", ["red brick"], "A red brick wall surface", "real_surface"),
        planned("rocketPrefab", "model", ["saturn v rocket"], "The Saturn V rocket"),
        planned("unicornPrefab", "model", ["unicorn"], "A unicorn")]}), encoding="utf-8")

    fits = {"Human Heart", "Red Brick Wall", "Saturn V", "Bricks 090", "Anatomical Heart", "Human Heart Model",
            "Heart Specimen"}
    seen = {"calls": 0, "prompts": []}

    def ranker(response_model, messages, max_retries):
        seen["calls"] += 1
        content = messages[0]["content"]
        seen["prompts"].append(content)
        names = [re.match(r"Candidate (\d+): (.+?) \(from", c).groups() for c in content
                 if isinstance(c, str) and c.startswith("Candidate ")]
        order = sorted(names, key=lambda x: (x[1] not in fits, int(x[0])))
        return response_model.model_validate({"ranking": [
            {"number": int(n), "fits": nm in fits, "reason": "test"} for n, nm in order]})

    def run(*args, call_fn=ranker):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = S.main([str(run_dir), *args], call_fn=call_fn)
        res = json.loads((run_dir / "search_results.json").read_text(encoding="utf-8"))
        return code, buf.getvalue(), {a["name"]: a for a in res["assets"]}, res

    try:
        # filters and ranking checks (units)
        c = lambda **kw: src.Candidate(**{"source": "x", "id": "i", "kind": "model", "name": "n",  # noqa: E731
                                          "page_url": "u", "thumbnail_url": None, "license": "CC0-1.0",
                                          "format": "glb", **kw})
        kept, dropped = S.filter_candidates([c(id="a"), c(id="b", license="CC-BY-NC-4.0"),
                                             c(id="c", license="CC-BY-4.0"), c(id="d", format="fbx"),
                                             c(id="e", triangles=10 ** 6)], "model")
        kept, dropped = S.filter_candidates([c(id="a"), c(id="b", license="CC-BY-NC-4.0"),
                                             c(id="c", license="CC-BY-4.0"), c(id="d", format="fbx"),
                                             c(id="e", triangles=10 ** 6), c(id="f", triangles=120_000)], "model")
        check([k.id for k in kept] == ["a", "f"] and dropped == {"license": ["x:b"], "license_without_author": ["x:c"],
                                                                  "format": ["x:d"], "too_many_triangles": ["x:e"]},
              "dropped and counted: license, format, and over 10x the triangle budget; a model a few times over is kept")
        check(src.query_terms(["human heart anatomy", "red blood cell"])
              == (["human heart anatomy", "red blood cell"], ["blood", "heart", "cell", "red"]),
              "literal-match sources try the phrases, then each meaningful word alone (not 'human', 'anatomy')")
        try:
            with S.ranking_scope(3):
                S.CandidateRanking.model_validate({"ranking": [{"number": 1, "fits": True, "reason": "r"},
                                                               {"number": 1, "fits": False, "reason": ""},
                                                               {"number": 7, "fits": False, "reason": "r"}]})
            check(False, "a bad ranking is refused with every problem at once")
        except lib.ValidationError as exc:
            m = str(exc)
            check(all(x in m for x in ("ranked more than once", "wasn't shown", "reason is empty",
                                       "aren't ranked: 2, 3")),
                  "a bad ranking is refused with every problem at once")

        code, text, by, res = run()
        h = by["heartPrefab"]
        check(code == 1 and h["status"] == "found" and h["chosen"] == {"rank": 1, "by": "auto"}
              and h["ranked"][0]["name"] == "Human Heart" and h["ranked"][0]["license"] == "CC0-1.0"
              and h["ranked"][0]["author"] == "Jane Doe" and h["ranked"][1]["fits"] is False
              and h["dropped"] == {"text_mismatch": 1} and h["merged_names"] == ["heartModel2"],
              "the heart: collected, the mismatch dropped, ranked by the vision call, rank 1 auto-picked")
        prompt = seen["prompts"][0]
        check(sum(isinstance(p, Path) for p in prompt) == 2 and "four chambers" in prompt[0]
              and all(p.suffix == ".png" and p.is_file() for p in prompt if isinstance(p, Path)),
              "the ranking call sees the description, the must-be-right facts and each thumbnail")
        model = trimesh.load(run_dir / h["files"]["model"], force="scene")
        check(abs(max(model.extents) - 1.0) < 1e-6 and (run_dir / h["files"]["thumbnail"]).is_file()
              and "converted from glTF to GLB" in h["entry"]["provenance"]["modifications"]
              and h["entry"]["provenance"]["license"] == "CC0-1.0" and h["entry"]["triangles"] == 320,
              "the chosen glTF (with its buffers) is converted to a centered unit-cube GLB, provenance kept")
        b, r = by["brickTexture"], by["rocketPrefab"]
        check(b["status"] == "found" and b["entry"]["pixels"] == [2048, 1024] and b["entry"]["mapping"] == "tileable",
              "a texture: its color map, resized to 2048 px, as JPG")
        check(r["status"] == "found" and r["ranked"][0]["source"] == "nasa"
              and r["entry"]["provenance"]["license"] == "public-domain"
              and trimesh.load(run_dir / r["files"]["model"], force="scene").extents.max() > 0.99,
              "a NASA model stored with Git LFS is fetched from the LFS media host")
        u = by["unicornPrefab"]
        check(u["status"] == "no_match" and "no usable candidate" in u["reason"] and "_files" not in u
              and "  NO MATCH unicornPrefab" in text and res["counts"]["found"] == 3,
              "nothing found -> no match, left for the composite builder or a reviewer")

        calls = seen["calls"]
        code, text, by, res = run()
        check(seen["calls"] == calls and "searching 0 of 4" in text and by["heartPrefab"]["status"] == "found",
              "a rerun searches only what isn't decided (resume), with no ranking calls for the rest")
        code, text, by, res = run("--only", "heartPrefab", "--redo", "--no-vision", call_fn=None)
        h = by["heartPrefab"]
        check(h["status"] == "needs_review" and "chosen" not in h and h["ranked"][0]["fits"] is None
              and by["brickTexture"]["status"] == "found",
              "--no-vision ranks by text, picks nothing and leaves it for a reviewer")
        code, text, by, res = run("--choose", "heartPrefab=2", call_fn=None)
        h = by["heartPrefab"]
        check(h["status"] == "found" and h["chosen"] == {"rank": 2, "by": "reviewer"}
              and h["entry"]["provenance"]["record_id"] == h["ranked"][1]["id"] and seen["calls"] == calls,
              "--choose takes a candidate from the saved ranking, with no new search or call")
        code, text, _, _ = run("--choose", "nothing=1", call_fn=None)
        check(code == 2, "--choose for an asset with no saved ranking -> exit 2")
        kept_glb = run_dir / h["files"]["model"]
        before = kept_glb.read_bytes()
        lib.MAX_MODEL_TRIANGLES = 100   # the 320-triangle heart (rank 1) now fails its check
        try:
            code, text, by, res = run("--choose", "heartPrefab=1", call_fn=None)
        finally:
            lib.MAX_MODEL_TRIANGLES = saved[-1]
        h = by["heartPrefab"]
        check(h["status"] == "found" and h["chosen"]["by"] == "reviewer" and "choose_error" in h
              and kept_glb.is_file() and kept_glb.read_bytes() == before
              and not list((run_dir / "found").glob(".*.new")),
              "a failed --choose keeps the earlier pick and its files (built in a staging folder)")
        import pathlib
        real_rename = pathlib.Path.rename

        def locked(self, target):
            if pathlib.Path(target).name.endswith(".old"):
                raise PermissionError("model.glb is open in another program")
            return real_rename(self, target)
        pathlib.Path.rename = locked
        try:
            code, text, by, res = run("--choose", "heartPrefab=1", call_fn=None)
        finally:
            pathlib.Path.rename = real_rename
        h = by["heartPrefab"]
        check("PermissionError" in h["choose_error"] and kept_glb.read_bytes() == before
              and all((run_dir / f).is_file() for f in h["files"].values())
              and not list((run_dir / "found").glob(".*")),
              "an earlier pick that can't be moved aside (a file open) stays whole; nothing half-swapped")
        code, text, by, res = run("--choose", "heartPrefab=2", call_fn=None)
        check(by["heartPrefab"]["status"] == "found" and "choose_error" not in by["heartPrefab"],
              "a later successful --choose drops the earlier choose_error")
        later = lib.LocalFolderStore(tmp / "libLater", create=True)
        code, text, by, res = run("--save", "--library", str(tmp / "libLater"), call_fn=None)
        later = lib.LocalFolderStore(tmp / "libLater")
        check("searching 0 of" in text and by["heartPrefab"]["saved"]["added"]
              and by["brickTexture"]["saved"]["added"] and len(later.entries()) == 3
              and {"heartPrefab", "heartModel2"} <= set(later.get(by["heartPrefab"]["saved"]["asset_id"]).aliases),
              "--save on a later run stores what earlier runs found (from found/), with no new search")
        rev = lib.LocalFolderStore(tmp / "libLater").identity()["revision"]
        code, text, by, res = run("--save", "--library", str(tmp / "libLater"), call_fn=None)
        check(lib.LocalFolderStore(tmp / "libLater").identity()["revision"] == rev and "  found" not in text,
              "saving again writes nothing for what was already saved")

        saved_budget = lib.MAX_MODEL_TRIANGLES
        lib.MAX_MODEL_TRIANGLES = 100
        try:
            code, text, by, res = run("--only", "heartPrefab", "--redo")
        finally:
            lib.MAX_MODEL_TRIANGLES = saved_budget
        h = by["heartPrefab"]
        check(h["status"] == "needs_simplifying" and h["pending"] == {"rank": 1} and h["fetch_errors"][0]["over_budget"]
              and "simplifying (#113)" in h["reason"] and "SIMPLIFY" in text,
              "the best match over the triangle budget is kept for the simplify step, not failed")

        acg = src.SOURCES["ambientcg"].search(["red brick"], "texture")
        check([c.id for c in acg] == ["Bricks090"] and acg[0].format == "jpg" and acg[0].mapping == "tileable"
              and "Created using Bricks 090 from ambientCG.com" in acg[0].notes
              and src.SOURCES["ambientcg"].search(["red brick"], "model") == [],
              "ambientCG: materials with a color map, the 2K JPG zip, its credit line; textures only")
        code, text, by, res = run("--only", "brickTexture", "--redo", "--sources", "ambientcg")
        b = by["brickTexture"]
        check(b["status"] == "found" and b["ranked"][0]["source"] == "ambientcg" and b["entry"]["pixels"] == [96, 48]
              and b["entry"]["mapping"] == "tileable" and not (run_dir / "evil_Color.txt").exists()
              and "/acg/huge" not in " ".join(srv.hits) and "/acg/get?file=x2" in srv.hits,
              "ambientCG: only the color map is read out of the 2K zip (nothing extracted, no 8K download)")

        code, text, _, _ = run("--only", "heartPrefab", "--redo", "--sources", "polypizza")
        check(code == 2 and "POLY_PIZZA_API_KEY" in text, "Poly Pizza named without a key -> exit 2")
        code, text, _, _ = run("--only", "heartPrefab", "--redo", "--no-vision", call_fn=None)
        check("note: skipping polypizza: no API key" in text, "by default, Poly Pizza without a key is skipped with a note")
        os.environ[src.POLYPIZZA_KEY_ENV] = "test-key"
        check(src.SOURCES["polypizza"].headers_for("https://evil.example/x.glb") == {}
              and src.SOURCES["polypizza"].headers_for(srv.base + "/pp/x") == {"x-auth-token": "test-key"},
              "the Poly Pizza key goes only to Poly Pizza's own hosts")
        srv.headers.clear()
        code, text, by, res = run("--only", "heartPrefab", "--redo", "--sources", "polypizza")
        h = by["heartPrefab"]
        sent = {path.split("?")[0]: hd.get("x-auth-token") for path, hd in srv.headers}
        prov = h["entry"]["provenance"] if "entry" in h else {}
        check(h["status"] == "found" and h["dropped"] == {"license": 1, "too_many_triangles": 1}
              and prov.get("license") == "CC-BY-3.0" and prov.get("author") == "drx"
              and prov.get("title") == "Anatomical Heart" and "Licence at" in (prov.get("notes") or "")
              and sent.get("/pp/search/heart") == "test-key" and sent.get("/static/heart.glb") == "test-key"
              and sent.get("/static/heart.webp") == "test-key",
              "Poly Pizza: found by the word 'heart' after the phrase found nothing; CC-BY credited, NC and "
              "far-over-budget hits dropped, the key sent")
        os.environ.pop(src.POLYPIZZA_KEY_ENV, None)

        os.environ[src.SMITHSONIAN_KEY_ENV] = "si-secret-key"
        si = {c.name: c for c in src.SOURCES["smithsonian"].search(["human heart anatomy"], "model")}
        check(si["Heart Specimen"].triangles == 150000 and si["Heart Specimen"].download_url.endswith("_std.glb")
              and si["Human Heart Model"].download_url.endswith("model-40k-2048_std.glb")
              and si["Heart Drawing"].license == "Usage conditions apply"
              and "Ritual ewer (gong) & heart" in si
              and si["Human Heart Model"].page_url.endswith("Human Heart Model") and "3d_package:small" in si["Human Heart Model"].id,
              "Smithsonian: 3d_voyager media, the smallest uncompressed GLB, its triangle count from the file name")
        srv.headers.clear()
        code, text, by, res = run("--only", "heartPrefab", "--redo", "--sources", "smithsonian")
        h = by["heartPrefab"]
        paths = [p for p, _ in srv.headers]
        q_sent = [p for p in paths if p.startswith("/si/search")]
        from urllib.parse import parse_qs, urlsplit
        first_q = parse_qs(urlsplit(q_sent[0]).query)["q"][0] if q_sent else ""
        specimen = next((r for r in h["ranked"] if r["name"] == "Heart Specimen"), {})
        check(h["status"] == "found" and h["ranked"][0]["name"] == "Human Heart Model"
              and h["dropped"] == {"license": 1} and specimen.get("needs_simplifying") is True
              and first_q == '(human heart anatomy) AND online_media_type:"3D Models"'
              and not any("heart-150k" in p for p in paths)
              and "query form: fielded" in h["sources"]["smithsonian"]["note"]
              and h["entry"]["provenance"]["license"] == "CC0-1.0"
              and all("api_key=si-secret-key" in p for p in paths if p.startswith("/si/"))
              and not any("si-secret-key" in p for p in paths if not p.startswith("/si/")),
              "Smithsonian: the 3D filter inside the query, CC0 only, a 150k scan that fits kept and marked "
              "for simplifying but never downloaded, the in-budget fit found, the key only on the API")
        code, text, by, res = run("--only", "heartPrefab", "--redo", "--sources", "smithsonian",
                                  "--max-triangles", "200000")
        h = by["heartPrefab"]
        check(h["dropped"] == {"license": 1} and res["max_triangles"] == 200000
              and any(r["name"] == "Heart Specimen" and not r["needs_simplifying"] for r in h["ranked"]),
              "--max-triangles raises the limit for a run (recorded), keeping big scans")
        lib.MAX_MODEL_TRIANGLES = saved[-1]
        src.SMITHSONIAN_API = srv.base + "/si-broken"
        try:
            src.SOURCES["smithsonian"].search(["heart"], "model")
            check(False, "a failed Smithsonian request never shows the key")
        except src.SourceError as exc:
            check("si-secret-key" not in str(exc) and "***" in str(exc) and not isinstance(exc, src.QueryRefused),
                  "a failed Smithsonian request never shows the key; a 500 is not a refused query")
        srv.routes["/si-400/search"] = (400, b"{}")
        src.SMITHSONIAN_API = srv.base + "/si-400"
        try:
            src.SOURCES["smithsonian"]._search("heart", 1, "fielded")
            check(False, "a 400 from the Smithsonian is a refused query")
        except src.QueryRefused:
            check(True, "a 400 from the Smithsonian is a refused query")
        src.SMITHSONIAN_API = srv.base + "/si"
        os.environ.pop(src.SMITHSONIAN_KEY_ENV, None)

        real_prepare = S.prepare

        def boom(*a, **k):
            raise RuntimeError("an unexpected bug")
        S.prepare = boom
        try:
            code, text, by, res = run("--only", "brickTexture", "--redo")
        finally:
            S.prepare = real_prepare
        b = by["brickTexture"]
        check(code == 1 and b["status"] == "failed" and "RuntimeError: an unexpected bug" in b["fetch_errors"][0]["error"],
              "an unexpected error in one candidate fails that candidate, not the run")
        si_src = src.SOURCES["smithsonian"]
        real_search, real_form = si_src._search, si_src._form
        os.environ[src.SMITHSONIAN_KEY_ENV] = "si-secret-key"
        row = si_row("Human Heart Model", si_media("small", [("model-40k-2048_std.glb", "GLB", False, 3)]))

        calls = []

        def fake(behaviour):
            def _search(q, rows, form):
                calls.append((form, q))
                return behaviour(q, form)
            return _search

        def raise_(exc):
            raise exc

        def run_si(behaviour, form=None):
            calls.clear()
            si_src._search, si_src._form = fake(behaviour), form
            try:
                return si_src.search(["human heart anatomy"], "model"), None
            except src.SourceError as exc:
                return None, exc
        try:
            got, _ = run_si(lambda q, form: raise_(src.QueryRefused("HTTP 400")) if form == "fielded" else [row])
            check(got and si_src._form == "fqs" and "refused: fielded" in si_src.note,
                  "a query form the Smithsonian refuses (4xx) falls through to the next")
            got, err = run_si(lambda q, form: raise_(src.QueryRefused("HTTP 400")))
            check(got is None and "every query form failed" in str(err), "every form refused -> an error")
            got, err = run_si(lambda q, form: raise_(src.SourceError("HTTP 503 after retries")))
            check(got is None and len(calls) == 1 and "503" in str(err),
                  "a network failure or 5xx is raised at once, not tried on every form")
            got, _ = run_si(lambda q, form: [] if form == "fielded" else ([row] if form == "plain" else []),
                            form="fielded")
            check(got and si_src._form == "plain" and [f for f, _ in calls][0] == "fielded",
                  "the remembered form goes first, and the others are tried when it finds nothing")
            got, _ = run_si(lambda q, form: [row] if q == "human heart anatomy" else raise_(src.QueryRefused("400")))
            check(got and si_src._form == "fielded" and {f for f, _ in calls} == {"fielded"},
                  "hits a form found before a refusal are kept, and that form is remembered")
        finally:
            si_src._search, si_src._form = real_search, real_form
            os.environ.pop(src.SMITHSONIAN_KEY_ENV, None)
        pick = src.Smithsonian._glb({"resources": [
            {"url": "u/full_std.glb", "filename": "full_std.glb", "attributes": [{"MODEL_FILE_TYPE": "GLB"}]},
            {"url": "u/low-40k-2048_std.glb", "filename": "low-40k-2048_std.glb",
             "attributes": [{"MODEL_FILE_TYPE": "GLB"}, {"FILE_SIZE": 900}]}]})
        check(pick["url"] == "u/low-40k-2048_std.glb", "a GLB with a known size beats one with none")

        class Noted(src.Source):
            name, kinds, note, truncated = "noted", ("model",), "form x", True

            def search(self, queries, kind, limit=src.PER_SOURCE):
                return []
        _, per = S.collect({"name": "a", "kind": "model", "plan": {}}, [Noted()])
        check(per["noted"]["note"] == "form x; the source's listing was cut short",
              "a source's note and the cut-short note are both kept")

        store = lib.LocalFolderStore(tmp / "libSearch", create=True)
        code, text, by, res = run("--only", "heartPrefab", "--redo", "--save", "--library", str(tmp / "libSearch"))
        h = by["heartPrefab"]
        entry = lib.LocalFolderStore(tmp / "libSearch").get("polyhaven_human_heart")
        check(h["saved"]["asset_id"] == "polyhaven_human_heart" and entry is not None
              and {"heartPrefab", "heartModel2"} <= set(entry.aliases)
              and entry.provenance.method == "search" and entry.provenance.page_url.endswith("/a/human_heart")
              and res["library_after"]["entries"] == 1 and not [p for p in lib.check_index(store.index, store.root)
                                                                if p.level == "error"],
              "--save stores the found model with search provenance and the lab's names as aliases")
    finally:
        srv.close()
        (src.POLYHAVEN_API, src.GITHUB_API, src.GITHUB_RAW, src.GITHUB_MEDIA, src.AMBIENTCG_API,
         src.POLYPIZZA_API, src.POLYPIZZA_HOSTS, src.SMITHSONIAN_API, lib.MAX_MODEL_TRIANGLES) = saved
        for env, value in ((src.POLYPIZZA_KEY_ENV, saved_key), (src.SMITHSONIAN_KEY_ENV, saved_si_key)):
            os.environ.pop(env, None)
            if value is not None:
                os.environ[env] = value
        src.SOURCES["polyhaven"].__init__()
        src.SOURCES["nasa"].__init__()


def main() -> int:
    saved_env = os.environ.pop(lib.LIBRARY_ENV, None)
    tmp = Path(tempfile.mkdtemp(prefix="provisioning_smoke_"))
    try:
        part1()
        part2(tmp)
        part3(tmp)
        part4(tmp)
        part5(tmp)
        part6(tmp)
        part7(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if saved_env is not None:
            os.environ[lib.LIBRARY_ENV] = saved_env
    print(f"\n{'FAILED: ' + str(len(FAILURES)) if FAILURES else 'all checks passed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
