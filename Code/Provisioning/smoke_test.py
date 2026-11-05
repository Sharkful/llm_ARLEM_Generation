"""
Offline smoke test for provisioning: no API keys, no network, no paid calls.

    python "Code/Provisioning/smoke_test.py"

Part 1 (the committed seed list, #107): it parses, every download is https and
pinned, ids and names are unique, built-ins match the moon-lab sets in
``lab_metrics`` and ``Code/Hierarchical/registry.py``, and its licenses are allowed.

Part 2 (the asset library, #107), against a local mock download server and a
throwaway library (never the real one):
- seeding: built-ins, pinning, a transient 503 retried, a 404 not retried, a
  non-JPEG rejected and never pinned, disallowed and uncredited licenses refused
- reruns download nothing; a damaged local file is re-downloaded; a changed
  source file is refused; ``--pin`` never overwrites an existing pin
- two libraries seeded from one list share a content hash, not a library id
- lookups by name and alias ignore case; alias clashes are refused
- a write is refused when another process wrote in between, and nothing is lost
- ``put`` refuses an entry without a license or size and leaves the library as it was
- the validator catches missing, altered and wrong-format files, duplicate names,
  and unlisted files
- credit lines, the GLB check, library path resolution and the CLI exit codes

Requires the project venv (pydantic, Pillow). Exit code 1 on any failure.
"""

from __future__ import annotations

import contextlib
import copy
import http.server
import io
import json
import os
import shutil
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
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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
             "download": {"url": srv.url(path), "filename": "texture.jpg"},
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
        tex("map_a", "/a.jpg"),
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
    lib.RETRY_WAITS = (0.0, 0.0)
    srv = Server()
    try:
        srv.files = {"/a.jpg": jpeg_bytes(), "/b.jpg": jpeg_bytes(color=(5, 6, 7)), "/flaky.jpg": jpeg_bytes(color=(1, 2, 3)),
                     "/png.jpg": png_bytes()}
        seed_path = tmp / "seed.json"
        lib.write_json_atomic(seed_path, seed_doc(srv))

        # Seeding
        store = lib.LocalFolderStore(tmp / "libA", create=True)
        check(store.identity()["revision"] == 0 and store.entries() == [], "a new library is empty")
        res, rep = run_seed(store, seed_path)
        check(res["map_a"].status == "failed" and "not pinned" in res["map_a"].detail,
              "an unpinned download is refused without --pin")
        check(store.get("map_a") is None, "...and nothing is added")
        check(res["box_builtin"].status == "added", "a built-in without a file is added")
        srv.fail_once = {"/flaky.jpg": 503}
        res, rep = run_seed(store, seed_path, pin=True)
        check(res["box_builtin"].status == "unchanged", "...and is unchanged on the next run")
        check(res["map_a"].status == "added", "a pinned download is added")
        check(res["flaky"].status == "added" and res["flaky"].retries == 1,
              "a 503 is retried once and then succeeds")
        check(res["missing"].status == "failed" and "404" in res["missing"].detail
              and srv.hits.get("/missing.jpg") == 2, "a 404 fails without retrying")
        check(res["not_jpeg"].status == "failed" and "not JPEG" in res["not_jpeg"].detail,
              "a PNG served as .jpg is refused")
        check(res["nc_map"].status == "failed" and "not allowed" in res["nc_map"].detail,
              "a CC-BY-NC license is refused")
        check(res["no_author"].status == "failed" and "needs credit" in res["no_author"].detail,
              "a CC-BY asset without an author is refused")
        pinned = {e["asset_id"]: e["download"].get("sha256")
                  for e in json.loads(seed_path.read_text())["entries"] if "download" in e}
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
        check(json.loads(seed_path.read_text())["entries"][1]["download"]["sha256"] == pinned["map_a"],
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
        check(line is not None and "A. Author" in line and "CC-BY-4.0" in line and "Changes: none" in line,
              "a CC-BY asset gets a credit line")
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
        check(cli("seed", "--library", str(tmp / "libD"), "--seed-file", str(seed_path),
                  "--only", "unknown_id") == 2, "CLI: seed --only with an unknown id -> exit 2")
        check(cli("seed", "--library", str(tmp / "libD"), "--seed-file", str(tmp / "nope.json")) == 2,
              "CLI: an unreadable seed list -> exit 2")
    finally:
        srv.close()


def main() -> int:
    saved_env = os.environ.pop(lib.LIBRARY_ENV, None)
    tmp = Path(tempfile.mkdtemp(prefix="provisioning_smoke_"))
    try:
        part1()
        part2(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if saved_env is not None:
            os.environ[lib.LIBRARY_ENV] = saved_env
    print(f"\n{'FAILED: ' + str(len(FAILURES)) if FAILURES else 'all checks passed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
