"""
Asset library for model and texture provisioning (#107; spec: PROVISIONING.md).

One shared library of models and textures that grows with every accepted asset and
is what provisioning reuses. For now it is a **local folder**; a server can take its
place later behind the same ``AssetStore`` interface. Because a local library grows
differently on every machine, results depend on which library a run used: every
store reports an ``identity()`` (library id, revision, content hash) that runs
record, and tests always use a throwaway library built from the seed list.

    <library>/
      library.json              the index: one entry per asset
      <asset_id>/texture.jpg    or model.glb
      <asset_id>/thumbnail.png

The library folder is chosen by ``--library``, else ``$ARLEM_ASSET_LIBRARY``, else
``Artifacts/Assets/Library/`` (gitignored). Asset files never go into git; what git
tracks is ``seed_assets.json``: metadata, download URLs and checksums for the assets
we explicitly keep, from which ``seed`` rebuilds a library on any machine.

    python "Code/Provisioning/library.py" seed     [--library PATH] [--only ID ...] [--pin]
    python "Code/Provisioning/library.py" validate [--library PATH] [--json]
    python "Code/Provisioning/library.py" info     [--library PATH]
    python "Code/Provisioning/library.py" list     [--library PATH] [--kind model|texture] [--json]
    python "Code/Provisioning/library.py" find     NAME [--library PATH]

Exit codes: 0 ok; 1 validation errors or failed seed entries; 2 bad input (no
library, unreadable seed list).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LIBRARY = PROJECT_ROOT / "Artifacts" / "Assets" / "Library"
LIBRARY_ENV = "ARLEM_ASSET_LIBRARY"
SEED_FILE = Path(__file__).resolve().parent / "seed_assets.json"
INDEX_NAME = "library.json"
SCHEMA_VERSION = 1

# License policy (owner decision, 2026-11-03): CC0, public domain and CC-BY. Never
# NC, ND, share-alike, "all rights reserved" or unknown. Value: credit required.
ALLOWED_LICENSES = {
    "CC0-1.0": False,
    "public-domain": False,
    "CC-BY-3.0": True,
    "CC-BY-4.0": True,
}
LICENSE_URLS = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
}

# Headset limits (proposals; set in #105). Textures load as JPG today.
MAX_TEXTURE_SIDE = 2048
THUMBNAIL_SIDE = 256
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
DOWNLOAD_TIMEOUT = 60
# Transient download errors (429 / 5xx / connection) are retried after these waits.
RETRY_WAITS: tuple[float, ...] = (2.0, 4.0)
USER_AGENT = "llm_ARLEM_Generation-asset-provisioning/1"

ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FileRole = Literal["model", "texture", "thumbnail"]
# The file each kind must carry, unless it is a built-in.
PRIMARY_ROLE = {"model": "model", "texture": "texture"}


class LibraryError(Exception):
    """A library operation that can't proceed (bad input, missing library)."""


class ConcurrentWriteError(LibraryError):
    """library.json changed on disk since this store read it."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def name_key(name: str) -> str:
    """How lab names are compared: ignoring case and surrounding spaces."""
    return name.strip().lower()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ── Data model ────────────────────────────────────────────────────────


class Provenance(BaseModel):
    """Where an asset came from and under what terms."""
    model_config = ConfigDict(extra="forbid")

    method: Literal["builtin", "seed", "search", "builder", "kit", "composite", "manual"] = "seed"
    source_site: Optional[str] = None
    source_url: Optional[str] = Field(None, description="The file as downloaded, if any.")
    page_url: Optional[str] = Field(None, description="The human-readable page for the asset.")
    record_id: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    license: str
    license_url: Optional[str] = None
    modifications: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    obtained: Optional[str] = None


class AssetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: FileRole
    path: str = Field(description="File name inside the entry's folder.")
    sha256: str
    bytes: int


class LibraryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    kind: Literal["model", "texture"]
    name: str = Field(description="The name labs use and the file stem it ships under.")
    builtin: bool = Field(False, description="Already on the headset: matched, never shipped.")
    reusable: bool = Field(True, description="False for system objects the matcher must skip.")
    form: Optional[str] = None
    mapping: Optional[Literal["equirectangular", "tileable", "uv"]] = None
    description: str
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list, description="Other lab names matched to it.")
    files: list[AssetFile] = Field(default_factory=list)
    bounds: Optional[list[float]] = Field(None, description="Model size in meters [x, y, z].")
    triangles: Optional[int] = None
    pixels: Optional[list[int]] = Field(None, description="Texture size [width, height].")
    review: Literal["builtin", "auto", "reviewed"] = "auto"
    provenance: Optional[Provenance] = None
    notes: Optional[str] = None
    added: Optional[str] = None

    def file(self, role: str) -> Optional[AssetFile]:
        return next((f for f in self.files if f.role == role), None)

    def names(self) -> list[str]:
        return [self.name, *self.aliases]


class LibraryIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    library_id: str
    revision: int = 0
    created: str
    updated: str
    entries: list[LibraryEntry] = Field(default_factory=list)


@dataclass
class Problem:
    level: Literal["error", "warning"]
    asset_id: Optional[str]
    message: str

    def __str__(self) -> str:
        where = f"{self.asset_id}: " if self.asset_id else ""
        return f"{self.level}: {where}{self.message}"


def credit_line(entry: LibraryEntry) -> Optional[str]:
    """The credits-file line for an asset whose license requires credit, else None."""
    p = entry.provenance
    if p is None or not ALLOWED_LICENSES.get(p.license, False):
        return None
    parts = [f'"{p.title or entry.name}" by {p.author or "unknown author"}']
    if p.page_url or p.source_url:
        parts.append(p.page_url or p.source_url)
    parts.append(f"{p.license} ({p.license_url or LICENSE_URLS.get(p.license, '')})".rstrip(" ()"))
    parts.append("Changes: " + ("; ".join(p.modifications) if p.modifications else "none"))
    return ", ".join(parts) + "."


# ── Checks ────────────────────────────────────────────────────────────


def _image_facts(path: Path) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    """(format, (width, height)) of an image, or (None, None) if it isn't one."""
    from PIL import Image
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.format, img.size
    except Exception:
        return None, None


def _is_glb(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(12)
    return len(head) == 12 and head[:4] == b"glTF" and int.from_bytes(head[4:8], "little") == 2


def check_entry(entry: LibraryEntry, folder: Optional[Path] = None) -> list[Problem]:
    """Problems with one entry. With ``folder``, its files are checked too."""
    out: list[Problem] = []

    def err(msg):
        out.append(Problem("error", entry.asset_id, msg))

    def warn(msg):
        out.append(Problem("warning", entry.asset_id, msg))

    if not ASSET_ID.match(entry.asset_id):
        err("asset_id must be lowercase letters, digits and underscores")
    if not entry.name.strip():
        err("name is empty")
    if not entry.description.strip():
        err("description is empty")

    p = entry.provenance
    if p is None:
        if not entry.builtin:
            err("no provenance: every asset needs a source and a license")
    else:
        if p.license not in ALLOWED_LICENSES:
            err(f"license {p.license!r} is not allowed (allowed: {', '.join(ALLOWED_LICENSES)})")
        elif ALLOWED_LICENSES[p.license]:
            missing = [k for k in ("author", "title") if not getattr(p, k)]
            if not (p.page_url or p.source_url):
                missing.append("page_url or source_url")
            if missing:
                err(f"{p.license} needs credit, but {', '.join(missing)} is missing")

    primary = PRIMARY_ROLE[entry.kind]
    if not entry.builtin and entry.file(primary) is None:
        err(f"no {primary} file")
    if entry.file(primary) is not None or not entry.builtin:
        if entry.kind == "model" and (not entry.bounds or entry.triangles is None):
            err("model has no known size (bounds and triangles are required)")
        if entry.kind == "texture" and not entry.pixels:
            err("texture has no known size (pixels is required)")
    if entry.kind == "texture" and entry.mapping is None:
        warn("texture has no mapping (equirectangular, tileable or uv)")
    if entry.pixels and max(entry.pixels) > MAX_TEXTURE_SIDE:
        warn(f"texture is {entry.pixels[0]}x{entry.pixels[1]}, larger than {MAX_TEXTURE_SIDE} px")
    if not entry.builtin and entry.file(primary) is not None and entry.file("thumbnail") is None:
        warn("no thumbnail")

    roles = [f.role for f in entry.files]
    for role in set(roles):
        if roles.count(role) > 1:
            err(f"more than one {role} file")
    for f in entry.files:
        if not FILE_NAME.match(f.path) or f.path in (".", ".."):
            err(f"file name {f.path!r} is not a plain file name")
            continue
        if folder is None:
            continue
        path = folder / f.path
        if not path.is_file():
            err(f"{f.role} file {f.path} is missing")
            continue
        if path.stat().st_size != f.bytes or sha256_of(path) != f.sha256:
            err(f"{f.role} file {f.path} does not match its recorded checksum")
            continue
        if f.role == "texture":
            fmt, size = _image_facts(path)
            if fmt != "JPEG":
                err(f"texture file {f.path} is {fmt or 'not an image'}, not JPEG")
            elif entry.pixels and list(size) != list(entry.pixels):
                err(f"texture is {size[0]}x{size[1]} but the entry says {entry.pixels}")
        elif f.role == "model" and not _is_glb(path):
            err(f"model file {f.path} is not a glTF 2.0 binary (GLB)")
        elif f.role == "thumbnail" and _image_facts(path)[0] is None:
            err(f"thumbnail {f.path} is not an image")
    return out


def check_index(index: LibraryIndex, root: Optional[Path] = None) -> list[Problem]:
    """Problems across the whole library, including each entry's files when ``root``."""
    out: list[Problem] = []
    seen_ids: set[str] = set()
    owner: dict[str, str] = {}
    for e in index.entries:
        if e.asset_id in seen_ids:
            out.append(Problem("error", e.asset_id, "duplicate asset_id"))
        seen_ids.add(e.asset_id)
        for n in e.names():
            k = name_key(n)
            if k in owner and owner[k] != e.asset_id:
                out.append(Problem("error", e.asset_id,
                                   f"name {n!r} is already used by {owner[k]}"))
            owner.setdefault(k, e.asset_id)
        out += check_entry(e, root / e.asset_id if root else None)
    if root is not None and root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name not in seen_ids:
                out.append(Problem("warning", None, f"folder {child.name}/ has no library entry"))
        for e in index.entries:
            folder = root / e.asset_id
            if folder.is_dir():
                known = {f.path for f in e.files}
                for f in sorted(folder.iterdir()):
                    if f.name not in known:
                        out.append(Problem("warning", e.asset_id, f"unlisted file {f.name}"))
    return out


# ── Stores ────────────────────────────────────────────────────────────


class AssetStore(Protocol):
    """What provisioning needs from a library. ``LocalFolderStore`` is the only
    implementation today; a server-backed store can implement the same methods."""

    def identity(self) -> dict: ...
    def entries(self) -> list[LibraryEntry]: ...
    def get(self, asset_id: str) -> Optional[LibraryEntry]: ...
    def find(self, name: str) -> Optional[LibraryEntry]: ...
    def put(self, entry: LibraryEntry, files: dict[str, Path] | None = None) -> LibraryEntry: ...
    def add_alias(self, asset_id: str, name: str) -> LibraryEntry: ...
    def path_of(self, asset_id: str, role: str) -> Optional[Path]: ...
    def validate(self) -> list[Problem]: ...


def resolve_library_path(cli_value: str | os.PathLike | None = None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get(LIBRARY_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_LIBRARY


class LocalFolderStore:
    """The library as a folder on this machine.

    Writes are atomic, and refused if ``library.json`` changed since this store read
    it (another process wrote in between), so two runs can't silently overwrite each
    other's additions. One writer at a time is the expected use.
    """

    def __init__(self, root: Path, *, create: bool = False):
        self.root = Path(root)
        self.index_path = self.root / INDEX_NAME
        if not self.index_path.is_file():
            if not create:
                raise LibraryError(
                    f"no asset library at {self.root} (create one with "
                    f"`library.py seed --library {self.root}`)")
            self.root.mkdir(parents=True, exist_ok=True)
            stamp = now_iso()
            self.index = LibraryIndex(library_id=uuid.uuid4().hex, created=stamp, updated=stamp)
            write_json_atomic(self.index_path, self.index.model_dump(exclude_none=True))
        else:
            self.index = self._read()

    def _read(self) -> LibraryIndex:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            index = LibraryIndex.model_validate(data)
        except (OSError, ValueError, ValidationError) as exc:
            raise LibraryError(f"{self.index_path} is not a valid library index: {exc}") from exc
        if index.schema_version != SCHEMA_VERSION:
            raise LibraryError(f"{self.index_path} has schema version {index.schema_version}; "
                               f"this tool reads version {SCHEMA_VERSION}")
        return index

    def _write(self) -> None:
        on_disk = self._read()
        if on_disk.revision != self.index.revision or on_disk.library_id != self.index.library_id:
            raise ConcurrentWriteError(
                f"{self.index_path} changed since it was read (revision {on_disk.revision}, "
                f"expected {self.index.revision}); rerun to pick up the other writer's changes")
        self.index.revision += 1
        self.index.updated = now_iso()
        write_json_atomic(self.index_path, self.index.model_dump(exclude_none=True))

    # Reading

    def content_hash(self) -> str:
        """Hash of every entry (files by checksum), independent of when it was written."""
        # Download dates differ between machines, so they are left out.
        skip = {"added": True, "provenance": {"obtained"}}
        rows = sorted((e.model_dump(exclude_none=True, exclude=skip) for e in self.index.entries),
                      key=lambda r: r["asset_id"])
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]

    def identity(self) -> dict:
        """What a run records about the library it used. Two runs used the same
        library contents exactly when ``library_id`` and ``content_hash`` match."""
        return {
            "store": "local-folder",
            "location": str(self.root),
            "library_id": self.index.library_id,
            "revision": self.index.revision,
            "content_hash": self.content_hash(),
            "entries": len(self.index.entries),
        }

    def entries(self) -> list[LibraryEntry]:
        return list(self.index.entries)

    def get(self, asset_id: str) -> Optional[LibraryEntry]:
        return next((e for e in self.index.entries if e.asset_id == asset_id), None)

    def find(self, name: str) -> Optional[LibraryEntry]:
        """The entry whose name or an alias equals ``name``, ignoring case."""
        k = name_key(name)
        return next((e for e in self.index.entries if k in {name_key(n) for n in e.names()}), None)

    def path_of(self, asset_id: str, role: str) -> Optional[Path]:
        e = self.get(asset_id)
        f = e.file(role) if e else None
        return self.root / asset_id / f.path if f else None

    def validate(self) -> list[Problem]:
        return check_index(self.index, self.root)

    # Writing

    def put(self, entry: LibraryEntry, files: dict[str, Path] | None = None) -> LibraryEntry:
        """Add or replace an entry. ``files`` maps a role to a source file, copied into
        the entry's folder under its own name. Roles not given keep their current file
        (when replacing). Refused if the entry, with its files, has any error."""
        if not ASSET_ID.match(entry.asset_id):
            raise LibraryError(f"bad asset_id {entry.asset_id!r}")
        entry = entry.model_copy(deep=True)
        old = self.get(entry.asset_id)
        folder = self.root / entry.asset_id
        staging = Path(tempfile.mkdtemp(prefix=f".{entry.asset_id}.", dir=self.root))
        os.chmod(staging, 0o755)  # mkdtemp makes it owner-only
        try:
            kept = {f.role: f for f in (old.files if old else [])}
            for f in kept.values():
                src = folder / f.path
                if src.is_file():
                    shutil.copy2(src, staging / f.path)
            for role, src in (files or {}).items():
                src = Path(src)
                if role in kept:
                    (staging / kept[role].path).unlink(missing_ok=True)
                name = src.name
                shutil.copyfile(src, staging / name)
                kept[role] = AssetFile(role=role, path=name, sha256=sha256_of(staging / name),
                                       bytes=(staging / name).stat().st_size)
            entry.files = sorted(kept.values(), key=lambda f: f.role)
            entry.added = (old.added if old and old.added else None) or now_iso()
            errors = [p for p in check_entry(entry, staging) if p.level == "error"]
            if errors:
                raise LibraryError("; ".join(str(p) for p in errors))
            others = [e for e in self.index.entries if e.asset_id != entry.asset_id]
            taken = {name_key(n): e.asset_id for e in others for n in e.names()}
            clash = [n for n in entry.names() if name_key(n) in taken]
            if clash:
                raise LibraryError(f"{entry.asset_id}: name {clash[0]!r} is already used by "
                                   f"{taken[name_key(clash[0])]}")
            # Swap the folder in, then record the entry.
            backup = None
            if folder.exists():
                backup = self.root / f".{entry.asset_id}.old"
                if backup.exists():
                    shutil.rmtree(backup)
                folder.rename(backup)
            staging.rename(folder)
            staging = None
            self.index.entries = sorted([*others, entry], key=lambda e: e.asset_id)
            try:
                self._write()
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                if backup is not None:
                    backup.rename(folder)
                self.index = self._read()
                raise
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
            return entry
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def add_alias(self, asset_id: str, name: str) -> LibraryEntry:
        e = self.get(asset_id)
        if e is None:
            raise LibraryError(f"no entry {asset_id!r}")
        if name_key(name) in {name_key(n) for n in e.names()}:
            return e
        other = self.find(name)
        if other is not None:
            raise LibraryError(f"name {name!r} is already used by {other.asset_id}")
        e.aliases.append(name.strip())
        try:
            self._write()
        except BaseException:
            self.index = self._read()
            raise
        return e


def open_store(path: str | os.PathLike | None = None, *, create: bool = False) -> LocalFolderStore:
    return LocalFolderStore(resolve_library_path(path), create=create)


# ── Downloads ─────────────────────────────────────────────────────────


class DownloadError(Exception):
    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def _fetch_once(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, open(dest, "wb") as out:
            total = 0
            while chunk := resp.read(1 << 16):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise DownloadError(f"{url} is larger than {MAX_DOWNLOAD_BYTES} bytes")
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        transient = exc.code == 429 or exc.code >= 500
        raise DownloadError(f"{url}: HTTP {exc.code}", transient=transient) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
        reason = getattr(exc, "reason", exc)
        raise DownloadError(f"{url}: {reason}", transient=True) from exc


def download(url: str, dest: Path) -> int:
    """Download ``url`` to ``dest``, retrying transient errors. Returns retries used."""
    if not url.startswith(("https://", "http://")):
        raise DownloadError(f"{url}: only http(s) downloads are allowed")
    retries = 0
    for wait in (*RETRY_WAITS, None):
        try:
            _fetch_once(url, dest)
            return retries
        except DownloadError as exc:
            dest.unlink(missing_ok=True)
            if not exc.transient or wait is None:
                raise
            retries += 1
            time.sleep(wait)
    return retries


def make_thumbnail(src: Path, dest: Path) -> None:
    from PIL import Image
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE))
        img.save(dest, format="PNG")


# ── Seeding ───────────────────────────────────────────────────────────


class SeedDownload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    filename: str
    sha256: Optional[str] = None
    bytes: Optional[int] = None


class SeedEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    kind: Literal["model", "texture"]
    name: str
    builtin: bool = False
    reusable: bool = True
    form: Optional[str] = None
    mapping: Optional[Literal["equirectangular", "tileable", "uv"]] = None
    description: str
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    download: Optional[SeedDownload] = None
    provenance: Optional[Provenance] = None


class SeedList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    about: Optional[str] = None
    entries: list[SeedEntry]


def load_seed(path: Path = SEED_FILE) -> SeedList:
    try:
        seed = SeedList.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        raise LibraryError(f"{path} is not a valid seed list: {exc}") from exc
    ids = [e.asset_id for e in seed.entries]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise LibraryError(f"{path}: duplicate asset_id {', '.join(dupes)}")
    return seed


@dataclass
class SeedResult:
    asset_id: str
    status: Literal["added", "updated", "unchanged", "failed"]
    detail: str = ""
    retries: int = 0


@dataclass
class SeedReport:
    results: list[SeedResult] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[SeedResult]:
        return [r for r in self.results if r.status == "failed"]


def _entry_from_seed(s: SeedEntry, old: Optional[LibraryEntry]) -> LibraryEntry:
    prov = s.provenance.model_copy(deep=True) if s.provenance else None
    if prov is not None:
        prov.method = "builtin" if s.builtin and not s.download else "seed"
        if s.download and not prov.source_url:
            prov.source_url = s.download.url
        if not prov.license_url:
            prov.license_url = LICENSE_URLS.get(prov.license)
        prov.obtained = (old.provenance.obtained if old and old.provenance else None) or now_iso()
    return LibraryEntry(
        asset_id=s.asset_id, kind=s.kind, name=s.name, builtin=s.builtin, reusable=s.reusable,
        form=s.form, mapping=s.mapping, description=s.description, tags=list(s.tags),
        aliases=list(old.aliases) if old else [],
        files=list(old.files) if old else [],
        bounds=old.bounds if old else None, triangles=old.triangles if old else None,
        pixels=old.pixels if old else None,
        review="builtin" if s.builtin else (old.review if old else "auto"),
        provenance=prov, notes=s.notes,
    )


def seed_library(store: LocalFolderStore, seed_path: Path = SEED_FILE, *,
                 only: Optional[set[str]] = None, pin: bool = False, log=print) -> SeedReport:
    """Bring ``store`` up to date with the seed list. Downloads are fetched only for
    entries whose file is missing or differs from the pinned checksum. With ``pin``,
    checksums missing from the seed list are recorded in it; without it, an unpinned
    download is refused, and a download whose checksum changed is always refused."""
    seed = load_seed(seed_path)
    if only:
        unknown = sorted(only - {e.asset_id for e in seed.entries})
        if unknown:
            raise LibraryError(f"not in the seed list: {', '.join(unknown)}")
    report = SeedReport()
    seed_dirty = False
    for s in seed.entries:
        if only and s.asset_id not in only:
            continue
        old = store.get(s.asset_id)
        entry = _entry_from_seed(s, old)
        files: dict[str, Path] = {}
        retries = 0
        work = None
        new_pin = None
        try:
            d = s.download
            have = old.file(PRIMARY_ROLE[s.kind]) if old else None
            have_path = store.path_of(s.asset_id, have.role) if have else None
            # Skip the download only when the file on disk really is the pinned one.
            current = (have is not None and d is not None and d.sha256 == have.sha256
                       and have_path.is_file() and sha256_of(have_path) == d.sha256)
            if d is not None and not current:
                if not FILE_NAME.match(d.filename):
                    raise DownloadError(f"bad file name {d.filename!r}")
                work = Path(tempfile.mkdtemp(prefix="seed_dl_"))
                target = work / d.filename
                log(f"  downloading {s.asset_id} from {d.url}")
                retries = download(d.url, target)
                digest, size = sha256_of(target), target.stat().st_size
                if d.sha256 is None:
                    if not pin:
                        raise DownloadError("download is not pinned in the seed list "
                                            "(rerun with --pin to record its checksum)")
                    new_pin = (digest, size)  # recorded once the asset is accepted
                elif digest != d.sha256:
                    raise DownloadError(f"the source file changed (sha256 {digest[:12]}..., "
                                        f"pinned {d.sha256[:12]}...); not added")
                if s.kind == "texture":
                    fmt, px = _image_facts(target)
                    if fmt != "JPEG":
                        raise DownloadError(f"downloaded file is {fmt or 'not an image'}, not JPEG")
                    entry.pixels = list(px)
                    thumb = work / "thumbnail.png"
                    make_thumbnail(target, thumb)
                    files = {"texture": target, "thumbnail": thumb}
                else:
                    raise DownloadError("seeding model files is not supported yet "
                                        "(normalization comes with #113)")
            same = old is not None and not files and (
                old.model_dump(exclude={"added"}) == entry.model_dump(exclude={"added"}))
            if same:
                report.results.append(SeedResult(s.asset_id, "unchanged", retries=retries))
                continue
            store.put(entry, files)
            if new_pin is not None:
                d.sha256, d.bytes = new_pin
                seed_dirty = True
                report.pinned.append(s.asset_id)
            status = "updated" if old is not None else "added"
            report.results.append(SeedResult(s.asset_id, status, retries=retries))
        except (DownloadError, LibraryError) as exc:
            report.results.append(SeedResult(s.asset_id, "failed", str(exc), retries))
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
    if seed_dirty:
        write_json_atomic(Path(seed_path), seed.model_dump(exclude_none=True))
    return report


# ── CLI ───────────────────────────────────────────────────────────────


def _header(store: LocalFolderStore) -> str:
    i = store.identity()
    return (f"library {i['location']} (id {i['library_id'][:8]}, revision {i['revision']}, "
            f"content {i['content_hash']}, {i['entries']} entries)")


def _cmd_seed(args) -> int:
    store = open_store(args.library, create=True)
    print(_header(store))
    report = seed_library(store, Path(args.seed_file), only=set(args.only or []) or None,
                          pin=args.pin)
    for r in report.results:
        extra = f" ({r.retries} retries)" if r.retries else ""
        print(f"  {r.status:9} {r.asset_id}{extra}{': ' + r.detail if r.detail else ''}")
    if report.pinned:
        print(f"pinned {len(report.pinned)} checksum(s) in {args.seed_file}")
    print(_header(store))
    return 1 if report.failed else 0


def _cmd_validate(args) -> int:
    store = open_store(args.library)
    problems = store.validate()
    errors = [p for p in problems if p.level == "error"]
    if args.json:
        print(json.dumps({"library": store.identity(),
                          "problems": [p.__dict__ for p in problems]}, indent=2))
    else:
        print(_header(store))
        for p in problems:
            print(f"  {p}")
        print(f"{len(errors)} error(s), {len(problems) - len(errors)} warning(s)")
    return 1 if errors else 0


def _cmd_info(args) -> int:
    store = open_store(args.library)
    print(json.dumps(store.identity(), indent=2))
    entries = store.entries()
    for kind in ("model", "texture"):
        of_kind = [e for e in entries if e.kind == kind]
        print(f"{kind}s: {len(of_kind)} ({sum(e.builtin for e in of_kind)} built-in)")
    print("Results depend on this library's contents: compare library_id and content_hash "
          "when comparing runs from different machines.")
    return 0


def _cmd_list(args) -> int:
    store = open_store(args.library)
    entries = [e for e in store.entries() if not args.kind or e.kind == args.kind]
    if args.json:
        print(json.dumps([e.model_dump(exclude_none=True) for e in entries], indent=2))
        return 0
    print(_header(store))
    for e in entries:
        flags = ("builtin " if e.builtin else "") + (e.provenance.license if e.provenance else "")
        print(f"  {e.kind:7} {e.asset_id:24} {e.name:24} {flags.strip()}")
    return 0


def _cmd_find(args) -> int:
    store = open_store(args.library)
    e = store.find(args.name)
    if e is None:
        print(f"no entry named {args.name!r}")
        return 1
    print(json.dumps(e.model_dump(exclude_none=True), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Manage the provisioning asset library (#107).")
    sub = ap.add_subparsers(dest="command", required=True)

    def with_library(p):
        p.add_argument("--library", help=f"library folder (default: ${LIBRARY_ENV}, "
                                         f"else {DEFAULT_LIBRARY.relative_to(PROJECT_ROOT)})")
        return p

    p = with_library(sub.add_parser("seed", help="build or update a library from the seed list"))
    p.add_argument("--only", nargs="+", metavar="ID", help="seed only these asset ids")
    p.add_argument("--pin", action="store_true",
                   help="record missing download checksums in the seed list")
    p.add_argument("--seed-file", default=str(SEED_FILE), help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_seed)
    p = with_library(sub.add_parser("validate", help="check every entry and file"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_validate)
    with_library(sub.add_parser("info", help="identity and counts")).set_defaults(func=_cmd_info)
    p = with_library(sub.add_parser("list", help="list entries"))
    p.add_argument("--kind", choices=["model", "texture"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_list)
    p = with_library(sub.add_parser("find", help="look up a lab name (name or alias, any case)"))
    p.add_argument("name")
    p.set_defaults(func=_cmd_find)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except LibraryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
