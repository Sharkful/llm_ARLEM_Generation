"""
Free, openly licensed sources to search for models and textures (#110; spec:
PROVISIONING.md, "Search"). One small adapter per source, each returning the same
``Candidate`` record; ``search_assets.py`` ranks them and fetches the chosen one.

Built so far, the sources that need no key or login, their terms checked:

- **Poly Haven** (``polyhaven``; models and textures; CC0). ``/search`` ranks assets for
  a query; ``/assets`` gives names, tags and thumbnails; ``/files/{id}`` the files.
  Models come as glTF with their buffers and textures (``include``), converted to GLB
  on fetch. Terms (Public-API ToS.md): free for any use, no key; every call must carry
  a User-Agent naming the software (``library.USER_AGENT``); live API content shown to
  people must credit Poly Haven (stored CC0 assets need no credit).
- **NASA 3D Resources** (``nasa``; models; no copyright, NASA media guidelines: no
  endorsement, no NASA insignia use). The GitHub repository has one folder per model
  under ``3D Models/`` with a ``.glb`` (and usually a ``.png``); the tree is listed once
  per process through the GitHub API, and names are matched by words. A file stored
  with Git LFS is fetched from ``media.githubusercontent.com``.

- **ambientCG** (``ambientcg``; textures; CC0, credit not required but appreciated,
  kept as a note). ``/api/v3/assets?type=material&q=...`` (keywords combine with AND;
  each query alone, then single words if nothing matched). Downloads are zips per
  resolution; the 2K JPG zip (else 1K) is fetched and only its color map is read from
  it, never extracted, and only for materials that have a ``color`` map.

Not built yet (#110): Poly Pizza (a free key) and Smithsonian Open Access (an
api.data.gov key); their docs couldn't be checked from the build machine.

Every request goes through ``library.download`` / ``library.fetch_json``: the shared
retry policy, the size limit and the network-failure classification. Base URLs are
module attributes, so tests point them at a mock server.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import library  # noqa: E402

POLYHAVEN_API = "https://api.polyhaven.com"
POLYHAVEN_PAGE = "https://polyhaven.com/a/{slug}"
AMBIENTCG_API = "https://ambientcg.com/api/v3"
AMBIENTCG_ZIPS = ("2K-JPG", "1K-JPG", "2K-PNG", "1K-PNG")   # preferred first; never 4K+ (huge)
MAX_MAP_BYTES = 64 * 1024 * 1024   # the color map read out of a zip
GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
GITHUB_MEDIA = "https://media.githubusercontent.com/media"
NASA_REPO, NASA_BRANCH = "nasa/NASA-3D-Resources", "master"
NASA_GUIDELINES = "https://www.nasa.gov/nasa-brand-center/images-and-media"

PER_SOURCE = 15          # hits asked of each source per asset
TEXTURE_MAPS = ("diffuse", "diff", "color", "albedo", "basecolor")   # a texture's color map
TEXTURE_RESOLUTIONS = ("2k", "1k", "4k")   # preferred first; resized to the budget later
LFS_POINTER = b"version https://git-lfs.github.com/spec/"
_WORD = re.compile(r"[a-z0-9]+")


class SourceError(Exception):
    """A source couldn't be searched or a file fetched; the message says why."""


@dataclass
class Candidate:
    source: str                  # "polyhaven", "nasa"
    id: str                      # the source's own id
    kind: str                    # "model" or "texture"
    name: str
    page_url: str
    thumbnail_url: Optional[str]
    license: str                 # a library.ALLOWED_LICENSES key, or what the source says
    author: Optional[str] = None
    format: str = ""             # "glb", "gltf", "jpg"...
    triangles: Optional[int] = None
    file_size: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    description: str = ""
    source_score: float = 0.0    # the source's own relevance, 0-1 (0 when it has none)
    license_url: Optional[str] = None
    notes: str = ""
    mapping: Optional[str] = None   # textures: "tileable", "uv"... (library.LibraryEntry.mapping)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    def to_json(self) -> dict:
        return asdict(self)


def words(text: str) -> set[str]:
    """Lower-case words of 3+ letters, camelCase split (``dnaHelix`` -> dna, helix)."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def _json(url: str):
    try:
        return library.fetch_json(url)
    except library.DownloadError as exc:
        raise SourceError(str(exc)) from exc
    except ValueError as exc:
        raise SourceError(str(exc)) from exc


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        library.download(url, dest)
    except library.DownloadError as exc:
        raise SourceError(str(exc)) from exc


class Source:
    name = ""
    kinds: tuple[str, ...] = ()

    def search(self, queries: list[str], kind: str, limit: int = PER_SOURCE) -> list[Candidate]:
        raise NotImplementedError

    def fetch(self, cand: Candidate, dest_dir: Path) -> Path:
        """Download the candidate into ``dest_dir``; returns the main file (a GLB, a
        glTF with its files beside it, or a texture image)."""
        raise NotImplementedError


# ── Poly Haven ────────────────────────────────────────────────────────


class PolyHaven(Source):
    name = "polyhaven"
    kinds = ("model", "texture")
    _TYPES = {"model": "models", "texture": "textures"}

    def __init__(self):
        self._assets: dict[str, dict] = {}

    def _catalog(self, kind: str) -> dict:
        """Every asset of a type (slug -> info), once per process."""
        t = self._TYPES[kind]
        if t not in self._assets:
            # The spec documents ``type``; older docs and examples use ``t``. Both are sent.
            self._assets[t] = _json(f"{POLYHAVEN_API}/assets?type={t}&t={t}") or {}
        return self._assets[t]

    def search(self, queries, kind, limit=PER_SOURCE):
        catalog = self._catalog(kind)
        scores: dict[str, float] = {}
        for q in queries:
            reply = None
            try:
                reply = _json(f"{POLYHAVEN_API}/search?q={urllib.parse.quote(q[:100])}"
                              f"&t={self._TYPES[kind]}&limit={limit}")
            except SourceError:
                reply = None   # the ranked search is new; fall back to the catalog below
            for hit in (reply or {}).get("results") or []:
                slug = hit.get("slug")
                if slug in catalog:
                    scores[slug] = max(scores.get(slug, 0.0), float(hit.get("score") or 0.0))
        if not scores:   # no ranked search: match words against names, tags and categories
            want = set().union(*(words(q) for q in queries)) if queries else set()
            for slug, info in catalog.items():
                have = words(" ".join([info.get("name", ""), *info.get("tags", []),
                                       *info.get("categories", [])]))
                if want & have:
                    scores[slug] = len(want & have) / len(want)
        ranked = sorted(scores, key=lambda s: (-scores[s], s))[:limit]
        return [self._candidate(s, catalog[s], kind, scores[s]) for s in ranked]

    def _candidate(self, slug, info, kind, score) -> Candidate:
        authors = info.get("authors") or {}
        return Candidate(
            source=self.name, id=slug, kind=kind, name=info.get("name") or slug,
            page_url=POLYHAVEN_PAGE.format(slug=slug), thumbnail_url=info.get("thumbnail_url"),
            license="CC0-1.0", author=", ".join(authors) or None,
            format="gltf" if kind == "model" else "jpg",
            tags=list(dict.fromkeys([*(info.get("tags") or []), *(info.get("categories") or [])])),
            description=info.get("description") or "", source_score=score,
            license_url=library.LICENSE_URLS["CC0-1.0"], mapping="tileable" if kind == "texture" else None)

    def fetch(self, cand, dest_dir):
        files = _json(f"{POLYHAVEN_API}/files/{urllib.parse.quote(cand.id)}")
        if not files:
            raise SourceError(f"Poly Haven has no files for {cand.id!r}")
        if cand.kind == "model":
            by_res = files.get("gltf") or {}
            for res in (*TEXTURE_RESOLUTIONS, *sorted(by_res)):
                entry = (by_res.get(res) or {}).get("gltf")
                if entry and entry.get("url"):
                    break
            else:
                raise SourceError(f"Poly Haven has no glTF for {cand.id!r}")
            main = dest_dir / Path(urllib.parse.urlparse(entry["url"]).path).name
            _download(entry["url"], main)
            for rel, inc in (entry.get("include") or {}).items():
                target = (dest_dir / rel).resolve()
                if not target.is_relative_to(dest_dir.resolve()):
                    raise SourceError(f"Poly Haven listed a file outside the model's folder: {rel!r}")
                _download(inc["url"], target)
            return main
        maps = {k.lower(): v for k, v in files.items() if isinstance(v, dict)}
        color = next((maps[m] for m in TEXTURE_MAPS if m in maps), None)
        if color is None:
            raise SourceError(f"Poly Haven's {cand.id!r} has no color map")
        for res in (*TEXTURE_RESOLUTIONS, *sorted(color)):
            fmt = (color.get(res) or {}).get("jpg") or (color.get(res) or {}).get("png")
            if fmt and fmt.get("url"):
                dest = dest_dir / f"texture{Path(urllib.parse.urlparse(fmt['url']).path).suffix}"
                _download(fmt["url"], dest)
                return dest
        raise SourceError(f"Poly Haven's {cand.id!r} has no JPG or PNG color map")


# ── NASA 3D Resources ─────────────────────────────────────────────────


class Nasa(Source):
    name = "nasa"
    kinds = ("model",)

    def __init__(self):
        self._models: Optional[list[dict]] = None
        self.truncated = False   # GitHub cut the tree listing short (very large repos)

    def _list(self) -> list[dict]:
        """One record per model folder with a .glb: {name, glb, image, size}."""
        if self._models is None:
            tree = _json(f"{GITHUB_API}/repos/{NASA_REPO}/git/trees/{NASA_BRANCH}?recursive=1") or {}
            folders: dict[str, dict] = {}
            for item in tree.get("tree") or []:
                path = item.get("path", "")
                parts = path.split("/")
                if item.get("type") != "blob" or len(parts) < 3 or parts[0] != "3D Models":
                    continue
                rec = folders.setdefault(parts[1], {"name": parts[1]})
                low = path.lower()
                if low.endswith(".glb") and "glb" not in rec:
                    rec["glb"], rec["size"] = path, item.get("size")
                elif low.endswith((".png", ".jpg", ".jpeg")) and "image" not in rec:
                    rec["image"] = path
            self._models = [r for _, r in sorted(folders.items()) if "glb" in r]
            self.truncated = bool(tree.get("truncated"))
        return self._models

    @staticmethod
    def _raw(path: str) -> str:
        return f"{GITHUB_RAW}/{NASA_REPO}/{NASA_BRANCH}/{urllib.parse.quote(path)}"

    def search(self, queries, kind, limit=PER_SOURCE):
        if kind != "model":
            return []
        want = set().union(*(words(q) for q in queries)) if queries else set()
        scored = []
        for rec in self._list():
            have = words(rec["name"])
            if want & have:
                scored.append((len(want & have) / len(want), rec))
        scored.sort(key=lambda x: (-x[0], x[1]["name"]))
        return [Candidate(
            source=self.name, id=rec["glb"], kind="model", name=rec["name"],
            page_url=f"https://github.com/{NASA_REPO}/tree/{NASA_BRANCH}/"
                     f"{urllib.parse.quote(rec['glb'].rsplit('/', 1)[0])}",
            thumbnail_url=self._raw(rec["image"]) if rec.get("image") else None,
            license="public-domain", author="NASA", format="glb", file_size=rec.get("size"),
            source_score=score, license_url=NASA_GUIDELINES,
            notes="NASA media: no copyright; don't imply NASA endorsement or use the NASA insignia")
            for score, rec in scored[:limit]]

    def fetch(self, cand, dest_dir):
        dest = dest_dir / "model.glb"
        _download(self._raw(cand.id), dest)
        if dest.read_bytes()[:len(LFS_POINTER)] == LFS_POINTER:   # stored with Git LFS
            _download(f"{GITHUB_MEDIA}/{NASA_REPO}/{NASA_BRANCH}/{urllib.parse.quote(cand.id)}", dest)
        return dest


# ── ambientCG ─────────────────────────────────────────────────────────


class AmbientCG(Source):
    name = "ambientcg"
    kinds = ("texture",)
    CREDIT = ("Created using {title} from ambientCG.com, licensed under the Creative Commons "
              "CC0 1.0 Universal License.")

    def _query(self, q: str, limit: int) -> list[dict]:
        reply = _json(f"{AMBIENTCG_API}/assets?type=material&q={urllib.parse.quote(q)}&limit={limit}"
                      "&include=title,tags,url,downloads,thumbnails,maps,dimensions") or {}
        return [a for a in reply.get("assets") or [] if isinstance(a, dict) and a.get("id")]

    def search(self, queries, kind, limit=PER_SOURCE):
        if kind != "texture":
            return []
        found: dict[str, dict] = {}
        for q in queries:
            for a in self._query(q, limit):
                found.setdefault(a["id"], a)
        if not found:   # keywords combine with AND: try each word alone
            for w in sorted(set().union(*(words(q) for q in queries)) if queries else set()):
                for a in self._query(w, limit):
                    found.setdefault(a["id"], a)
        out = []
        for a in found.values():
            if "color" not in (a.get("maps") or []):
                continue
            zips = {d.get("attributes"): d for d in a.get("downloads") or [] if d.get("extension") == "zip"}
            pick = next((zips[z] for z in AMBIENTCG_ZIPS if z in zips), None)
            if pick is None:
                continue
            title = a.get("title") or a["id"]
            out.append(Candidate(
                source=self.name, id=a["id"], kind="texture", name=title,
                page_url=a.get("url") or f"https://ambientcg.com/a/{a['id']}",
                thumbnail_url=(a.get("thumbnails") or {}).get("256-PNG"), license="CC0-1.0",
                author="ambientCG", format=pick["attributes"].split("-")[1].lower(),
                file_size=pick.get("size"), tags=list(a.get("tags") or []),
                license_url=library.LICENSE_URLS["CC0-1.0"], mapping="tileable",
                notes=self.CREDIT.format(title=title)))
        return out[:limit]

    def fetch(self, cand, dest_dir):
        reply = _json(f"{AMBIENTCG_API}/assets?id={urllib.parse.quote(cand.id)}&include=downloads") or {}
        asset = next((a for a in reply.get("assets") or [] if a.get("id") == cand.id), None)
        zips = {d.get("attributes"): d for d in (asset or {}).get("downloads") or [] if d.get("extension") == "zip"}
        pick = next((zips[z] for z in AMBIENTCG_ZIPS if z in zips), None)
        if pick is None:
            raise SourceError(f"ambientCG has no 1K or 2K download for {cand.id!r}")
        archive = dest_dir / "download.zip"
        _download(pick["url"], archive)
        try:
            with zipfile.ZipFile(archive) as z:
                colors = [i for i in z.infolist() if not i.is_dir()
                          and re.search(r"_color\.(jpe?g|png)$", i.filename, re.IGNORECASE)]
                if not colors:
                    raise SourceError(f"ambientCG's {cand.id!r} zip has no color map")
                info = colors[0]
                if info.file_size > MAX_MAP_BYTES:
                    raise SourceError(f"ambientCG's {cand.id!r} color map is larger than {MAX_MAP_BYTES} bytes")
                # read the one member's bytes: nothing is extracted, so no path in the zip is used
                dest = dest_dir / f"texture{Path(info.filename).suffix.lower()}"
                dest.write_bytes(z.read(info))
        except zipfile.BadZipFile as exc:
            raise SourceError(f"ambientCG's {cand.id!r} download isn't a zip ({exc})") from exc
        finally:
            archive.unlink(missing_ok=True)
        return dest


SOURCES: dict[str, Source] = {s.name: s for s in (PolyHaven(), Nasa(), AmbientCG())}


def sources_for(kind: str, names: Optional[list[str]] = None) -> list[Source]:
    return [s for n, s in SOURCES.items() if kind in s.kinds and (not names or n in names)]


def fetch_thumbnail(cand: Candidate, dest: Path) -> bool:
    """The candidate's thumbnail as a PNG (at most library.THUMBNAIL_SIDE); False if it
    has none or it can't be fetched or read."""
    if not cand.thumbnail_url:
        return False
    raw = dest.with_suffix(".download")
    try:
        _download(cand.thumbnail_url, raw)
        library.make_thumbnail(raw, dest)   # any format Pillow reads (webp too) -> PNG
        return True
    except (SourceError, OSError, ValueError):
        dest.unlink(missing_ok=True)
        return False
    finally:
        raw.unlink(missing_ok=True)
