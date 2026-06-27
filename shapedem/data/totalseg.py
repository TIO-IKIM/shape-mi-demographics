"""Access to the TotalSegmentator dataset (Zenodo record 8367088).

The dataset is a single 23.6 GB zip whose members are
``s####/ct.nii.gz`` and ``s####/segmentations/<organ>.nii.gz`` plus a
semicolon-delimited ``meta.csv`` at the root.

`Source` abstracts over two backends:
  * ``remote``: HTTP range requests via ``remotezip`` (pull a few members
    without downloading the whole archive — used for the smoke test);
  * ``local``: a downloaded zip on the shared volume (used for full runs).

Both expose the same API: ``list_subjects()``, ``read_meta()``,
``extract_member(member, dstdir) -> path``.
"""
from __future__ import annotations
import os
import re
import io
import csv
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pandas as pd

_SUBJ_RE = re.compile(r"^s\d+/")


def _parse_meta_bytes(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    df = pd.DataFrame(rows)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    if "age" in df:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


class Source:
    def __init__(self, backend: str, *, url: str | None = None, local_zip: str | None = None,
                 meta_member: str = "meta.csv"):
        assert backend in ("remote", "local")
        self.backend = backend
        self.url = url
        self.local_zip = local_zip
        self.meta_member = meta_member
        self._zf = None  # lazily opened handle

    def _new_handle(self):
        if self.backend == "local":
            if not self.local_zip or not os.path.exists(self.local_zip):
                raise FileNotFoundError(f"local zip not found: {self.local_zip}")
            return zipfile.ZipFile(self.local_zip)
        from remotezip import RemoteZip
        return RemoteZip(self.url)

    def open(self):
        """Open and hold a persistent handle (reused by all subsequent calls)."""
        if self._zf is None:
            self._zf = self._new_handle()
        return self

    def close(self):
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def _open(self):
        if self._zf is not None:
            yield self._zf            # reuse persistent handle; do not close it
        else:
            zf = self._new_handle()
            try:
                yield zf
            finally:
                zf.close()

    def list_subjects(self) -> list[str]:
        with self._open() as zf:
            names = zf.namelist()
        return sorted({n.split("/")[0] for n in names if _SUBJ_RE.match(n)})

    def read_meta(self) -> pd.DataFrame:
        with self._open() as zf:
            raw = zf.read(self.meta_member)
        return _parse_meta_bytes(raw)

    def extract_member(self, member: str, dstdir: str | os.PathLike) -> str | None:
        """Extract one member to dstdir (flat), return its path, or None if absent."""
        dstdir = Path(dstdir)
        dstdir.mkdir(parents=True, exist_ok=True)
        out = dstdir / Path(member).name
        with self._open() as zf:
            try:
                data = zf.read(member)
            except KeyError:
                return None
        out.write_bytes(data)
        return str(out)

    def extract_members(self, members: Iterable[str], dstdir: str | os.PathLike) -> dict[str, str]:
        """Extract several members in ONE zip-open (important for remote: avoids
        re-reading the central directory per file)."""
        dstdir = Path(dstdir)
        dstdir.mkdir(parents=True, exist_ok=True)
        got: dict[str, str] = {}
        with self._open() as zf:
            available = set(zf.namelist())
            for m in members:
                if m not in available:
                    continue
                try:
                    data = zf.read(m)              # can raise BadZipFile on rare corrupt members
                except Exception:
                    continue
                out = dstdir / m.replace("/", "__")
                out.write_bytes(data)
                got[m] = str(out)
        return got


def from_config(cfg, backend: str = "remote") -> Source:
    ts = cfg["datasets"]["totalseg"]
    return Source(
        backend,
        url=ts["zip_url"],
        local_zip=ts["local_zip"],
        meta_member=ts["meta_member"],
    )


def seg_member(cfg, subject: str, organ: str) -> str:
    return cfg["datasets"]["totalseg"]["seg_member_fmt"].format(subject=subject, organ=organ)


def build_label_table(cfg, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Add typed label columns (sex, age, pathology_binary) to the meta frame."""
    lab = cfg["labels"]
    out = df_meta.copy()
    # sex
    smap = lab["sex"]["map"]
    out["y_sex"] = out[lab["sex"]["col"]].map(smap)
    # age
    out["y_age"] = pd.to_numeric(out[lab["age"]["col"]], errors="coerce")
    # pathology binary: 1 if any pathology, 0 if in negatives, NaN if in drop
    pcol = lab["pathology_binary"]["col"]
    negs = set(lab["pathology_binary"]["negatives"])
    drop = set(lab["pathology_binary"]["drop"])
    def _path(v):
        v = (v or "").strip()
        if v in drop:
            return float("nan")
        return 0.0 if v in negs else 1.0
    out["y_pathology"] = out[pcol].map(_path)
    return out
