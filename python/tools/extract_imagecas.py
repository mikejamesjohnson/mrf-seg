"""Pull the CT volumes out of the ImageCAS Kaggle archives without any other tool.

    python tools/extract_imagecas.py ../data/casx/archive ../data/casx/images \
        [--filelist ../data/casx/ImageCAS-X_dataset/filelist] [--labels-too]

Kaggle's ImageCAS comes as five split zips -- `1-200.z01` ... `1-200.z04` plus `1-200.change2zip`,
the archive's final `.zip` part under a name Kaggle leaves alone -- each holding
`<range>/<id>.img.nii.gz` and the original `<id>.label.nii.gz`. `unzip` cannot read split
archives and Python's zipfile refuses them, so this reads them itself: the central directory from
the last part, then each wanted member from whichever part it starts in, carrying on into the
next part where a member straddles two. Nothing is renamed or joined. Only `*.img.nii.gz` are
taken (`--labels-too` adds the ImageCAS labels, which ImageCAS-X replaces), and with
`--filelist` only the scans in its train / val / test lists. Members are written flat into OUT
as `<id>.img.nii.gz`, CRC-checked, and skipped when already there.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

LOCAL_HEADER = b"PK\x03\x04"
CENTRAL_HEADER = b"PK\x01\x02"
EOCD = b"PK\x05\x06"
ZIP64_EOCD = b"PK\x06\x06"
ZIP64_LOCATOR = b"PK\x06\x07"


class SplitZip:
    """A split zip archive: `stem.z01`, `stem.z02`, ... and its last part (`stem.zip`, or
    `stem.change2zip` as Kaggle has it)."""

    def __init__(self, last_part: Path) -> None:
        stem = last_part.name.rsplit(".", 1)[0]
        parts = sorted(last_part.parent.glob(f"{stem}.z[0-9][0-9]"), key=lambda p: int(p.suffix[2:]))
        self.parts = parts + [last_part]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.entries = self._central_directory()

    def _central_directory(self) -> list[dict]:
        last = self.parts[-1]
        with open(last, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536 - 22))
            tail = f.read()
            i = tail.rfind(EOCD)
            if i < 0:
                raise ValueError(f"{last}: no end-of-central-directory record; is this the archive's last part?")
            _, disk, cd_disk, _, n_total, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
            if 0xFFFF in (n_total, disk, cd_disk) or 0xFFFFFFFF in (cd_size, cd_offset):
                j = tail.rfind(ZIP64_EOCD)
                _, _, _, _, disk, cd_disk, _, n_total, cd_size, cd_offset = struct.unpack("<IQHHIIQQQQ", tail[j:j + 56])
            if cd_disk != len(self.parts) - 1:
                raise ValueError(f"{last}: the central directory is on part {cd_disk + 1}, not the last part; parts missing?")
            f.seek(cd_offset)
            cd = f.read(cd_size)
        entries, pos = [], 0
        while pos + 46 <= len(cd) and cd[pos:pos + 4] == CENTRAL_HEADER:
            (_, _, _, flags, method, _, _, crc, csize, usize, nlen, elen, clen, dnum, _, _, offset) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
            name = cd[pos + 46:pos + 46 + nlen].decode("utf-8")
            extra = cd[pos + 46 + nlen:pos + 46 + nlen + elen]
            if 0xFFFFFFFF in (csize, usize, offset) or dnum == 0xFFFF:       # zip64 extra field
                csize, usize, offset, dnum = _zip64(extra, csize, usize, offset, dnum)
            entries.append(dict(name=name, method=method, flags=flags, crc=crc, csize=csize, usize=usize, disk=dnum, offset=offset))
            pos += 46 + nlen + elen + clen
        if len(entries) != n_total:
            raise ValueError(f"{last}: read {len(entries)} central directory entries, expected {n_total}")
        return entries

    def _reader(self, disk: int, offset: int):
        """Bytes from `offset` on part `disk`, continuing into the following parts."""
        for d in range(disk, len(self.parts)):
            with open(self.parts[d], "rb") as f:
                f.seek(offset)
                while True:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
                    yield chunk
            offset = 0

    def extract(self, entry: dict, out: Path) -> None:
        stream = self._reader(entry["disk"], entry["offset"])
        buffer = b""
        for chunk in stream:
            buffer += chunk
            if len(buffer) >= 30:
                break
        if buffer[:4] != LOCAL_HEADER:
            raise ValueError(f"{entry['name']}: no local file header at part {entry['disk'] + 1} offset {entry['offset']}")
        nlen, elen = struct.unpack("<HH", buffer[26:30])
        header = 30 + nlen + elen
        while len(buffer) < header:
            buffer += next(stream)
        data = buffer[header:]
        remaining = entry["csize"] - len(data)
        if entry["method"] not in (0, 8):
            raise ValueError(f"{entry['name']}: compression method {entry['method']} is not supported")
        inflate = zlib.decompressobj(-15) if entry["method"] == 8 else None
        crc = 0
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f:
            def write(compressed: bytes) -> None:
                nonlocal crc
                plain = inflate.decompress(compressed) if inflate else compressed
                crc = zlib.crc32(plain, crc)
                f.write(plain)

            if remaining < 0:                                   # the first chunks held more than this member
                data = data[:entry["csize"]]
                remaining = 0
            write(data)
            for chunk in stream:
                if remaining <= 0:
                    break
                piece = chunk[:remaining]
                remaining -= len(piece)
                write(piece)
            if inflate:
                plain = inflate.flush()
                crc = zlib.crc32(plain, crc)
                f.write(plain)
        if crc != entry["crc"]:
            tmp.unlink()
            raise ValueError(f"{entry['name']}: CRC mismatch after extraction")
        tmp.replace(out)


def _zip64(extra: bytes, csize: int, usize: int, offset: int, disk: int) -> tuple[int, int, int, int]:
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[pos:pos + 4])
        if tag == 0x0001:
            fields, q = [], pos + 4
            if usize == 0xFFFFFFFF:
                usize, = struct.unpack("<Q", extra[q:q + 8]); q += 8
            if csize == 0xFFFFFFFF:
                csize, = struct.unpack("<Q", extra[q:q + 8]); q += 8
            if offset == 0xFFFFFFFF:
                offset, = struct.unpack("<Q", extra[q:q + 8]); q += 8
            if disk == 0xFFFF:
                disk, = struct.unpack("<I", extra[q:q + 4])
            return csize, usize, offset, disk
        pos += 4 + size
    return csize, usize, offset, disk


def wanted_ids(filelist: Path | None) -> set[str] | None:
    if filelist is None:
        return None
    ids = set()
    for split in ("train", "val", "test"):
        path = filelist / f"{split}.txt"
        if path.exists():
            ids |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive", help="the folder of Kaggle parts (*.z01 ... and *.change2zip or *.zip)")
    ap.add_argument("out", help="where the .nii.gz files go, flat")
    ap.add_argument("--filelist", help="ImageCAS-X filelist/: take only the scans in its train/val/test lists")
    ap.add_argument("--labels-too", action="store_true", help="also the original ImageCAS .label.nii.gz")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    archive, out = Path(args.archive), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = wanted_ids(Path(args.filelist) if args.filelist else None)
    last_parts = sorted([*archive.glob("*.change2zip"), *archive.glob("*.zip")])
    if not last_parts:
        print(f"no *.change2zip or *.zip in {archive}", file=sys.stderr)
        return 1
    started, done, skipped, failed = time.perf_counter(), 0, 0, []
    for last in last_parts:
        try:
            zf = SplitZip(last)
        except (OSError, ValueError) as error:
            print(f"{last.name}: {error}", file=sys.stderr)
            failed.append(last.name)
            continue
        members = [e for e in zf.entries if e["name"].endswith(".img.nii.gz") or (args.labels_too and e["name"].endswith(".label.nii.gz"))]
        if ids is not None:
            members = [e for e in members if e["name"].rsplit("/", 1)[-1].split(".", 1)[0] in ids]
        print(f"{last.name}: {len(zf.parts)} parts, {len(zf.entries)} members, {len(members)} to extract", flush=True)
        for entry in members:
            target = out / entry["name"].rsplit("/", 1)[-1]
            if target.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                zf.extract(entry, target)
                done += 1
                print(f"  {target.name}  {entry['usize'] / 1e6:.0f} MB  ({done} done, {time.perf_counter() - started:.0f} s)", flush=True)
            except (OSError, ValueError) as error:
                print(f"  {entry['name']}: {error}", file=sys.stderr)
                failed.append(entry["name"])
    print(f"{done} extracted, {skipped} already there, {len(failed)} failed, {(time.perf_counter() - started) / 60:.1f} min")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
