"""Shared helpers for the test files.

The tests run under pytest and as plain scripts, so that the Python that ships
with PyMOL is enough:

    python -m pytest tests
    python tests/test_reference_complexes.py
"""
import functools
import importlib.util
import os
import sys
import tempfile
import traceback

from pymol import cmd

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(os.path.dirname(HERE), "bindingsite.py")
CACHE = os.path.join(HERE, "pdb_cache")             # the reference PDB entries (mmCIF)
OUT = tempfile.mkdtemp(prefix="bindingsite_test_")  # images and tables written by the tests

_failed = []


def _load_plugin():
    """The plugin of this checkout, loaded once and shared by all test files."""
    if "bindingsite" not in sys.modules:
        spec = importlib.util.spec_from_file_location("bindingsite", PLUGIN)
        module = importlib.util.module_from_spec(spec)
        sys.modules["bindingsite"] = module
        spec.loader.exec_module(module)
    return sys.modules["bindingsite"]


bs = _load_plugin()


def fresh():
    """An empty PyMOL session and a plugin that remembers nothing."""
    cmd.reinitialize()
    bs._SITES.clear()
    bs._SAVED.clear()
    bs._GALLERY["scenes"] = []
    bs._GALLERY["objs"] = []
    cmd.viewport(1000, 750)


def pdb(pid, name="cplx"):
    """Load a PDB entry from tests/pdb_cache, downloading it if it is not there."""
    path = os.path.join(CACHE, pid + ".cif")
    if os.path.exists(path):
        cmd.load(path, name)
    else:
        os.makedirs(CACHE, exist_ok=True)
        cmd.fetch(pid, name, type="cif", path=CACHE, quiet=1)


def found(site, drawn=True):
    """{(interaction type, residue)} of a site, e.g. ("hbond", "Ser45")."""
    out = set()
    for r in site["inter"]:
        if r["type"] in bs.ITYPES and (r["draw"] or not drawn) and r["res"]:
            out.add((r["type"], "%s%s" % (r["res"][4].capitalize(), r["res"][3])))
    return out


def labels(name="bs"):
    """The label texts of a site."""
    out = []
    cmd.iterate(name + ".labels", "out.append(label)", space={"out": out})
    return [text for text in out if text]


def pixels(path):
    """The pixels of a PNG as an array (rows, columns, channels).

    A reader for the 8-bit, non-interlaced files PyMOL writes, so that the tests
    need no imaging library.
    """
    import struct
    import zlib
    import numpy as np
    with open(path, "rb") as handle:
        data = handle.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "%s is not a PNG" % path
    pos, packed = 8, b""
    while pos < len(data):
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + size]
        pos += 12 + size
        if kind == b"IHDR":
            w, h, depth, colour, _, _, interlaced = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            packed += body
    assert depth == 8 and not interlaced, "unsupported PNG: %s" % path
    n = {0: 1, 2: 3, 4: 2, 6: 4}[colour]            # bytes per pixel
    rows = np.frombuffer(zlib.decompress(packed), dtype=np.uint8).reshape(h, 1 + w * n)
    out = np.zeros((h, w * n), dtype=np.uint8)
    above = np.zeros(w * n, dtype=np.int64)
    for y in range(h):
        kind, line = int(rows[y, 0]), rows[y, 1:].astype(np.int64)
        if kind == 0:                               # none
            cur = line
        elif kind == 1:                             # sub: the pixel to the left
            cur = np.cumsum(line.reshape(w, n), axis=0).reshape(-1) & 255
        elif kind == 2:                             # up
            cur = (line + above) & 255
        else:                                       # average, Paeth: one byte after the other
            got, up, src = [0] * (w * n), above.tolist(), line.tolist()
            for i in range(w * n):
                a = got[i - n] if i >= n else 0
                b = up[i]
                if kind == 3:
                    guess = (a + b) >> 1
                else:
                    c = up[i - n] if i >= n else 0
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    guess = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                got[i] = (src[i] + guess) & 255
            cur = np.array(got, dtype=np.int64)
        out[y] = cur
        above = cur
    return out.reshape(h, w, n)


def differing(a, b, by=40):
    """Fraction of the pixels of two pictures that differ by more than `by` in a channel."""
    import numpy as np
    if a.shape != b.shape:
        return 1.0
    return float((np.abs(a.astype(int) - b.astype(int)).max(axis=2) > by).mean())


def check(tag, cond, msg=""):
    """A soft assertion: every check of a test is printed, the test fails at its end."""
    print("  %-4s %s %s" % ("ok" if cond else "FAIL", tag, msg))
    if not cond:
        _failed.append(tag)


def case(fn):
    """A test: starts from a fresh session and fails if any of its checks did."""
    @functools.wraps(fn)
    def wrapper():
        del _failed[:]
        fresh()
        fn()
        assert not _failed, "failed: " + "; ".join(_failed)
    return wrapper


class Skip(Exception):
    pass


def skip(reason):
    try:
        import pytest
    except ImportError:
        raise Skip(reason)
    pytest.skip(reason)


def main(namespace):
    """Run the tests of a module without pytest; returns the exit code."""
    failed = []
    for name, fn in list(namespace.items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        print(name)
        try:
            fn()
        except AssertionError as err:
            failed.append("%s (%s)" % (name, err))
        except Exception as err:
            if isinstance(err, Skip) or type(err).__name__ == "Skipped":
                print("  --   skipped: %s" % err)
                continue
            traceback.print_exc()
            failed.append("%s (exception)" % name)
        except BaseException as err:        # pytest.skip() raises a BaseException
            if type(err).__name__ != "Skipped":
                raise
            print("  --   skipped: %s" % err)
    print("\nFAILED:\n  " + "\n  ".join(failed) if failed else "\nALL CHECKS PASSED")
    return 1 if failed else 0
