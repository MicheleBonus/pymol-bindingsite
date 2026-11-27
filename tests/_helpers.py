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
