"""Regenerate the figures and tables in this folder.

Every picture is one `bindingsite` call on a public PDB entry, followed by
`bindingsite_png`.  Run it with the Python that has PyMOL:

    python examples/make_examples.py            # all of them
    python examples/make_examples.py 1stp 3ert  # only these
"""
import os
import sys

from pymol import cmd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "tests", "pdb_cache")    # the entries are downloaded once
sys.path.insert(0, ROOT)
import bindingsite as bs  # noqa: E402

WIDTH, HEIGHT = 1600, 1200

# file name -> (PDB entry, options of the `bindingsite` call)
EXAMPLES = [
    ("1stp_biotin_streptavidin", "1stp",
     dict()),
    ("3ert_tamoxifen_surface_distances", "3ert",
     dict(surface="pocket", distances=1, ligand_label="4-hydroxytamoxifen")),
    ("1acj_tacrine_soft", "1acj",
     dict(style="soft", ligand_label="tacrine")),
    ("3hs4_acetazolamide_zinc_flat", "3hs4",
     dict(style="flat")),
    ("1k8q_covalent_phosphonate", "1k8q",
     dict(ligand_label="C11 phosphonate")),
    ("1ycr_p53_peptide_dark", "1ycr",
     dict(ligand="chain B", style="dark", label_format="{r}{resi}")),
    ("1p5e_halogen_bonds", "1p5e",
     dict(ligand_style="ballstick")),
    ("1hsg_indinavir_polar_surface", "1hsg",
     dict(surface="polar", hydrophobic=0)),
]


def load(pid):
    cmd.reinitialize()
    cmd.viewport(1000, 750)
    path = os.path.join(CACHE, pid + ".cif")
    if os.path.exists(path):
        cmd.load(path, pid)
    else:
        os.makedirs(CACHE, exist_ok=True)
        cmd.fetch(pid, pid, type="cif", path=CACHE, quiet=1)


def main(only):
    for name, pid, options in EXAMPLES:
        if only and pid not in only and name not in only:
            continue
        load(pid)
        bs.bindingsite(quiet=1, **options)
        bs.bindingsite_png(os.path.join(HERE, name + ".png"), width=WIDTH, height=HEIGHT, quiet=1)
        print("wrote %s.png   <- bindingsite %s" % (name, ", ".join("%s=%s" % kv for kv in options.items())))
        if pid == "1stp":
            bs.bindingsite_report(os.path.join(HERE, name + ".csv"), quiet=1)
            bs.bindingsite_report(os.path.join(HERE, name + ".md"), quiet=1)
            print("wrote %s.csv and .md" % name)


if __name__ == "__main__":
    main(set(a.lower() for a in sys.argv[1:]))
