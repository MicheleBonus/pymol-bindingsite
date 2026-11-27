"""Reference complexes must show the interactions the literature describes; then the
whole workflow on one of them: labels, numbering offset, reports, pictures, camera,
separate objects, states and the gallery.

    python -m pytest tests/test_reference_complexes.py
    python tests/test_reference_complexes.py

Set BINDINGSITE_EXTRA_STRUCTURES to one or more folders (separated like PATH) to
also run the plugin on every .cif / .pdb in them.
"""
import csv
import glob
import os
import sys

import numpy as np
from pymol import cmd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import OUT, bs, case, check, found, labels, main, pdb, skip  # noqa: E402

# ---- reference complexes: the literature interactions must be among those found ------------
EXPECT = {
    # biotin - streptavidin
    "1stp": [("hbond", "Asn23"), ("hbond", "Ser27"), ("hbond", "Tyr43"), ("hbond", "Ser45"),
             ("hbond", "Asn49"), ("hbond", "Ser88"), ("hbond", "Asp128")],
    # acetazolamide - carbonic anhydrase II
    "3hs4": [("metal", "Zn301"), ("hbond", "Thr199"), ("metal", "His94"), ("metal", "His96"),
             ("metal", "His119")],
    # 4-hydroxytamoxifen - oestrogen receptor alpha
    "3ert": [("hbond", "Glu353"), ("hbond", "Arg394"), ("saltbridge", "Asp351")],
    # tetrabromobenzotriazole - CDK2
    "1p5e": [("halogen", "Leu83")],
    # tacrine - acetylcholinesterase
    "1acj": [("pistack", "Trp84"), ("pistack", "Phe330"), ("hbond", "His440")],
    # methotrexate - dihydrofolate reductase
    "4dfr": [("hbond", "Asp27"), ("saltbridge", "Arg57"), ("hbond", "Ile5"), ("hbond", "Ile94")],
    # covalent alkyl phosphonate - dog gastric lipase
    "1k8q": [("covalent", "Ser153"), ("hbond", "Leu67"), ("hbond", "Gln154")],
    # N3 - SARS-CoV-2 main protease
    "6lu7": [("covalent", "Cys145"), ("hbond", "Glu166"), ("hbond", "His163"), ("hbond", "Gly143")],
    # indinavir - HIV-1 protease
    "1hsg": [("hbond", "Asp25"), ("waterbridge", "Ile50")],
    # p53 peptide - MDM2
    "1ycr": [("hbond", "Leu54"), ("hbond", "Gln72")],
}


def _reference(pid):
    @case
    def test():
        pdb(pid)
        got = found(bs.bindingsite(quiet=1))
        missing = [e for e in EXPECT[pid] if e not in got]
        check(pid, not missing, "" if not missing else "missing %s" % missing)
    test.__name__ = test.__qualname__ = "test_reference_" + pid
    return test


for _pid in EXPECT:
    globals()["test_reference_" + _pid] = _reference(_pid)


# ---- any other structures you want covered ------------------------------------------------
def test_extra_structures():
    folders = [d for d in os.environ.get("BINDINGSITE_EXTRA_STRUCTURES", "").split(os.pathsep) if d]
    if not folders:
        skip("BINDINGSITE_EXTRA_STRUCTURES is not set")

    @case
    def run():
        files = sorted(f for d in folders for ext in ("*.cif", "*.pdb")
                       for f in glob.glob(os.path.join(d, ext)))
        check("structures found in %s" % os.pathsep.join(folders), bool(files))
        for f in files:
            cmd.reinitialize()
            cmd.viewport(1000, 750)
            cmd.load(f, "cplx")
            if not bs.find_ligands("enabled"):
                print("  --   %s: no ligand (apo)" % os.path.basename(f))
                continue
            try:
                site = bs.bindingsite(quiet=1)
            except Exception as err:
                check(os.path.basename(f), False, repr(err))
                continue
            n = sum(1 for r in site["inter"] if r["draw"] and r["type"] in bs.ITYPES)
            check(os.path.basename(f), True,
                  "%d interactions drawn, %d labels" % (n, len(site["label_boxes"])))
    run()


# ---- one complex from start to finish: 1K8Q, a lipase with a covalent phosphonate -----------
@case
def test_worked_example():
    pdb("1k8q")
    site = bs.bindingsite(quiet=1, offset=100, ligand_label="C11 phosphonate", distances=1)
    got = found(site)
    check("example: covalent link and oxyanion hole",
          {("covalent", "Ser153"), ("hbond", "Leu67"), ("hbond", "Gln154")} <= got, str(sorted(got)))
    check("example: phosphonate warhead recognised",
          any(v["kind"] == "phosph(on)ate ester" for i, v in site["chem"].electrophiles.items()
              if site["atoms"][i].lig))
    labs = labels()
    check("example: offset and ligand name in the labels",
          "Ser253" in labs and "Ser153" not in labs and "C11 phosphonate" in labs, str(sorted(labs)))

    rows = bs.bindingsite_report(os.path.join(OUT, "report.csv"), quiet=1)
    with open(os.path.join(OUT, "report.csv")) as fh:
        table = list(csv.DictReader(fh))
    check("report csv", len(table) == len(rows) > 5
          and any(r["type"] == "Covalent bond" and r["residue"] == "Ser253" for r in table),
          "%d rows" % len(table))
    bs.bindingsite_report(os.path.join(OUT, "report.md"), quiet=1)
    with open(os.path.join(OUT, "report.md")) as fh:
        md = fh.read().splitlines()
    check("report md", len(md) == len(rows) + 2 and md[0].startswith("| type | residue |"))
    bs.bindingsite_report(os.path.join(OUT, "report.tsv"), quiet=1)
    with open(os.path.join(OUT, "report.tsv")) as fh:
        check("report tsv", fh.readline().split("\t")[:2] == ["type", "residue"])

    bs.bindingsite_png(os.path.join(OUT, "transparent.png"), width=500, height=400,
                       transparent=1, quiet=1)
    check("png written", os.path.getsize(os.path.join(OUT, "transparent.png")) > 20000)

    view_before = cmd.get_view()
    bs.bindingsite(quiet=1, view=0)
    check("view=0 keeps the camera",
          max(abs(a - b) for a, b in zip(view_before[:9], cmd.get_view()[:9])) < 1e-4)
    bs.bindingsite_view()
    check("bindingsite_view turns the camera",
          max(abs(a - b) for a, b in zip(view_before[:9], cmd.get_view()[:9])) > 1e-3)
    bs.bindingsite_relabel()
    check("relabel keeps the labels", "Ser153" in labels(), str(sorted(labels())))

    bs.bindingsite(quiet=1, restyle=0, recolor=0, name="extra", labels=0, legend=0)
    check("second named site coexists",
          "bs" in cmd.get_names("all") and "extra" in cmd.get_names("all"))
    bs.bindingsite_clear()
    check("clear removes everything", cmd.get_names("all") == ["cplx"], str(cmd.get_names("all")))


# ---- a pose that overlaps the pocket is reported, not prettified ----------------------------
@case
def test_clash_is_reported():
    pdb("1stp")
    check("crystal pose: no clash", not any(t == "clash" for t, _ in found(bs.bindingsite(quiet=1))))
    # push biotin 1.5 A into the indole of Trp79
    lig = np.array(cmd.get_coords("resn BTN"))
    wall = np.array(cmd.get_coords("polymer and resi 79 and not name N+CA+C+O"))
    v = wall.mean(0) - lig.mean(0)
    cmd.translate((1.5 * v / np.linalg.norm(v)).tolist(), "resn BTN", camera=0)
    site = bs.bindingsite(quiet=1)
    clashes = sorted(res for t, res in found(site) if t == "clash")
    check("displaced pose: clash drawn", bool(clashes), str(clashes))
    rows = bs.bindingsite_report(quiet=1)
    check("... and in the report", any(r["type"] == "Steric clash" for r in rows))


# ---- ligand and receptor as separate objects; multi-state -----------------------------------
@case
def test_separate_objects_and_states():
    pdb("1stp")
    cmd.extract("lig", "cplx and organic")
    site = bs.bindingsite(quiet=1)
    check("separate objects", ("hbond", "Asp128") in found(site), str(sorted(found(site)))[:80])
    cmd.create("traj", "cplx", 1, 1)
    cmd.create("traj", "cplx", 1, 2)
    cmd.create("trajlig", "lig", 1, 1)
    cmd.create("trajlig", "lig", 1, 2)
    cmd.delete("cplx")
    cmd.delete("lig")
    cmd.frame(2)
    site = bs.bindingsite("trajlig", quiet=1)
    check("multi-state, state 2", site["state"] == 2 and ("hbond", "Asp128") in found(site))


# ---- the gallery: one scene per complex ------------------------------------------------------
@case
def test_gallery():
    # the two DHFR-methotrexate complexes of 4DFR as two separate objects
    pdb("4dfr", "both")
    cmd.create("dhfr_a", "both and chain A")
    cmd.create("dhfr_b", "both and chain B")
    cmd.delete("both")
    bs.bindingsite_all(quiet=1)
    check("gallery scenes", cmd.get_scene_list() == ["dhfr_a", "dhfr_b"], str(cmd.get_scene_list()))
    a, b = (np.array(cmd.centerofmass("%s and resn MTX" % obj)) for obj in ("dhfr_a", "dhfr_b"))
    apart = float(np.linalg.norm(a - b))
    check("gallery: complexes superposed, ligand on ligand", apart < 1.5, "%.2f A apart" % apart)
    for obj in ("dhfr_b", "dhfr_a"):
        cmd.scene(obj, "recall")
        out = os.path.join(OUT, "gallery_%s.png" % obj)
        bs.bindingsite_png(out, width=900, height=675, name="bs_" + obj, fit=0, quiet=1)
        check("gallery png %s" % obj, os.path.getsize(out) > 20000)


if __name__ == "__main__":
    sys.exit(main(globals()))
