"""Edge cases, each pinned after it went wrong once: names that must never be deleted,
per-object states, duplicate bonds, chain breaks, valence-filled hydrogens, quoted
prompt arguments, clean-up, alternate locations, labels in pictures larger than the
window, the dialog.

    python -m pytest tests/test_edge_cases.py
    python tests/test_edge_cases.py
"""
import os
import sys

from pymol import cmd, CmdException

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (OUT, bs, case, check, differing, found, fresh, labels, main, pdb,  # noqa: E402
                      pixels, skip)


def two_complexes(first, second):
    """The two DHFR-methotrexate complexes of 4DFR as two separate objects."""
    pdb("4dfr", "both")
    cmd.create(first, "both and chain A")
    cmd.create(second, "both and chain B")
    cmd.delete("both")


def refused(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except CmdException:
        return True
    return False


# ---- names: nothing of the user's is ever deleted ---------------------------------------------
@case
def test_user_names_survive():
    pdb("1stp", "bs_complex")
    bs.bindingsite(quiet=1)
    check("object 'bs_complex' survives", "bs_complex" in cmd.get_names("objects"))
    bs.bindingsite(quiet=1)
    check("... and a second run", "bs_complex" in cmd.get_names("objects"))
    check("name of an existing object is refused",
          refused(bs.bindingsite, quiet=1, name="bs_complex") and "bs_complex" in cmd.get_names("objects"))
    cmd.group("mine", "bs_complex")
    check("a foreign group name is refused",
          refused(bs.bindingsite, quiet=1, name="mine") and "bs_complex" in cmd.get_names("objects"))
    bs.bindingsite_clear()
    check("clear leaves the molecule", "bs_complex" in cmd.get_names("objects"))
    fresh()
    two_complexes("lead", "lead_min")
    bs.bindingsite_all(quiet=1)
    names = cmd.get_names("all")
    check("gallery: 'lead' and 'lead_min' do not collide",
          all(x in names for x in ("bs_lead.hbond", "bs_lead_min.hbond", "bs_lead_ligand", "bs_lead_min_ligand")))


# ---- per-object states ------------------------------------------------------------------------
@case
def test_per_object_states():
    # a docking session: one receptor, three poses in the states of a second object
    pdb("1stp")
    cmd.extract("poses", "cplx and organic")
    cmd.set_name("cplx", "receptor")
    for k in (2, 3):
        cmd.create("poses", "poses", 1, k)
    cmd.translate([0.4, 0.0, 0.0], "poses", state=3, camera=0)
    res = {}
    for k in (1, 2, 3):
        cmd.frame(k)
        site = bs.bindingsite(quiet=1)
        res[k] = len(site["residues"]), ("hbond", "Asp128") in found(site, drawn=False)
    check("docking session: every pose analysed",
          all(v[0] > 5 for v in res.values()) and res[1][1] and res[2][1], str(res))
    site = bs.bindingsite("poses", quiet=1, state=2)
    check("explicit state=2", len(site["residues"]) > 5)
    check("note about the receptor fallback", any("alone in its object" in n for n in site["chem"].notes))
    fresh()
    pdb("1stp")
    cmd.fragment("ala", "other")
    cmd.create("other", "other", 1, 2)
    cmd.create("other", "other", 1, 3)
    cmd.frame(3)
    site = bs.bindingsite(quiet=1)
    check("single-state complex at global state 3", ("hbond", "Asp128") in found(site))


# ---- duplicate bonds, peptide-bond N of a non-standard residue --------------------------------
@case
def test_duplicate_bonds_and_nonstandard_residues():
    pdb("6lu7")
    site = bs.bindingsite(quiet=1)
    check("6lu7: Ala(P4) N-H...O Thr190", ("hbond", "Thr190") in found(site),
          str(sorted(x for x in found(site) if x[0] == "hbond")))
    atoms, C = site["atoms"], site["chem"]
    check("no atom has a duplicated neighbour", all(len(a.nb) == len(set(a.nb)) for a in atoms))
    pje_n = [a.i for a in atoms if a.resn == "PJE" and a.name == "N"]
    check("PJE backbone N is an amide, not an ammonium",
          bool(pje_n) and not any(pje_n[0] in g["atoms"] for g in C.pos)
          and C.donors.get(pje_n[0], {}).get("hyb") == "sp2", str(C.donors.get(pje_n[0]) if pje_n else None))
    kinds = set(v["kind"] for i, v in C.electrophiles.items() if atoms[i].lig)
    check("6lu7: the ester of N3 is recognised", "ester" in kinds, str(sorted(kinds)))


# ---- the ligand selection never picks up our own helper objects --------------------------------
@case
def test_helper_objects_stay_out():
    pdb("1ycr", "1ycr")
    bs.bindingsite(quiet=1, surface="pocket")
    n_surf = cmd.count_atoms("bs.surface and chain B")
    check("the pocket surface excludes the peptide ligand", n_surf == 0, "%d ligand atoms in it" % n_surf)
    site = bs.bindingsite("chain B", quiet=1)
    check("'chain B' after a surface run", ("hbond", "Leu54") in found(site))
    check("no helper object left in the analysis", all("." not in a.obj for a in site["atoms"]))


# ---- valence-filled hydrogens ------------------------------------------------------------------
@case
def test_valence_filled_hydrogens():
    pdb("3ert")
    cmd.h_add("polymer")
    site = bs.bindingsite(quiet=1)
    check("h_add: Asp351 salt bridge kept", ("saltbridge", "Asp351") in found(site), str(sorted(found(site)))[:120])
    check("... and said so", any("valence-filled" in n for n in site["chem"].notes))


# ---- aromatic rings with one stretched bond ----------------------------------------------------
@case
def test_aromatic_rings():
    pdb("4dfr")
    site = bs.bindingsite("chain A and resn MTX", quiet=1)
    C, atoms = site["chem"], site["atoms"]
    rings = [set(atoms[i].name for i in r["atoms"]) for r in C.rings if atoms[r["atoms"][0]].lig]
    check("4dfr chain A MTX: pteridine pyrimidine ring aromatic",
          {"N1", "C2", "N3", "C4", "C4A", "C8A"} in rings, str(rings))
    check("... no amidinium invented",
          not any(g["kind"] in ("amidinium", "guanidinium") and atoms[g["atoms"][0]].lig for g in C.pos))
    fresh()
    pdb("1hsg")
    site = bs.bindingsite(quiet=1)
    C, atoms = site["chem"], site["atoms"]
    n_rings = sum(1 for r in C.rings if atoms[r["atoms"][0]].lig)
    check("1hsg MK1: indane benzo ring aromatic (3 ligand rings)", n_rings == 3, "%d" % n_rings)
    check("1hsg: N5...Arg8 not blocked by a water", ("hbond", "Arg8") in found(site),
          str(sorted(x for x in found(site) if x[0] == "hbond")))


# ---- a chain break is not an N-terminus --------------------------------------------------------
@case
def test_chain_break():
    pdb("1stp")
    cmd.remove("cplx and polymer and resi 48")
    f = found(bs.bindingsite(quiet=1))
    check("Asn49 N...biotin stays an H-bond after a chain break",
          ("hbond", "Asn49") in f and ("saltbridge", "Asn49") not in f, str(sorted(f))[:100])


# ---- surface cut in a multi-state object -------------------------------------------------------
@case
def test_surface_in_state_2():
    pdb("1stp")
    cmd.create("cplx", "cplx", 1, 2)
    cmd.frame(2)
    bs.bindingsite(quiet=1, surface="pocket")
    total = cmd.count_atoms("bs.surface")
    shown = cmd.count_atoms("bs.surface and rep surface")
    check("surface is cut in state 2", 0 < shown < total, "%d of %d atoms carry surface" % (shown, total))


# ---- one-atom ligands ---------------------------------------------------------------------------
@case
def test_one_atom_ligands():
    pdb("1stp")
    cmd.pseudoatom("xe", pos=[60.0, 60.0, 60.0], elem="Xe", name="XE", resn="XE", chain="X", hetatm=1)
    site = bs.bindingsite("xe", "cplx", quiet=1)
    check("one-atom ligand far from everything does not crash", site is not None)
    fresh()
    pdb("3hs4")
    site = bs.bindingsite("resn ZN", quiet=1)
    check("metal-only ligand", any(r["type"] == "metal" for r in site["inter"]))


# ---- other complexes loaded alongside stay out --------------------------------------------------
@case
def test_other_complexes_stay_out():
    # the same complex twice, exactly on top of each other
    pdb("1stp", "copy1")
    pdb("1stp", "copy2")
    site = bs.bindingsite("copy1 and organic", quiet=1)
    objs = set(x.obj for x in site["atoms"])
    n_clash = sum(1 for r in site["inter"] if r["type"] == "clash")
    check("two superposed complexes: only the ligand's own object",
          objs == {"copy1"} and n_clash == 0, "%s, %d clashes" % (objs, n_clash))


# ---- split groups, charge-separated groups -------------------------------------------------------
@case
def test_split_and_charge_separated_groups():
    pdb("4dfr")
    site = bs.bindingsite("chain A and resn MTX and not name O1", quiet=1)
    check("ligand selection that splits a carboxylate", site is not None)
    fresh()
    pdb("1stp")
    site = bs.bindingsite("resn BTN and name O11", quiet=1)
    check("one-atom piece of a carboxylate", site is not None)
    fresh()
    pdb("1stp")
    cmd.alter("resn BTN and name C11", "formal_charge=1")
    cmd.alter("resn BTN and name O12", "formal_charge=-1")
    cmd.alter("resn BTN and name C11", "elem='N'")
    cmd.alter("resn BTN and name O11", "elem='C'")
    site = bs.bindingsite("resn BTN", quiet=1)
    C, atoms = site["chem"], site["atoms"]
    check("N+/O- drawn charge-separated is not a charged group",
          not any(g["kind"] == "formal charge" and atoms[g["atoms"][0]].lig for g in C.pos + C.neg))


# ---- clean-up -------------------------------------------------------------------------------------
@case
def test_cleanup():
    pdb("1stp")
    before = dict((k, cmd.get(k)) for k in ("fog_start", "bg_rgb", "ray_trace_mode", "orthoscopic", "depth_cue"))
    bs.bindingsite(quiet=1)
    bs.bindingsite_clear()
    after = dict((k, cmd.get(k)) for k in before)
    check("global settings restored", before == after,
          str([(k, before[k], after[k]) for k in before if before[k] != after[k]]))
    check("object-level settings removed",
          cmd.get("cartoon_transparency", "cplx") == cmd.get("cartoon_transparency")
          and cmd.get("valence", "cplx") == cmd.get("valence"))
    check("an unknown ligand is a clean error", refused(bs.bindingsite, "resn NOPE", quiet=1))
    check("no temporary selections left",
          not [n for n in cmd.get_names("all") if n.startswith("_bs_")], str(cmd.get_names("all")))
    bs.bindingsite(quiet=1)
    check("a bad colour is refused before anything is deleted",
          refused(bs.bindingsite, quiet=1, ligand_color="no_such_colour")
          and "bs.hbond" in cmd.get_names("all") and "bs" in bs._SITES)
    cmd.reinitialize()
    for fn in (bs.bindingsite_relabel, bs.bindingsite_view, bs.bindingsite_report):
        check("%s after reinitialize: clean error, nothing created" % fn.__name__,
              refused(fn) and not cmd.get_names("all"))
    fresh()
    two_complexes("dhfr_a", "dhfr_b")
    bs.bindingsite_all(quiet=1)
    bs.bindingsite_clear()
    check("gallery cleared: scenes gone, objects enabled again",
          not cmd.get_scene_list() and set(cmd.get_names("objects", enabled_only=1)) == {"dhfr_a", "dhfr_b"},
          "%s %s" % (cmd.get_scene_list(), cmd.get_names("objects", enabled_only=1)))


# ---- arguments as typed at the PyMOL prompt -------------------------------------------------------
@case
def test_prompt_arguments():
    pdb("1ycr")
    cmd.do('bindingsite offset="A:100,B:0", ligand_label="p53 peptide", label_format="{r}{resi}", quiet=1', echo=0)
    labs = labels()
    check("quoted arguments from the prompt",
          "L154" in labs and "L54" not in labs and "p53 peptide" in labs, str(sorted(labs)))
    cmd.do("bindingsite surface=off, quiet=1", echo=0)
    check("surface=off draws none", "bs.surface" not in cmd.get_names("all"))
    cmd.do('bindingsite_png "%s", 300, 200, quiet=1' % os.path.join(OUT, "quoted name.png").replace("\\", "/"), echo=0)
    check("quoted file name", os.path.exists(os.path.join(OUT, "quoted name.png")))
    fresh()
    pdb("1stp")
    site = bs.bindingsite(receptor="polymer", quiet=1)
    check("receptor=polymer with the ligand on auto", ("hbond", "Asp128") in found(site))
    view = cmd.get_view()
    bs.bindingsite_png(os.path.join(OUT, "wide.png"), width=600, height=200, quiet=1)
    check("png leaves the camera alone", max(abs(a - b) for a, b in zip(view, cmd.get_view())) < 1e-4)
    fresh()
    pdb("4dfr")
    full = bs.bindingsite(quiet=1)
    n_full = sum(1 for r in full["inter"] if r["type"] in ("hbond", "saltbridge") and r.get("donor") is not None)
    only = bs.bindingsite(quiet=1, interactions="hbond")
    n_only = sum(1 for r in only["inter"] if r["type"] == "hbond")
    check("interactions=hbond keeps the H-bonds between charged groups", n_only == n_full, "%d vs %d" % (n_only, n_full))
    check("an unknown interaction type is refused", refused(bs.bindingsite, quiet=1, interactions="hbond,magic"))
    before = bs.PARAM["hbond_dist"]
    try:
        cmd.do("bindingsite_param hbond_dist, 3.3", echo=0)
        check("bindingsite_param from the prompt", bs.PARAM["hbond_dist"] == 3.3)
    finally:
        bs.PARAM["hbond_dist"] = before


# ---- blank chain IDs, alternate locations ----------------------------------------------------------
@case
def test_blank_chains_and_alternate_locations():
    pdb("1ycr")
    cmd.alter("chain A", "segi='PROA'")
    cmd.alter("chain B", "segi='PROB'")
    cmd.alter("all", "chain=''")
    cands = bs.find_ligands("enabled")
    check("peptide with blank chain IDs and segi", bool(cands) and cands[0]["kind"] == "peptide",
          str([(c["kind"], c["pretty"]) for c in cands]))
    site = bs.bindingsite(quiet=1)
    check("... and analysed", ("hbond", "Leu54") in found(site))
    fresh()
    pdb("1stp")
    cmd.alter("resn BTN", "alt='B'")
    cmd.create("copyA", "resn BTN")
    cmd.alter("copyA", "alt='A'")
    cmd.translate([0.3, 0, 0], "copyA", camera=0)
    cmd.create("both", "cplx or copyA")
    cmd.delete("cplx")
    cmd.delete("copyA")
    check("(setup) two alternate ligand copies",
          cmd.count_atoms("resn BTN and alt A") == 16 and cmd.count_atoms("resn BTN and alt B") == 16)
    site = bs.bindingsite("resn BTN and alt B", quiet=1)
    check("ligand selected by its alternate location",
          len(site["lig"]) == 16 and all(site["atoms"][i].alt == "B" for i in site["lig"]))


# ---- histidine at acidic pH --------------------------------------------------------------------------
@case
def test_his_positive():
    pdb("6lu7")
    site = bs.bindingsite(quiet=1, his_positive=1)
    C, atoms = site["chem"], site["atoms"]
    his_n = [a.i for a in atoms if a.resn == "HIS" and a.name in ("ND1", "NE2")]
    check("his_positive: imidazolium N are donors only", bool(his_n) and not any(i in C.acceptors for i in his_n))


# ---- pictures larger than the window -----------------------------------------------------------
@case
def test_plain_ray_and_png_keep_the_labels():
    """`ray 4000,4000` shrank the labels to specks: PyMOL's ray-tracer enlarges labels sized
    in pixels for a larger picture, but not those sized in Angstrom."""
    cmd.viewport(320, 240)
    pdb("1stp")
    bs.bindingsite(quiet=1, distances=1)
    W, H = 640, 480                                     # twice the window

    def sizes():
        return sorted((o, cmd.get("label_size", o)) for o in cmd.get_names("objects")
                      if float(cmd.get("label_size", o)) < 0)

    def path(tag):
        return os.path.join(OUT, "ray_%s.png" % tag).replace("\\", "/")

    before = sizes()
    check("the scene has labels sized in Angstrom", {"bs.labels", "bs.hbond"} <= {o for o, _ in before})
    # the reference is the plugin's own command with the camera left alone; the first
    # picture of a session is discarded, its distance labels sit a few pixels off
    bs.bindingsite_png(path("first"), W, H, dpi=150, fit=0, quiet=1)
    bs.bindingsite_png(path("reference"), W, H, dpi=150, fit=0, quiet=1)
    reference = pixels(path("reference"))
    check("reference picture is %d x %d" % (W, H), reference.shape[:2] == (H, W))

    def off(tag, make, save=True):
        """Share of the pixels in which the picture `make` produces differs from the reference."""
        make()
        if save:
            cmd.png(path(tag), dpi=150)                 # the picture `ray` left in the window
        return differing(reference, pixels(path(tag)))

    wrapped = [
        ("`ray W, H` typed at the prompt", off("prompt", lambda: cmd.do("ray %d, %d" % (W, H)))),
        ("cmd.ray(W, H)", off("api", lambda: cmd.ray(W, H))),
        ("`ray W` (height follows the window)", off("width", lambda: cmd.do("ray %d" % W))),
        ("the Draw/Ray button (`ray W, H, async=1`)",
         off("button", lambda: cmd.do("ray %d, %d, async=1" % (W, H)))),
        ("cmd.png(file, W, H, ray=1)",
         off("png", lambda: cmd.png(path("png"), W, H, dpi=150, ray=1), save=False)),
        ("`png file, W, H` without OpenGL",
         off("sized", lambda: cmd.do("png %s, %d, %d, dpi=150" % (path("sized"), W, H)), save=False)),
        ("cmd.png(file, '10.84cm', dpi=150, ray=1)",
         off("cm", lambda: cmd.png(path("cm"), "10.84cm", dpi=150, ray=1), save=False)),
    ]
    for what, share in wrapped:
        check("%s: labels as in bindingsite_png" % what, share < 0.003, "(%.2f %% of pixels differ)" % (100 * share))
    check("label sizes are put back", sizes() == before, str(sizes()))

    # PyMOL's own ray, unwrapped: if this ever matches, PyMOL has changed and the
    # correction would enlarge the labels twice
    stock = off("stock", lambda: bs._STOCK["ray"](W, H))
    check("PyMOL's own ray still shrinks them", stock > 0.01, "(%.2f %% of pixels differ)" % (100 * stock))

    # a picture of the size of the window, and a session without a binding site: nothing to do
    check("the size of the window needs no correction", bs._to_correct(0, 0) is None and bs._to_correct(320, 240) is None)
    fresh()
    cmd.viewport(320, 240)
    pdb("1stp")
    cmd.hide("everything")
    cmd.show("sticks", "resn BTN")
    cmd.label("resn BTN and name S1", "'biotin'")
    cmd.set("label_size", -1.5, "cplx")
    cmd.orient("resn BTN")
    check("labels of other objects are left to PyMOL", bs._to_correct(W, H) is None)
    cmd.ray(W, H)
    cmd.png(path("theirs"), dpi=150)
    bs._STOCK["ray"](W, H)
    cmd.png(path("theirs_stock"), dpi=150)
    share = differing(pixels(path("theirs")), pixels(path("theirs_stock")))
    check("... so their picture is PyMOL's own", share < 0.001, "(%.2f %% of pixels differ)" % (100 * share))


@case
def test_ray_and_png_are_wrapped_once():
    """Running the plugin file again must wrap PyMOL's commands, not its own wrappers."""
    check("ray and png go through the plugin", cmd.ray is bs._ray and cmd.png is bs._png
          and cmd.keyword["ray"][0] is bs._ray and cmd.keyword["png"][0] is bs._png)
    check("their help is PyMOL's", "ray" in (cmd.ray.__doc__ or "") and "PNG" in (cmd.png.__doc__ or ""))
    stock = dict(bs._STOCK)
    check("PyMOL's own commands are kept", stock["ray"] is not bs._ray and stock["png"] is not bs._png)
    bs._wrap_ray_and_png()
    check("wrapping a second time finds them again",
          bs._STOCK["ray"] is stock["ray"] and bs._STOCK["png"] is stock["png"] and cmd.ray is bs._ray)


# ---- the dialog -----------------------------------------------------------------------------------------
@case
def test_dialog():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from pymol.Qt import QtWidgets
    except Exception as err:
        skip("no Qt bindings: %s" % err)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    pdb("3hs4")
    dlg = bs.bindingsite_gui()
    w = dlg._bs["widgets"]
    check("dialog passes the exact selection", w["ligand"].count() > 2 and "index" in str(w["ligand"].itemData(1)))
    w["ligand"].setCurrentIndex(2)
    chosen = w["ligand"].currentText()
    dlg._bs["find"]()
    check("'Find' keeps the chosen ligand", w["ligand"].currentText() == chosen, w["ligand"].currentText())
    dlg._bs["show"]()
    check("'Show' analyses the chosen ligand", "bs" in bs._SITES and "bs.labels" in cmd.get_names("all"))
    dlg.hide()


if __name__ == "__main__":
    sys.exit(main(globals()))
