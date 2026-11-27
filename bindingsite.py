# -*- coding: utf-8 -*-
"""
bindingsite: binding-site figures in one PyMOL command
======================================================

Turns a protein-ligand complex into a finished binding-site view: the ligand,
the residues that matter, colour-coded non-covalent interactions, readable
residue labels that do not overlap, and a camera that looks into the pocket.

    PyMOL> bindingsite                    # find the ligand, do everything
    PyMOL> bindingsite resn STI           # ... for a ligand you name
    PyMOL> bindingsite_report             # table of what was found
    PyMOL> bindingsite_png site.png       # ray-traced figure, framed for the picture

It needs nothing but PyMOL and numpy.  The chemistry (donors, acceptors,
charges, aromatic rings, electrophiles) is perceived from geometry, so it also
works on heavy-atom-only files without bond orders, such as crystal structures,
Boltz/AlphaFold-style co-folds and docking poses.

Install:  Plugin > Plugin Manager > Install New Plugin > Choose file...
          (or add `run /path/to/bindingsite.py` to your pymolrc)

Type `help bindingsite` in PyMOL for the options.

https://github.com/MicheleBonus/pymol-bindingsite
Copyright (c) 2026 Michele Bonus.  Released under the MIT License.
"""
# SPDX-License-Identifier: MIT

from __future__ import print_function

import contextlib
import math
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np
from pymol import cmd, CmdException

__version__ = "1.0.1"

# =============================================================================
#  Tunable detection thresholds.  Change at runtime, e.g. from the PyMOL prompt:
#      bindingsite_param hbond_dist, 3.3
# =============================================================================
PARAM = OrderedDict([
    ("hbond_dist", 3.5),        # max donor...acceptor distance, heavy atoms (A)
    ("hbond_min", 2.3),         # polar contacts shorter than this are clashes (A)
    ("hbond_angle", 120.0),     # min D-H...A angle; H is placed geometrically if absent
    ("hbond_acc_angle", 90.0),  # min X-A...D angle at the acceptor
    ("salt_dist", 4.0),         # max distance between oppositely charged atoms (A)
    ("pi_dist", 5.5),           # max ring-centroid distance for pi-stacking (A)
    ("pi_angle", 30.0),         # max deviation from parallel / perpendicular (deg)
    ("pi_offset", 2.0),         # max lateral offset of the ring centroids (A)
    ("cation_pi_dist", 5.0),    # max cation...ring-centroid distance (A)
    ("halogen_slack", 0.3),     # X...A distance allowed beyond the vdW sum (A)
    ("halogen_don_angle", 140.0),   # min C-X...A angle
    ("halogen_acc_min", 90.0),  # allowed range of the X...A-Y angle
    ("halogen_acc_max", 150.0),
    ("hydrophobic_dist", 4.0),  # max C...C distance for a hydrophobic contact (A)
    ("water_dist", 3.2),        # max length of each leg of a water bridge (A)
    ("water_angle_min", 71.0),  # ligand...water...protein angle window (deg)
    ("water_angle_max", 140.0),
    ("metal_dist", 2.8),        # max metal...N/O/S distance (A)
    ("covalent_slack", 0.3),    # bonded if closer than covalent radii + slack (A)
    ("clash_overlap", 0.6),     # vdW overlap that counts as a clash, non-polar pairs (A)
    ("nac_dist", 3.5),          # nucleophile...electrophile near-attack distance (A)
    ("nac_angle_min", 80.0),    # Buergi-Dunitz window for the attack angle (deg)
    ("nac_angle_max", 130.0),
])

# -----------------------------------------------------------------------------
#  Interaction types: label, colour (Okabe-Ito based, colour-blind safe) and the
#  dash geometry (gap, length, radius in A).
# -----------------------------------------------------------------------------
ITYPES = OrderedDict([
    ("hbond",       dict(label="Hydrogen bond",       rgb=(0, 114, 178),   dash=(0.22, 0.26, 0.070))),
    ("saltbridge",  dict(label="Salt bridge",         rgb=(213, 94, 0),    dash=(0.22, 0.26, 0.080))),
    ("pistack",     dict(label="pi-stacking",         rgb=(0, 158, 115),   dash=(0.28, 0.34, 0.070))),
    ("pication",    dict(label="Cation-pi",           rgb=(230, 159, 0),   dash=(0.28, 0.34, 0.070))),
    ("halogen",     dict(label="Halogen bond",        rgb=(86, 180, 233),  dash=(0.22, 0.26, 0.070))),
    ("metal",       dict(label="Metal coordination",  rgb=(124, 77, 190),  dash=(0.12, 0.32, 0.070))),
    ("waterbridge", dict(label="Water bridge",        rgb=(110, 160, 215), dash=(0.20, 0.18, 0.055))),
    ("hydrophobic", dict(label="Hydrophobic contact", rgb=(150, 150, 150), dash=(0.30, 0.14, 0.040))),
    ("covalent",    dict(label="Covalent bond",       rgb=(25, 25, 25),    dash=(0.00, 0.40, 0.090))),
    ("nearattack",  dict(label="Near-attack contact", rgb=(204, 121, 167), dash=(0.16, 0.32, 0.075))),
    ("clash",       dict(label="Steric clash",        rgb=(225, 25, 35),   dash=(0.13, 0.13, 0.090))),
])
SPECIFIC = ("hbond", "saltbridge", "pistack", "pication", "halogen", "metal",
            "waterbridge", "covalent", "nearattack", "clash")

# -----------------------------------------------------------------------------
#  Visual styles.  Colours are RGB 0-1.  `settings` are global PyMOL settings
#  (their previous values are remembered and restored by bindingsite_clear).
# -----------------------------------------------------------------------------
_COMMON = OrderedDict([
    ("orthoscopic", 1), ("depth_cue", 1), ("ray_trace_fog", 1), ("ray_shadow", 0),
    ("antialias", 2), ("float_labels", 1), ("dash_round_ends", 1), ("dash_as_cylinders", 1),
    ("two_sided_lighting", 1), ("transparency_mode", 1), ("ray_transparency_oblique", 0.0),
    ("specular", 0.18), ("shininess", 28), ("spec_reflect", 0.4), ("light_count", 2),
    ("ambient", 0.42), ("direct", 0.55), ("reflect", 0.45), ("ray_opaque_background", 1),
    ("ray_label_specular", 0),
    ("cartoon_fancy_helices", 0), ("stick_ball", 0), ("line_width", 2.0),
])
STYLES = OrderedDict()
STYLES["publication"] = dict(
    doc="white background, thin ray-traced outlines",
    bg=(1.0, 1.0, 1.0), ligand=(0.98, 0.62, 0.11), residue=(0.74, 0.77, 0.81),
    het=(0.62, 0.55, 0.82), cartoon=(0.87, 0.89, 0.92), surface=(0.90, 0.92, 0.95),
    label=(0.08, 0.08, 0.08), label_dim=(0.36, 0.36, 0.38), cartoon_transparency=0.80,
    settings=OrderedDict(list(_COMMON.items()) + [
        ("fog", 0.80), ("ray_trace_mode", 1), ("ray_trace_gain", 0.12),
        ("ray_trace_color", "black"), ("ray_trace_trans_cutoff", 0.5)]))
STYLES["soft"] = dict(
    doc="white background, no outlines, slightly glossy",
    bg=(1.0, 1.0, 1.0), ligand=(0.10, 0.60, 0.44), residue=(0.60, 0.65, 0.72),
    het=(0.58, 0.48, 0.80), cartoon=(0.80, 0.84, 0.90), surface=(0.86, 0.89, 0.94),
    label=(0.08, 0.08, 0.08), label_dim=(0.34, 0.34, 0.36), cartoon_transparency=0.72,
    settings=OrderedDict(list(_COMMON.items()) + [
        ("fog", 0.65), ("ray_trace_mode", 0), ("specular", 0.30)]))
STYLES["flat"] = dict(
    doc="illustrative: flat colours with black outlines (posters, slides)",
    bg=(1.0, 1.0, 1.0), ligand=(0.98, 0.62, 0.11), residue=(0.80, 0.82, 0.86),
    het=(0.62, 0.55, 0.82), cartoon=(0.90, 0.92, 0.95), surface=(0.93, 0.95, 0.97),
    label=(0.05, 0.05, 0.05), label_dim=(0.33, 0.33, 0.35), cartoon_transparency=0.75,
    settings=OrderedDict(list(_COMMON.items()) + [
        ("fog", 0.55), ("ray_trace_mode", 3), ("ray_trace_gain", 0.05),
        ("ray_trace_color", "black"), ("specular", 0.0), ("ambient", 0.6), ("direct", 0.4)]))
STYLES["dark"] = dict(
    doc="black background for screens and talks",
    bg=(0.0, 0.0, 0.0), ligand=(1.0, 0.78, 0.20), residue=(0.66, 0.70, 0.76),
    het=(0.72, 0.62, 0.95), cartoon=(0.34, 0.37, 0.42), surface=(0.45, 0.48, 0.54),
    label=(0.97, 0.97, 0.97), label_dim=(0.70, 0.70, 0.72), cartoon_transparency=0.55,
    settings=OrderedDict(list(_COMMON.items()) + [
        ("fog", 0.85), ("ray_trace_mode", 0), ("specular", 0.30)]))

# -----------------------------------------------------------------------------
#  Chemistry tables
# -----------------------------------------------------------------------------
VDW = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80,
       "Cl": 1.75, "Br": 1.85, "I": 1.98, "Se": 1.90, "B": 1.92, "Si": 2.10}
COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05,
       "Cl": 1.02, "Br": 1.20, "I": 1.39, "Se": 1.20, "B": 0.84, "Si": 1.11}
METALS = set("Li Na K Rb Cs Be Mg Ca Sr Ba Al Ga In Tl Sc Ti V Cr Mn Fe Co Ni Cu Zn "
             "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg Pb Bi La Ce Pr Nd "
             "Sm Eu Gd Tb Dy Ho Er Tm Yb Lu U".split())
WATERS = set("HOH WAT H2O DOD D2O TIP TIP3 TIP4 TP3 T3P T4P SOL SPC OH2".split())
# crystallisation additives, buffers, cryoprotectants: not offered as "the ligand"
ADDITIVES = set("GOL EDO PEG PGE PG4 1PE P6G 2PE P33 PE4 PE8 MPD DMS ACT ACY FMT EPE MES "
                "TRS BME DTT IPA EOH MOH PO4 SO4 CIT TLA MLI IMD BU3 PGO BTB B3P NHE CAC "
                "MRD TAM HEZ DIO PG0 7PE 12P 15P CO3 NO3 SCN AZI BCT OXL MLA SIN FLC "
                "ACE NH2 NME GLY".split())
SUGARS = set("NAG NDG MAN BMA FUC FUL GAL GLA GLC BGC SIA XYS XYP A2G NGA".split())

AA3 = ("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL").split()
AA1 = dict(zip(AA3, "ARNDCQEGHILKMFPSTWYV"))
AA_ALIAS = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
            "HISD": "HIS", "HISE": "HIS", "HISH": "HIS", "CYX": "CYS", "CYM": "CYS",
            "CYS2": "CYS", "ASH": "ASP", "ASPH": "ASP", "GLH": "GLU", "GLUH": "GLU",
            "LYN": "LYS", "MSE": "MET"}
HIS_TAUT = {"HID": "D", "HSD": "D", "HISD": "D", "HIE": "E", "HSE": "E", "HISE": "E",
            "HIP": "P", "HSP": "P", "HISH": "P"}
# side-chain donors: atom -> (number of H, hybridisation used to place them)
AA_DON = {
    "ARG": {"NE": (1, "sp2"), "NH1": (2, "sp2"), "NH2": (2, "sp2")},
    "ASN": {"ND2": (2, "sp2")}, "GLN": {"NE2": (2, "sp2")},
    "HIS": {"ND1": (1, "sp2"), "NE2": (1, "sp2")},
    "LYS": {"NZ": (3, "sp3")}, "SER": {"OG": (1, "sp3")}, "THR": {"OG1": (1, "sp3")},
    "TYR": {"OH": (1, "sp3")}, "TRP": {"NE1": (1, "sp2")},
}
AA_ACC = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "ASN": ("OD1",), "GLN": ("OE1",),
          "HIS": ("ND1", "NE2"), "SER": ("OG",), "THR": ("OG1",), "TYR": ("OH",)}
AA_RINGS = {"PHE": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
            "TYR": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
            "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
            "TRP": (("CG", "CD1", "NE1", "CE2", "CD2"), ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"))}
AA_NUC = {"SER": "OG", "THR": "OG1", "CYS": "SG", "LYS": "NZ", "TYR": "OH"}
BACKBONE = ("N", "C", "O", "OXT")
WARHEADS = set(("carbamate", "ester", "thioester", "carbonate", "beta-lactam", "aldehyde",
                "nitrile", "boronic acid", "Michael acceptor", "epoxide", "aziridine",
                "alpha-halo carbonyl", "sulfonyl fluoride", "phosph(on)ate ester"))

# aromatic-ring perception (geometry only)
# Max out-of-plane deviation of a ring atom (A).  Aromatic rings measure 0.00-0.02
# in models and up to ~0.07 in old crystal structures; a flattened piperidine in
# a sloppy docking pose still shows 0.13, a thiazolidine 0.18.
RING_PLANAR_TOL = 0.10
# A planar ring is rejected only if a bond is clearly a single bond to an sp3
# atom (CH2 in a hydantoin, succinimide, ...); hetero-hetero bonds are left alone
# because refinement dictionaries often stretch them.
RING_MAX_BOND = {("C", "C"): 1.475, ("C", "N"): 1.47, ("C", "O"): 1.45, ("C", "S"): 1.79,
                 ("N", "N"): 1.48, ("N", "O"): 1.50, ("N", "S"): 1.76, ("C", "Se"): 1.93}

# label metrics, as a fraction of label_size (DejaVu Sans Bold, measured)
CHAR_W = 0.64
CHAR_H = 0.78

_SITES = OrderedDict()      # name -> site record of the last analysis
_SAVED = {}                 # global settings as they were before the first call
_GALLERY = dict(scenes=[], objs=[])     # what bindingsite_all left behind
# objects this plugin creates are called <site>.<suffix>
_SUFFIXES = set(ITYPES) | set(("labels", "labels_lining", "leaders", "leader_ends", "centers",
                               "legend", "surface"))


# =============================================================================
#  small helpers
# =============================================================================
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def _angle(a, b, c):
    """Angle a-b-c at b, in degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2)))))


def _perp(v):
    """Some unit vector perpendicular to v."""
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(v, a))


def _truthy(x):
    if isinstance(x, str):
        return x.strip().lower() not in ("0", "", "no", "off", "false", "none")
    return bool(x)


def _say(quiet, *args):
    if not quiet:
        print(" bindingsite:", *args)


def _set(name, value, selection="", quiet=True):
    """cmd.set that tolerates settings missing from older/newer PyMOL builds."""
    try:
        if selection:
            cmd.set(name, value, selection, quiet=1)
        else:
            cmd.set(name, value, quiet=1)
        return True
    except Exception:
        return False


def _color(name, rgb):
    rgb = [float(x) for x in rgb]
    if max(rgb) > 1.0:
        rgb = [x / 255.0 for x in rgb]
    cmd.set_color(name, rgb)
    return name


def _ranges(ids):
    ids = sorted(set(ids))
    out, start, prev = [], None, None
    for i in ids:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append("%d-%d" % (start, prev) if prev > start else "%d" % start)
            start = prev = i
    if start is not None:
        out.append("%d-%d" % (start, prev) if prev > start else "%d" % start)
    return out


def _not_ours(expr):
    """`expr` minus the helper objects of this plugin (labels, surface copy, ...)."""
    own = [n for n in cmd.get_names("objects")
           if (("." in n and n.rsplit(".", 1)[1] in _SUFFIXES)
               or (n.startswith("_") and n.endswith("_legend_end")))
           and cmd.get_type(n) == "object:molecule"]
    expr = "((%s) and not elem PS)" % expr
    for k in range(0, len(own), 40):
        expr = "(%s and not (%s))" % (expr, " or ".join("%" + n for n in own[k:k + 40]))
    return expr


def _unquote(text):
    """PyMOL hands quoted command-line arguments over with the quotes still on."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text


def _coords(selection, state):
    """[(object, indices, xyz)] for `selection`, each object read in ITS OWN state:
    a single-state receptor next to a multi-state set of poses has no state 2."""
    out = []
    for obj in cmd.get_object_list("(%s)" % selection) or []:
        rows = []
        cmd.iterate_state(_object_state(obj, state), "(%s) and %%%s" % (selection, obj),
                          "rows.append((index, x, y, z))", space={"rows": rows})
        if rows:
            arr = np.array(rows, dtype=float)
            out.append((obj, arr[:, 0].astype(int), arr[:, 1:]))
    return out


def _near(rows, points, cutoff):
    """{object: (indices, xyz)} of the atoms in `rows` within `cutoff` of `points`."""
    out = OrderedDict()
    if not len(points):
        return out
    lo, hi = points.min(0) - cutoff, points.max(0) + cutoff
    for obj, idx, X in rows:
        box = np.all((X >= lo) & (X <= hi), axis=1)
        if not box.any():
            continue
        Xb, ib = X[box], idx[box]
        hit = np.zeros(len(ib), dtype=bool)
        for k in range(0, len(points), 48):
            d2 = ((Xb[:, None, :] - points[None, k:k + 48, :]) ** 2).sum(-1)
            hit |= (d2 <= cutoff * cutoff).any(1)
        if hit.any():
            out[obj] = (ib[hit], Xb[hit])
    return out


def _osel(found):
    """Selection string for the {object: (indices, xyz)} mapping of _near()."""
    parts = []
    for obj, (idx, _) in found.items():
        r = _ranges(idx.tolist())
        for k in range(0, len(r), 200):
            parts.append("(%%%s and index %s)" % (obj, "+".join(r[k:k + 200])))
    return " or ".join(parts) if parts else "none"


def _isel(atoms, idxs):
    """PyMOL selection string for a list of local atom indices."""
    by = defaultdict(list)
    for i in idxs:
        by[atoms[i].obj].append(atoms[i].index)
    parts = []
    for obj, ids in by.items():
        r = _ranges(ids)
        for k in range(0, len(r), 200):
            parts.append("(%%%s and index %s)" % (obj, "+".join(r[k:k + 200])))
    return " or ".join(parts) if parts else "none"


# =============================================================================
#  atoms
# =============================================================================
class _Atom(object):
    __slots__ = ("i", "obj", "index", "name", "resn", "resi", "resv", "chain", "segi", "elem",
                 "xyz", "fc", "het", "alt", "nb", "hnb", "lig", "rkey")

    def __repr__(self):
        return "<%s/%s`%s/%s>" % (self.chain, self.resn, self.resi, self.name)


def _norm_elem(sym, name):
    e = (sym or "").strip()
    if not e:
        e = re.sub("[^A-Za-z]", "", name)[:1]
    e = e.capitalize()
    return "H" if e == "D" else e


def _object_state(obj, state):
    n = cmd.count_states("%" + obj)
    st = state if state > 0 else cmd.get_state()
    if n < 1:
        return 1
    return min(st, n)


def _bond_length_ok(e1, e2, d):
    r = COV.get(e1, 0.8) + COV.get(e2, 0.8)
    return r - 0.40 <= d <= r + PARAM["covalent_slack"]


def _extract(selection, state, lig_ids):
    """Read atoms + bonds of `selection` into plain Python/numpy structures."""
    atoms, bo, spurious = [], {}, []
    models, lig_alt = [], None
    for obj in cmd.get_object_list("(%s)" % selection) or []:
        model = cmd.get_model("(%s) and %%%s" % (selection, obj), _object_state(obj, state))
        models.append((obj, model))
        for at in model.atom:
            if at.alt and (obj, int(at.index)) in lig_ids and (lig_alt is None or at.alt < lig_alt):
                lig_alt = at.alt
    for obj, model in models:
        # one alternate location per residue: the ligand's own where there is a choice
        alts = {}
        for at in model.atom:
            if at.alt:
                alts.setdefault((at.segi, at.chain, at.resi), set()).add(at.alt)
        keep = dict((key, lig_alt if lig_alt in found else min(found)) for key, found in alts.items())
        local = {}
        for k, at in enumerate(model.atom):
            if at.alt and keep.get((at.segi, at.chain, at.resi)) != at.alt:
                continue
            a = _Atom()
            a.i = len(atoms)
            a.obj, a.index, a.name = obj, int(at.index), at.name
            a.resn, a.resi, a.chain, a.segi = at.resn, at.resi, at.chain, at.segi
            try:
                a.resv = int(at.resi_number)
            except Exception:
                m = re.match(r"-?\d+", at.resi or "")
                a.resv = int(m.group()) if m else 0
            a.elem = _norm_elem(at.symbol, at.name)
            a.xyz = np.array(at.coord, dtype=float)
            a.fc = int(getattr(at, "formal_charge", 0) or 0)
            a.het = bool(getattr(at, "hetatm", 0))
            a.alt = at.alt
            a.nb, a.hnb = [], []
            a.lig = (obj, a.index) in lig_ids
            a.rkey = (obj, a.segi, a.chain, a.resi, a.resn)
            local[k] = a.i
            atoms.append(a)
        seen = set()
        for b in model.bond:
            i, j = local.get(b.index[0]), local.get(b.index[1])
            if i is None or j is None or i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:             # mmCIF struct_conn links arrive twice
                continue
            seen.add(key)
            ai, aj = atoms[i], atoms[j]
            if ai.elem in METALS or aj.elem in METALS:
                continue                # coordination is handled as an interaction
            if ai.lig != aj.lig and ai.elem != "H" and aj.elem != "H":
                # PyMOL bonds by distance when a file has no CONECT/LINK for the pair;
                # a ligand-protein "bond" longer than a real one is a bad contact.
                d = float(np.linalg.norm(ai.xyz - aj.xyz))
                if not _bond_length_ok(ai.elem, aj.elem, d):
                    spurious.append((i, j, d))
                    continue
            if aj.elem == "H":
                ai.hnb.append(j)
            else:
                ai.nb.append(j)
            if ai.elem == "H":
                aj.hnb.append(i)
            else:
                aj.nb.append(i)
            bo[key] = int(b.order)
    return atoms, bo, spurious


# =============================================================================
#  chemistry perception
# =============================================================================
class _Chem(object):
    """What the atoms can do: donors, acceptors, charges, rings, ..."""

    def __init__(self, atoms, bo):
        self.atoms, self.bo = atoms, bo
        self.donors = {}        # idx -> dict(nH, hyb[, amb])
        self.acceptors = {}     # idx -> dict([amb])
        self.pos, self.neg = [], []     # charged groups: dict(atoms, center, kind, res)
        self.rings = []         # dict(atoms, center, normal, res, sys)
        self.aromatic = set()
        self.hydrophobic = set()
        self.xdonors = {}       # halogen idx -> carbon idx
        self.metals, self.waters = set(), set()
        self.electrophiles = {}     # idx -> dict(kind, ref)
        self.nucleophiles = {}      # idx -> description
        self.notes = []


def _planar3(atoms, i, tol=0.25, with_h=False):
    nb = list(atoms[i].nb) + (list(atoms[i].hnb) if with_h else [])
    if len(nb) != 3:
        return False
    p0, p1, p2 = (atoms[j].xyz for j in nb)
    n = np.cross(p1 - p0, p2 - p0)
    ln = float(np.linalg.norm(n))
    if ln < 1e-6:
        return True
    return abs(float(np.dot(atoms[i].xyz - p0, n / ln))) <= tol


def _ring_geom(atoms, ring):
    xyz = np.array([atoms[i].xyz for i in ring])
    c = xyz.mean(0)
    vt = np.linalg.svd(xyz - c)[2]
    return c, vt[2], float(np.abs((xyz - c) @ vt[2]).max())


def _shortest_path(atoms, u, v, members, maxsize):
    prev = {u: None}
    frontier, depth = [u], 1
    while frontier and depth <= maxsize - 1:
        nxt = []
        for x in frontier:
            for y in atoms[x].nb:
                if y not in members or y in prev or (x == u and y == v):
                    continue
                prev[y] = x
                if y == v:
                    path = [v]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path
                nxt.append(y)
        frontier, depth = nxt, depth + 1
    return None


def _find_rings(atoms, members, maxsize=7):
    found = {}
    for u in members:
        for v in atoms[u].nb:
            if v <= u or v not in members:
                continue
            path = _shortest_path(atoms, u, v, members, maxsize)
            if path is not None:
                found.setdefault(frozenset(path), tuple(path))
    return list(found.values())


def _is_aromatic_ring(atoms, ring):
    n = len(ring)
    if n not in (5, 6):
        return False
    hetero = chalc = 0
    for i in ring:
        a = atoms[i]
        if a.elem not in ("C", "N", "O", "S", "Se"):
            return False
        if len(a.nb) + len(a.hnb) > 3:
            return False
        if a.elem in ("O", "S", "Se"):
            if len(a.nb) != 2:
                return False
            chalc += 1
        if a.elem != "C":
            hetero += 1
    if chalc > 1 or (n == 5 and hetero == 0):
        return False
    if _ring_geom(atoms, ring)[2] > RING_PLANAR_TOL:
        return False
    def maybe_sp3(i):
        a = atoms[i]
        if len(a.nb) >= 3:
            return not _planar3(atoms, i, 0.25)
        return not a.hnb                # two ring neighbours and nothing else we can see

    for k in range(n):
        i, j = ring[k], ring[(k + 1) % n]
        a, b = atoms[i], atoms[j]
        key = tuple(sorted((a.elem, b.elem)))
        if float(np.linalg.norm(a.xyz - b.xyz)) > RING_MAX_BOND.get(key, 1.60) \
                and (maybe_sp3(i) or maybe_sp3(j)):
            return False
    return True


def _group(atoms, idxs, kind, center=None):
    idxs = list(idxs)
    if center is None:
        center = np.mean([atoms[i].xyz for i in idxs], axis=0)
    return dict(atoms=idxs, center=np.array(center, dtype=float), kind=kind, res=atoms[idxs[0]].rkey)


def _perceive_aa(C, idxs, resn, his_positive, chain_start=None):
    """Template chemistry for the 20 standard amino acids (and common aliases)."""
    atoms = C.atoms
    canon = AA_ALIAS.get(resn, resn)
    names = {}
    for i in idxs:
        names.setdefault(atoms[i].name, i)
    has_h = any(atoms[i].hnb for i in idxs)

    def n_h(i):
        return len(atoms[i].hnb)

    # ---- backbone
    n = names.get("N")
    if n is not None:
        deg = len(atoms[n].nb)
        an = atoms[n]
        # a missing C-N bond is a chain break unless this residue starts the chain
        first = chain_start is None or \
            chain_start.get((an.obj, an.segi, an.chain), an.resv) == an.resv
        if has_h:
            if n_h(n):
                C.donors[n] = dict(nH=n_h(n), hyb="expl")
            if n_h(n) >= 3 or (canon == "PRO" and n_h(n) >= 2):
                C.pos.append(_group(atoms, [n], "N-terminus"))
        elif canon == "PRO":
            if deg <= 2:
                C.donors[n] = dict(nH=2 if first else 1, hyb="sp3")
                if first:
                    C.pos.append(_group(atoms, [n], "N-terminus"))
        elif deg >= 2 or not first:
            C.donors[n] = dict(nH=1, hyb="sp2" if deg >= 2 else "sp2free")
        else:
            C.donors[n] = dict(nH=3, hyb="sp3")
            C.pos.append(_group(atoms, [n], "N-terminus"))
    o, oxt = names.get("O"), names.get("OXT")
    if o is not None:
        C.acceptors[o] = {}
    if oxt is not None:
        C.acceptors[oxt] = {}
        if o is not None and not (has_h and (n_h(o) or n_h(oxt))):
            C.neg.append(_group(atoms, [o, oxt], "C-terminus"))

    # ---- side chain
    taut = HIS_TAUT.get(resn)
    for nm, (nh, hyb) in AA_DON.get(canon, {}).items():
        i = names.get(nm)
        if i is None:
            continue
        if has_h:
            if n_h(i):
                C.donors[i] = dict(nH=n_h(i), hyb="expl")
        elif canon == "HIS":
            if taut == "P" or (taut == "D" and nm == "ND1") or (taut == "E" and nm == "NE2"):
                C.donors[i] = dict(nH=1, hyb="sp2")
            elif taut is None:
                C.donors[i] = dict(nH=1, hyb="sp2", amb=not his_positive)
        elif resn == "LYN" and nm == "NZ":
            C.donors[i] = dict(nH=2, hyb="sp3")
        else:
            C.donors[i] = dict(nH=nh, hyb=hyb)
    for nm in AA_ACC.get(canon, ()):
        i = names.get(nm)
        if i is None:
            continue
        if canon == "HIS":
            if has_h:
                if n_h(i):
                    continue
            elif taut == "P" or (taut == "D" and nm == "ND1") or (taut == "E" and nm == "NE2"):
                continue
            elif his_positive and taut is None:
                continue                # taken as doubly protonated: no lone pair left
            C.acceptors[i] = dict(amb=(taut is None and not has_h))
        else:
            C.acceptors[i] = {}
    if resn in ("ASH", "ASPH", "GLH", "GLUH") and not has_h:
        i = names.get("OD2" if canon == "ASP" else "OE2")
        if i is not None:
            C.donors[i] = dict(nH=1, hyb="sp3")

    # ---- charges
    if canon == "ARG":
        g = [names[x] for x in ("NE", "NH1", "NH2") if x in names]
        if g:
            C.pos.append(_group(atoms, g, "guanidinium",
                                atoms[names["CZ"]].xyz if "CZ" in names else None))
    elif canon == "LYS" and "NZ" in names:
        nz = names["NZ"]
        if resn != "LYN" and not (has_h and n_h(nz) < 3):
            C.pos.append(_group(atoms, [nz], "ammonium"))
    elif canon == "HIS":
        g = [names[x] for x in ("ND1", "NE2") if x in names]
        charged = taut == "P" or (has_h and len(g) == 2 and all(n_h(i) for i in g))
        if g and (charged or (his_positive and taut is None and not has_h)):
            ring = [names[x] for x in AA_RINGS["HIS"][0] if x in names]
            C.pos.append(_group(atoms, g, "imidazolium",
                                np.mean([atoms[i].xyz for i in ring], axis=0)))
    elif canon in ("ASP", "GLU") and resn not in ("ASH", "ASPH", "GLH", "GLUH"):
        g = [names[x] for x in AA_ACC[canon] if x in names]
        if g and not (has_h and any(n_h(i) for i in g)):
            C.neg.append(_group(atoms, g, "carboxylate"))

    # ---- aromatic rings
    for ring_names in AA_RINGS.get(canon, ()):
        if all(x in names for x in ring_names):
            ring = tuple(names[x] for x in ring_names)
            c, nrm, _ = _ring_geom(atoms, ring)
            C.rings.append(dict(atoms=ring, center=c, normal=nrm, res=atoms[ring[0]].rkey))
            C.aromatic.update(ring)

    # ---- nucleophiles (targets of covalent warheads)
    nm = AA_NUC.get(canon)
    if nm and nm in names:
        i = names[nm]
        if not (canon == "CYS" and any(atoms[j].elem == "S" for j in atoms[i].nb)):
            C.nucleophiles[i] = "%s %s" % (canon.capitalize(), nm)


def _perceive_generic(C, idxs, has_h):
    """Geometry-based chemistry for ligands, cofactors and non-standard residues."""
    atoms = C.atoms
    members = set(idxs)
    if not members:
        return

    def hd(i):
        return len(atoms[i].nb)

    def dist(i, j):
        return float(np.linalg.norm(atoms[i].xyz - atoms[j].xyz))

    def term(i, elems):
        return [j for j in atoms[i].nb if atoms[j].elem in elems and len(atoms[j].nb) == 1]

    def order(i, j):
        return C.bo.get((min(i, j), max(i, j)), 1)

    # ---- rings
    rings = _find_rings(atoms, members)
    ring_of = defaultdict(list)
    arom_rings = []
    for ring in rings:
        if _is_aromatic_ring(atoms, ring):
            for i in ring:
                ring_of[i].append(len(arom_rings))
            arom_rings.append(ring)
            c, nrm, _ = _ring_geom(atoms, ring)
            C.rings.append(dict(atoms=tuple(ring), center=c, normal=nrm, res=atoms[ring[0]].rkey))
    arom = set(ring_of)
    C.aromatic |= arom

    # ---- functional-group carbons
    carbonyl, carboxylate = {}, {}
    for i in idxs:
        a = atoms[i]
        if a.elem != "C" or hd(i) + len(a.hnb) > 3:
            continue
        t_o = term(i, ("O",))
        if not t_o:
            continue
        if len(t_o) >= 2:
            if any(atoms[o].hnb for o in t_o):      # carboxylic acid drawn with its H
                free = [o for o in t_o if not atoms[o].hnb]
                if free:
                    carbonyl[i] = free[0]
            else:
                carboxylate[i] = t_o
            continue
        o = t_o[0]
        if atoms[o].hnb:
            continue
        d = dist(i, o)
        if order(i, o) == 2:
            ok = True
        elif i in arom:
            ok = d < 1.30
        elif hd(i) == 3:
            ok = _planar3(atoms, i) and d < 1.36
        else:
            ok = d < 1.30
        if ok:
            carbonyl[i] = o
    def acyl_c(j):
        """A carbonyl carbon, including the C=O of a neighbouring standard residue."""
        if j in carbonyl:
            return True
        return (j not in members and atoms[j].elem == "C"
                and any(atoms[k].elem == "O" and len(atoms[k].nb) == 1 for k in atoms[j].nb))

    oxo_s = {i: term(i, ("O",)) for i in idxs if atoms[i].elem == "S" and term(i, ("O",))}
    oxo_p = {i: term(i, ("O",)) for i in idxs if atoms[i].elem == "P" and term(i, ("O",))}

    amidine = {}
    for i in idxs:
        a = atoms[i]
        if a.elem != "C" or i in arom or i in carbonyl or i in carboxylate:
            continue
        if hd(i) + len(a.hnb) > 3 or term(i, ("O", "S")):
            continue
        ns = [j for j in a.nb if atoms[j].elem == "N"]
        if len(ns) < 2 or (hd(i) == 3 and not _planar3(atoms, i)):
            continue
        if np.mean([dist(i, j) for j in ns]) > 1.40:
            continue
        if any(j in arom or any(k != i and (k in carbonyl or k in oxo_s or k in oxo_p)
                                for k in atoms[j].nb) for j in ns):
            continue
        amidine[i] = ns

    # ---- atom typing
    pos_single = []
    tetrazoles = {}
    aminopyridinium = []
    for i in idxs:
        a = atoms[i]
        e, deg, nh = a.elem, hd(i), len(a.hnb)
        explicit = has_h.get(a.rkey, False)
        if e == "O":
            if i in arom:
                continue
            if nh:
                C.donors[i] = dict(nH=nh, hyb="expl")
            elif not explicit:
                if deg == 0:
                    C.donors[i] = dict(nH=2, hyb="free")
                elif deg == 1:
                    x = a.nb[0]
                    ex = atoms[x].elem
                    if ex == "C":
                        if x not in carboxylate and carbonyl.get(x) != i:
                            C.donors[i] = dict(nH=1, hyb="sp3")
                    elif ex == "N":
                        if len(term(x, ("O",))) < 2 and dist(i, x) > 1.33:
                            C.donors[i] = dict(nH=1, hyb="sp3")
                    elif ex in ("B", "Si"):
                        C.donors[i] = dict(nH=1, hyb="sp3")
            C.acceptors[i] = {}
        elif e == "N":
            acyl = any(acyl_c(j) for j in a.nb)
            sulf = any(j in oxo_s or j in oxo_p for j in a.nb)
            aryl = any(j in arom for j in a.nb)
            amid = any(j in amidine for j in a.nb)
            if explicit:
                tot = deg + nh
                if nh:
                    C.donors[i] = dict(nH=nh, hyb="expl")
                if a.fc > 0 or tot >= 4:
                    pos_single.append(i)
                elif tot <= 2:
                    C.acceptors[i] = {}
                elif not (acyl or sulf or aryl or amid or i in arom) \
                        and not _planar3(atoms, i, 0.20, with_h=True):
                    C.acceptors[i] = {}
                continue
            if deg >= 4:
                pos_single.append(i)
            elif deg == 3:
                if i in arom or acyl or sulf or aryl or amid:
                    continue
                if _planar3(atoms, i, 0.20):
                    continue
                C.donors[i] = dict(nH=1, hyb="sp3")
                pos_single.append(i)
            elif deg == 2:
                if i in arom:
                    rids = ring_of[i]
                    six = any(len(arom_rings[r]) == 6 for r in rids)
                    five = [r for r in rids if len(arom_rings[r]) == 5]
                    if acyl:                                # lactam N-H (pyridone, uracil)
                        C.donors[i] = dict(nH=1, hyb="sp2")
                    elif five and not six:
                        ring = arom_rings[five[0]]
                        chalc = any(atoms[k].elem in ("O", "S", "Se") for k in ring)
                        subst = any(atoms[k].elem == "N" and hd(k) == 3 for k in ring)
                        cand = [k for k in ring if atoms[k].elem == "N" and hd(k) == 2]
                        n_n = sum(1 for k in ring if atoms[k].elem == "N")
                        if chalc or subst:
                            C.acceptors[i] = {}
                        elif len(cand) == 1:
                            C.donors[i] = dict(nH=1, hyb="sp2")
                        elif n_n >= 4:                      # tetrazole: anionic at pH 7
                            C.acceptors[i] = {}
                            tetrazoles[five[0]] = cand
                        else:                               # imidazole, pyrazole, triazole
                            C.donors[i] = dict(nH=1, hyb="sp2", amb=True)
                            C.acceptors[i] = dict(amb=True)
                    else:
                        ring = arom_rings[[r for r in rids if len(arom_rings[r]) == 6][0]]
                        para = ring[(ring.index(i) + 3) % 6]
                        amino = [k for k in atoms[para].nb if k not in ring and atoms[k].elem == "N"
                                 and k not in arom and hd(k) <= 2
                                 and not any(x in carbonyl or x in oxo_s for x in atoms[k].nb)]
                        if amino and sum(1 for k in ring if atoms[k].elem == "N") == 1:
                            C.donors[i] = dict(nH=1, hyb="sp2")
                            aminopyridinium.append([i] + amino)
                        else:
                            C.acceptors[i] = {}
                elif acyl or amid:
                    C.donors[i] = dict(nH=1, hyb="sp2")
                elif sulf:
                    C.donors[i] = dict(nH=1, hyb="sp3")
                else:
                    jmin = min(a.nb, key=lambda j: dist(i, j))
                    if (dist(i, jmin) < 1.31 or order(i, jmin) == 2) \
                            and atoms[jmin].elem in ("C", "N") and hd(jmin) <= 3:
                        C.acceptors[i] = {}
                    elif aryl:
                        C.donors[i] = dict(nH=1, hyb="sp2")
                    else:
                        C.donors[i] = dict(nH=2, hyb="sp3")
                        pos_single.append(i)
            elif deg == 1:
                j = a.nb[0]
                ej, d = atoms[j].elem, dist(i, j)
                if ej == "C" and hd(j) <= 2 and (d < 1.24 or order(i, j) == 3):
                    C.acceptors[i] = {}                     # nitrile
                elif ej == "N":
                    if d >= 1.25:
                        C.donors[i] = dict(nH=2, hyb="sp2")
                elif acyl or amid or aryl:
                    C.donors[i] = dict(nH=2, hyb="sp2")
                elif sulf:
                    C.donors[i] = dict(nH=2, hyb="sp3")
                elif ej == "C" and hd(j) == 3 and _planar3(atoms, j) and d < 1.40:
                    C.donors[i] = dict(nH=2, hyb="sp2")
                else:
                    C.donors[i] = dict(nH=3, hyb="sp3")
                    pos_single.append(i)
            else:
                C.donors[i] = dict(nH=4, hyb="free")
                pos_single.append(i)
        elif e in ("Cl", "Br", "I"):
            if deg == 1 and atoms[a.nb[0]].elem == "C":
                C.xdonors[i] = a.nb[0]

    # ---- charged groups
    used = set()
    for c, os_ in carboxylate.items():
        C.neg.append(_group(atoms, os_, "carboxylate"))
        used.update(os_)
    for p, os_ in oxo_p.items():
        if len(os_) >= 2 and not any(atoms[o].hnb for o in os_):
            C.neg.append(_group(atoms, os_, "phosphate", atoms[p].xyz))
            used.update(os_)
    for s, os_ in oxo_s.items():
        if len(os_) >= 3:
            C.neg.append(_group(atoms, os_, "sulfonate", atoms[s].xyz))
            used.update(os_)
    for rid, ns in tetrazoles.items():
        C.neg.append(_group(atoms, ns, "tetrazolate",
                            np.mean([atoms[k].xyz for k in arom_rings[rid]], axis=0)))
        used.update(ns)
    for i in pos_single:
        C.pos.append(_group(atoms, [i], "ammonium"))
        used.add(i)
    for grp in aminopyridinium:
        C.pos.append(_group(atoms, grp, "aminopyridinium"))
        used.update(grp)
    for c, ns in amidine.items():
        C.pos.append(_group(atoms, ns, "guanidinium" if len(ns) >= 3 else "amidinium", atoms[c].xyz))
        used.update(ns)
    for i in idxs:
        a = atoms[i]
        if a.fc and i not in used and a.elem in ("N", "O", "S", "P"):
            if any(atoms[j].fc * a.fc < 0 for j in a.nb):
                continue                # charge-separated drawing of a neutral group (nitro)
            (C.pos if a.fc > 0 else C.neg).append(_group(atoms, [i], "formal charge"))

    # ---- electrophilic centres (covalent warheads)
    for c, o in carbonyl.items():
        if any(atoms[j].elem == "O" and atoms[j].hnb for j in atoms[c].nb):
            continue                    # carboxylic acid
        others = sorted(atoms[j].elem for j in atoms[c].nb if j != o)
        kind = {("N", "O"): "carbamate", ("C", "O"): "ester", ("C", "N"): "amide",
                ("N", "N"): "urea", ("C", "C"): "ketone", ("O", "O"): "carbonate",
                ("C",): "aldehyde", ("C", "S"): "thioester"}.get(tuple(others), "carbonyl")
        if kind == "amide" and any(len(r) == 4 and c in r for r in rings):
            kind = "beta-lactam"
        C.electrophiles[c] = dict(kind=kind, ref=o)
    nitrile_c = set()
    for i in idxs:
        a = atoms[i]
        if a.elem == "C" and hd(i) <= 2:
            tn = [j for j in a.nb if atoms[j].elem == "N" and hd(j) == 1
                  and (dist(i, j) < 1.24 or order(i, j) == 3)]
            if tn:
                C.electrophiles[i] = dict(kind="nitrile", ref=tn[0])
                nitrile_c.add(i)
        elif a.elem == "B":
            C.electrophiles[i] = dict(kind="boronic acid", ref=None)
        elif a.elem == "P" and len(oxo_p.get(i, ())) == 1 and hd(i) == 4:
            C.electrophiles[i] = dict(kind="phosph(on)ate ester", ref=None)
        elif a.elem == "S" and len(oxo_s.get(i, ())) >= 2 and any(atoms[j].elem == "F" for j in a.nb):
            C.electrophiles[i] = dict(kind="sulfonyl fluoride", ref=None)

    def sp2c(i):
        a = atoms[i]
        return (a.elem == "C" and i not in arom and hd(i) + len(a.hnb) <= 3
                and (hd(i) != 3 or _planar3(atoms, i)))

    ewg = set(carbonyl) | nitrile_c | set(s for s, os_ in oxo_s.items() if len(os_) >= 2)
    for w in ewg:
        for al in atoms[w].nb:
            if al in carbonyl or not sp2c(al):
                if atoms[al].elem == "C" and w in carbonyl and \
                        any(atoms[x].elem in ("Cl", "Br", "I") for x in atoms[al].nb):
                    C.electrophiles.setdefault(al, dict(kind="alpha-halo carbonyl", ref=None))
                continue
            for be in atoms[al].nb:
                if be != w and be not in carbonyl and sp2c(be) and dist(al, be) < 1.38:
                    C.electrophiles.setdefault(be, dict(kind="Michael acceptor", ref=al))
    for ring in rings:
        if len(ring) == 3:
            het = [atoms[i].elem for i in ring if atoms[i].elem in ("O", "N")]
            if het:
                for i in ring:
                    if atoms[i].elem == "C":
                        C.electrophiles.setdefault(
                            i, dict(kind="epoxide" if het[0] == "O" else "aziridine", ref=None))


def _perceive(atoms, bo, his_positive=False, chain_start=None):
    C = _Chem(atoms, bo)
    by_res = OrderedDict()
    for a in atoms:
        by_res.setdefault(a.rkey, []).append(a.i)

    # Hydrogens added by filling valences (PyMOL's h_add) leave every lysine
    # neutral and every Asp/Glu protonated.  That is not a protonation state
    # anyone chose, so such hydrogens are ignored and the standard states used.
    n_ion = n_off = 0
    for rkey, idxs in by_res.items():
        resn = rkey[4].upper()
        if resn not in ("LYS", "ASP", "GLU"):
            continue
        names = dict((atoms[i].name, i) for i in idxs)
        if not any(atoms[i].hnb for i in idxs):
            continue
        if resn == "LYS" and "NZ" in names:
            n_ion += 1
            n_off += len(atoms[names["NZ"]].hnb) < 3
        elif resn != "LYS":
            os_ = [names[x] for x in AA_ACC[resn] if x in names]
            if len(os_) == 2:
                n_ion += 1
                n_off += any(atoms[o].hnb for o in os_)
    if n_ion >= 2 and n_off > 0.6 * n_ion:
        for a in atoms:
            a.hnb = []
        C.notes.append("the hydrogens look valence-filled (e.g. PyMOL h_add: neutral Lys, protonated "
                       "Asp/Glu); they were ignored and standard protonation states used.")

    has_h, generic = {}, []
    for rkey, idxs in by_res.items():
        resn = rkey[4].upper()
        heavy = [i for i in idxs if atoms[i].elem != "H"]
        has_h[rkey] = any(atoms[i].hnb for i in heavy)
        if not heavy:
            continue
        if resn in WATERS or (len(heavy) == 1 and atoms[heavy[0]].elem == "O" and not atoms[heavy[0]].nb):
            for i in heavy:
                C.waters.add(i)
                C.donors[i] = dict(nH=2, hyb="free")
                C.acceptors[i] = {}
            continue
        if len(heavy) == 1 and atoms[heavy[0]].elem in METALS:
            C.metals.add(heavy[0])
            continue
        names = set(atoms[i].name for i in heavy)
        if AA_ALIAS.get(resn, resn) in AA1 and "CA" in names and not all(atoms[i].lig for i in heavy):
            _perceive_aa(C, idxs, resn, his_positive, chain_start)
        elif AA_ALIAS.get(resn, resn) in AA1 and "CA" in names and "N" in names and "C" in names:
            _perceive_aa(C, idxs, resn, his_positive, chain_start)      # peptide ligand
        else:
            generic.extend(heavy)
            for i in heavy:
                if atoms[i].elem in METALS:
                    C.metals.add(i)
    _perceive_generic(C, generic, has_h)

    # hydrophobic atoms: carbon bonded only to C / S / H; thioether S; ligand halogens
    for a in atoms:
        if a.elem == "C":
            if all(atoms[j].elem in ("C", "S") for j in a.nb):
                C.hydrophobic.add(a.i)
        elif a.elem == "S":
            if len(a.nb) == 2 and all(atoms[j].elem == "C" for j in a.nb):
                C.hydrophobic.add(a.i)
        elif a.elem in ("F", "Cl", "Br", "I") and a.lig:
            if len(a.nb) == 1 and atoms[a.nb[0]].elem == "C":
                C.hydrophobic.add(a.i)

    # ring systems (fused rings share a system id)
    sys_of = list(range(len(C.rings)))
    for i in range(len(C.rings)):
        for j in range(i):
            if set(C.rings[i]["atoms"]) & set(C.rings[j]["atoms"]):
                old, new = sys_of[i], sys_of[j]
                sys_of = [new if s == old else s for s in sys_of]
    for r, s in zip(C.rings, sys_of):
        r["sys"] = s
    return C


# =============================================================================
#  interaction detection
# =============================================================================
def _donor_angle(C, d, target, info=None):
    """Best achievable D-H...A angle for donor atom d towards `target` (None = any)."""
    atoms = C.atoms
    a = atoms[d]
    info = info or C.donors[d]
    p, hyb = a.xyz, info["hyb"]
    best = 0.0
    if hyb == "expl":
        best = max(_angle(p, atoms[h].xyz, target) for h in a.hnb)
        # a hydroxyl / ammonium H added by a modelling tool points wherever the
        # tool left it; the group can turn, so judge it as a rotor as well
        if len(a.nb) == 1 and (a.elem in ("O", "S") or len(a.hnb) == 3):
            hyb = "sp3"
        else:
            return best
    deg = len(a.nb)
    if hyb == "free" or deg == 0:
        return None
    nbv = [_unit(atoms[j].xyz - p) for j in a.nb]
    hs = []
    cone = (0.334, 0.943)               # tetrahedral rotor: H 70.5 deg off the bond axis
    if hyb == "sp2free":                # amide N-H after a chain break: 60 deg, plane unknown
        hyb, cone = "sp3", (0.5, 0.866)
    if hyb == "sp2":
        if deg == 2:
            hs.append(p + 1.01 * _unit(-(nbv[0] + nbv[1])))
        elif deg == 1:
            j = a.nb[0]
            ax = _unit(p - atoms[j].xyz)
            refs = [k for k in atoms[j].nb if k != d]
            if refs:
                r = atoms[refs[0]].xyz - atoms[j].xyz
                b = _unit(r - np.dot(r, ax) * ax)
                if float(np.linalg.norm(b)) < 1e-6:
                    b = _perp(ax)
                hs.append(p + 1.01 * (0.5 * ax + 0.866 * b))
                hs.append(p + 1.01 * (0.5 * ax - 0.866 * b))
            else:
                hyb = "sp3"
    if hyb == "sp3":
        if deg >= 3:
            hs.append(p + 1.01 * _unit(-np.sum(nbv, axis=0)))
        elif deg == 2:
            bis, nrm = _unit(-(nbv[0] + nbv[1])), _unit(np.cross(nbv[0], nbv[1]))
            hs.append(p + 1.01 * (0.577 * bis + 0.816 * nrm))
            hs.append(p + 1.01 * (0.577 * bis - 0.816 * nrm))
        elif deg == 1:                      # rotor: H on a cone, turned towards the target
            ax = _unit(p - atoms[a.nb[0]].xyz)
            w = _unit(target - p)
            perp = w - np.dot(w, ax) * ax
            perp = _unit(perp) if float(np.linalg.norm(perp)) > 1e-6 else _perp(ax)
            hs.append(p + 1.0 * (cone[0] * ax + cone[1] * perp))
    if not hs:
        return best
    return max(best, max(_angle(p, h, target) for h in hs))


def _acceptor_ok(C, acc, target, min_angle):
    atoms = C.atoms
    p = atoms[acc].xyz
    return all(_angle(atoms[k].xyz, p, target) >= min_angle for k in atoms[acc].nb)


def _detect(C, lig, env, wanted=None):
    """All ligand-environment interactions.  Returns a list of dicts.
    `wanted` limits the types; it matters here because an H-bond between two
    charged groups is reported as a salt bridge only if salt bridges are asked for."""
    atoms, P = C.atoms, PARAM
    out = []
    if not lig or not env:
        return out
    lig_set, env_set = set(lig), set(env)
    XL = np.array([atoms[i].xyz for i in lig])
    XE = np.array([atoms[i].xyz for i in env])
    D = np.sqrt(((XL[:, None, :] - XE[None, :, :]) ** 2).sum(-1))
    lpos = {i: k for k, i in enumerate(lig)}
    epos = {i: k for k, i in enumerate(env)}

    def dd(i, j):       # i ligand, j environment
        return float(D[lpos[i], epos[j]])

    def add(itype, la, ea, dist, **kw):
        rec = dict(type=itype, la=la, ea=ea, dist=float(dist), angle=None, note="",
                   res=None, p_lig=None, p_env=None, draw=True)
        rec.update(kw)
        if rec["res"] is None and ea is not None:
            rec["res"] = atoms[ea].rkey
        if rec["p_lig"] is None and la is not None:
            rec["p_lig"] = atoms[la].xyz
        if rec["p_env"] is None and ea is not None:
            rec["p_env"] = atoms[ea].xyz
        out.append(rec)
        return rec

    # ---- covalent links --------------------------------------------------
    excl = set()
    li, ei = np.where(D < 2.6)
    links = []
    for k, m in zip(li, ei):
        i, j = lig[k], env[m]
        a, b = atoms[i], atoms[j]
        if a.elem in METALS or b.elem in METALS:
            continue
        if j in a.nb or _bond_length_ok(a.elem, b.elem, float(D[k, m])):
            links.append((i, j, float(D[k, m])))
    if len(links) > 3:
        links = []          # a ligand "bonded" all over the pocket is simply overlapping it
    for i, j, dist in links:
        a, b = atoms[i], atoms[j]
        add("covalent", i, j, dist, note="" if j in a.nb else "not bonded in the file")
        for x in [i] + [n for n in a.nb if n in lig_set]:
            for y in [j] + [n for n in b.nb if n in env_set]:
                excl.add((x, y))

    # ---- what sits in the first coordination shell of a metal -----------------
    metal_bound = {}
    for m in C.metals:
        if m not in lig_set and m not in env_set:
            continue
        for q in lig + env:
            if q != m and atoms[q].elem in ("N", "O", "S", "Se") and \
                    float(np.linalg.norm(atoms[q].xyz - atoms[m].xyz)) <= P["metal_dist"]:
                metal_bound.setdefault(q, set()).add(m)

    # ---- hydrogen bonds ----------------------------------------------------
    cands = []
    for don_side, acc_side, don_lig in ((lig_set, env_set, True), (env_set, lig_set, False)):
        dons = [i for i in C.donors if i in don_side]
        accs = [i for i in C.acceptors if i in acc_side]
        for d in dons:
            for a in accs:
                i, j = (d, a) if don_lig else (a, d)
                if (i, j) in excl:
                    continue
                if a in metal_bound and atoms[a].elem == "N":
                    continue            # its lone pair is on the metal
                if d in metal_bound and (d in C.aromatic or metal_bound[d] & metal_bound.get(a, set())):
                    continue            # two ligands of the same metal are not H-bonded
                dist = dd(i, j)
                if dist > P["hbond_dist"] or dist < P["hbond_min"]:
                    continue
                ang = _donor_angle(C, d, atoms[a].xyz)
                if ang is not None and ang < P["hbond_angle"]:
                    continue
                if not _acceptor_ok(C, a, atoms[d].xyz, P["hbond_acc_angle"]):
                    continue
                cands.append((dist, d, a, ang, don_lig))
    # Two acceptors at H-bond distance share a proton: an aromatic / imine N that
    # sits on a carboxylate or phosphate O is protonated there (or the acid is).
    anionic = set(i for g in C.neg for i in g["atoms"] if atoms[i].elem == "O")
    implied = {}
    for n_side, o_side, n_lig in ((lig_set, env_set, True), (env_set, lig_set, False)):
        for n in C.acceptors:
            if n not in n_side or atoms[n].elem != "N" or n in C.donors or n in metal_bound:
                continue
            if len(atoms[n].nb) != 2:
                continue
            for o in anionic:
                if o not in o_side:
                    continue
                i, j = (n, o) if n_lig else (o, n)
                if (i, j) in excl:
                    continue
                dist = dd(i, j)
                if dist > 3.1 or dist < P["hbond_min"]:
                    continue
                ang = _donor_angle(C, n, atoms[o].xyz, dict(nH=1, hyb="sp2"))
                if ang < P["hbond_angle"] or not _acceptor_ok(C, o, atoms[n].xyz, P["hbond_acc_angle"]):
                    continue
                cands.append((dist, n, o, ang, n_lig))
                implied[(n, o)] = True
    # nearest first, but a water never takes a slot a protein atom could fill
    cands.sort(key=lambda c: (c[1] in C.waters or c[2] in C.waters, c[0]))
    don_used, acc_used, seen = defaultdict(int), defaultdict(int), set()
    hbonds = []
    for dist, d, a, ang, don_lig in cands:
        pair = frozenset((d, a))
        if pair in seen:
            continue
        if don_used[d] >= (1 if (d, a) in implied else C.donors[d]["nH"]):
            continue
        if acc_used[a] >= (3 if atoms[a].elem == "O" else 1):
            continue
        seen.add(pair)
        don_used[d] += 1
        acc_used[a] += 1
        i, j = (d, a) if don_lig else (a, d)
        amb = C.donors.get(d, {}).get("amb") or C.acceptors[a].get("amb")
        bb = atoms[j].name in BACKBONE
        note = "%s donates" % ("ligand" if don_lig else "protein")
        if (d, a) in implied:
            note = "proton implied on %s N (or on the acid)" % ("ligand" if don_lig else "protein")
        if bb:
            note += ", backbone"
        if amb:
            note += ", tautomer assumed"
        rec = add("hbond", i, j, dist, angle=ang, note=note, donor=d, acceptor=a)
        if j in C.waters:
            rec["type"] = "water"       # direct ligand-water contact; used for bridges only
            rec["draw"] = False
        hbonds.append(rec)

    # ---- salt bridges ------------------------------------------------------
    for l_groups, e_groups in ((C.pos, C.neg), (C.neg, C.pos)):
        if wanted is not None and "saltbridge" not in wanted:
            break
        for gl in l_groups:
            ga = [i for i in gl["atoms"] if i in lpos]
            if not ga:
                continue
            for ge in e_groups:
                gb = [j for j in ge["atoms"] if j in epos]
                if not gb:
                    continue
                best = min(((dd(i, j), i, j) for i in ga for j in gb), key=lambda t: t[0])
                if best[0] > P["salt_dist"] or (best[1], best[2]) in excl:
                    continue
                ga, gb = set(ga), set(gb)
                shared = [h for h in hbonds if h["la"] in ga and h["ea"] in gb]
                if shared:
                    for h in shared:
                        h["type"] = "saltbridge"
                        h["note"] = "%s...%s, H-bonded" % (gl["kind"], ge["kind"])
                else:
                    add("saltbridge", best[1], best[2], best[0],
                        note="%s...%s" % (gl["kind"], ge["kind"]))

    # ---- pi-stacking -------------------------------------------------------
    best_pi = {}
    for rl in C.rings:
        if not all(k in lig_set for k in rl["atoms"]):
            continue
        for re_ in C.rings:
            if not all(k in env_set for k in re_["atoms"]):
                continue
            v = re_["center"] - rl["center"]
            dist = float(np.linalg.norm(v))
            if dist > P["pi_dist"]:
                continue
            ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(rl["normal"], re_["normal"]))))))
            off = min(float(np.linalg.norm(v - np.dot(v, rl["normal"]) * rl["normal"])),
                      float(np.linalg.norm(v - np.dot(v, re_["normal"]) * re_["normal"])))
            if off > P["pi_offset"]:
                continue
            if ang <= P["pi_angle"]:
                kind = "parallel"
            elif ang >= 90.0 - P["pi_angle"]:
                kind = "T-shaped"
            else:
                continue
            key = (rl["sys"], re_["res"])
            if key not in best_pi or dist < best_pi[key][0]:
                best_pi[key] = (dist, rl, re_, ang, kind, off)
    stacked = set()
    for dist, rl, re_, ang, kind, off in best_pi.values():
        add("pistack", None, None, dist, angle=ang, res=re_["res"], p_lig=rl["center"],
            p_env=re_["center"], note="%s, offset %.1f A" % (kind, off),
            lig_atoms=rl["atoms"], env_atoms=re_["atoms"])
        stacked.add(re_["res"])

    # ---- cation-pi -----------------------------------------------------------
    for rings, cats, ring_is_lig in ((C.rings, C.pos, True), (C.rings, C.pos, False)):
        ring_side = lig_set if ring_is_lig else env_set
        cat_side = env_set if ring_is_lig else lig_set
        best_cp = {}
        for r in rings:
            if not all(k in ring_side for k in r["atoms"]):
                continue
            for g in cats:
                if not any(k in cat_side for k in g["atoms"]):
                    continue
                v = g["center"] - r["center"]
                dist = float(np.linalg.norm(v))
                if dist > P["cation_pi_dist"]:
                    continue
                off = float(np.linalg.norm(v - np.dot(v, r["normal"]) * r["normal"]))
                if off > P["pi_offset"]:
                    continue
                res = g["res"] if ring_is_lig else r["res"]
                if res in stacked and any(k in C.aromatic for k in g["atoms"]):
                    continue
                key = (r["sys"], tuple(g["atoms"]))
                if key not in best_cp or dist < best_cp[key][0]:
                    best_cp[key] = (dist, r, g, res, off)
        for dist, r, g, res, off in best_cp.values():
            if ring_is_lig:
                add("pication", None, None, dist, res=res, p_lig=r["center"], p_env=g["center"],
                    note="ligand ring...%s" % g["kind"], lig_atoms=r["atoms"], env_atoms=g["atoms"])
            else:
                add("pication", None, None, dist, res=res, p_lig=g["center"], p_env=r["center"],
                    note="ligand %s...ring" % g["kind"], lig_atoms=g["atoms"], env_atoms=r["atoms"])

    # ---- halogen bonds -------------------------------------------------------
    for x, c in C.xdonors.items():
        x_lig = x in lig_set
        if not x_lig and x not in env_set:
            continue
        for a in C.acceptors:
            if (a in lig_set) == x_lig or (a not in lig_set and a not in env_set):
                continue
            i, j = (x, a) if x_lig else (a, x)
            dist = dd(i, j)
            if dist > VDW.get(atoms[x].elem, 1.8) + VDW.get(atoms[a].elem, 1.5) + P["halogen_slack"]:
                continue
            a1 = _angle(atoms[c].xyz, atoms[x].xyz, atoms[a].xyz)
            if a1 < P["halogen_don_angle"]:
                continue
            a2 = [_angle(atoms[x].xyz, atoms[a].xyz, atoms[k].xyz) for k in atoms[a].nb]
            if a2 and not all(P["halogen_acc_min"] <= t <= P["halogen_acc_max"] for t in a2):
                continue
            add("halogen", i, j, dist, angle=a1, note="C-%s...%s" % (atoms[x].elem, atoms[a].elem))

    # ---- metal coordination --------------------------------------------------
    shell_of = set()
    for m in C.metals:
        m_lig = m in lig_set
        if not m_lig and m not in env_set:
            continue
        partners = lig if not m_lig else env
        for q in partners:
            if atoms[q].elem not in ("N", "O", "S", "Cl", "Br", "I", "F", "Se"):
                continue
            if atoms[q].elem in ("S", "Se") and any(atoms[k].elem == "O" for k in atoms[q].nb):
                continue
            i, j = (q, m) if not m_lig else (m, q)
            dist = dd(i, j)
            if dist <= P["metal_dist"] + (0.3 if atoms[q].elem in ("S", "Se", "Cl", "Br", "I") else 0.0):
                add("metal", i, j, dist, note="%s...%s" % (atoms[m].elem, atoms[q].elem))
                if not m_lig:
                    shell_of.add(m)
    for m in shell_of:                  # protein atoms on a metal the ligand binds
        for q, ms in metal_bound.items():
            if m in ms and q in env_set and q not in C.waters:
                add("metal", m, q, float(np.linalg.norm(atoms[q].xyz - atoms[m].xyz)), shell=True,
                    note="%s...%s of the protein" % (atoms[m].elem, atoms[q].elem))

    # ---- water bridges -------------------------------------------------------
    polar_l = [i for i in lig if i in C.donors or i in C.acceptors]
    polar_e = [j for j in env if (j in C.donors or j in C.acceptors) and j not in C.waters]
    for w in C.waters:
        if w not in env_set:
            continue
        pw = atoms[w].xyz
        ls = sorted((dd(i, w), i) for i in polar_l if P["hbond_min"] <= dd(i, w) <= P["water_dist"])
        if not ls:
            continue
        es = sorted((float(np.linalg.norm(atoms[j].xyz - pw)), j) for j in polar_e)
        es = [(d, j) for d, j in es if P["hbond_min"] <= d <= P["water_dist"]]
        dl, i = ls[0]
        n = 0
        for de, j in es:
            om = _angle(atoms[i].xyz, pw, atoms[j].xyz)
            if not (P["water_angle_min"] <= om <= P["water_angle_max"]):
                continue
            add("waterbridge", i, j, dl, angle=om, water=w,
                note="via %s%s (%.1f + %.1f A)" % (atoms[w].resn, atoms[w].resi, dl, de))
            n += 1
            if n == 2:
                break

    # ---- hydrophobic contacts: the closest pair per residue -------------------
    hl = [i for i in lig if i in C.hydrophobic]
    he = [j for j in env if j in C.hydrophobic and j not in C.waters]
    best_h = {}
    for i in hl:
        for j in he:
            dist = dd(i, j)
            if dist < VDW.get(atoms[i].elem, 1.7) + VDW.get(atoms[j].elem, 1.7) - P["clash_overlap"]:
                continue                # that is a clash, not a contact
            if dist <= P["hydrophobic_dist"] and (i, j) not in excl:
                key = atoms[j].rkey
                if key not in best_h or dist < best_h[key][0]:
                    best_h[key] = (dist, i, j)
    for key, (dist, i, j) in best_h.items():
        add("hydrophobic", i, j, dist, draw=key not in stacked)

    # ---- near-attack geometry of covalent warheads ----------------------------
    cov_env = set(r["ea"] for r in out if r["type"] == "covalent")
    for e, info in C.electrophiles.items():
        if e not in lig_set:
            continue
        for nu, what in C.nucleophiles.items():
            if nu not in env_set or nu in cov_env or (e, nu) in excl:
                continue
            dist = dd(e, nu)
            if dist > P["nac_dist"] + (0.3 if atoms[nu].elem == "S" else 0.0):
                continue
            ang = None
            ok = True
            if info["ref"] is not None:
                ang = _angle(atoms[nu].xyz, atoms[e].xyz, atoms[info["ref"]].xyz)
                ok = P["nac_angle_min"] <= ang <= P["nac_angle_max"]
            if not ok and info["kind"] not in WARHEADS:
                continue                # an ordinary amide/ketone that merely sits nearby
            add("nearattack", e, nu, dist, angle=ang, draw=ok,
                note="%s -> %s%s" % (what, info["kind"], "" if ok else ", off-angle"))

    # ---- clashes ---------------------------------------------------------------
    li, ei = np.where(D < 3.1)
    for k, m in zip(li, ei):
        i, j = lig[k], env[m]
        a, b = atoms[i], atoms[j]
        if (i, j) in excl or a.elem in METALS or b.elem in METALS:
            continue
        dist = float(D[k, m])
        vsum = VDW.get(a.elem, 1.7) + VDW.get(b.elem, 1.7)
        polar = a.elem in ("N", "O") and b.elem in ("N", "O")
        if dist < (P["hbond_min"] if polar else vsum - P["clash_overlap"]):
            add("clash", i, j, dist, note="vdW overlap %.1f A" % (vsum - dist))

    return out


# =============================================================================
#  ligand discovery
# =============================================================================
def find_ligands(selection="enabled", state=-1):
    """Candidate ligands in `selection`, best first.  Each is a dict with keys
    sel, pretty, obj, resn, natoms, contacts, kind."""
    found, whole_chains = [], set()
    selection = _not_ours(selection)
    poly = _coords("(%s) and polymer and not hydro" % selection, state)

    def contacts(sel):
        """(protein atoms within 4.5 A, heavy atoms) of a candidate."""
        rows = _coords("(%s) and not hydro" % sel, state)
        if not rows:
            return 0, 0
        pts = np.vstack([X for _, _, X in rows])
        mine = set((obj, int(i)) for obj, idx, _ in rows for i in idx)
        n = sum(1 for obj, (idx, _) in _near(poly, pts, 4.5).items()
                for i in idx if (obj, int(i)) not in mine)
        return n, len(pts)

    def chain_sel(obj, segi, chain):
        text = "%%%s and chain %s" % (obj, chain if chain else "''")
        return "(%s and segi %s)" % (text, segi) if segi else "(%s)" % text

    def add_chain(obj, segi, chain):
        """A short chain (peptide, peptidomimetic) taken as one ligand."""
        if (obj, segi, chain) in whole_chains:
            return
        whole_chains.add((obj, segi, chain))
        csel = "(%s and (polymer or organic))" % chain_sel(obj, segi, chain)
        n, natoms = contacts(csel)
        if n:
            pretty = "%s and %s and (polymer or organic)" % (
                obj, "chain %s" % chain if chain else "segi %s" % segi)
            found.append(dict(sel=csel, pretty=pretty, obj=obj,
                              resn="peptide", resi="", chain=chain, natoms=natoms,
                              contacts=n, covalent=False, kind="peptide"))

    for obj in cmd.get_object_list("(%s) and organic" % selection) or []:
        model = cmd.get_model("(%s) and organic and %%%s" % (selection, obj), _object_state(obj, state))
        parent = {}

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        keys = []
        for at in model.atom:
            key = (at.segi, at.chain, at.resi, at.resn)
            keys.append(key)
            parent.setdefault(key, key)
        for b in model.bond:
            ra, rb = root(keys[b.index[0]]), root(keys[b.index[1]])
            if ra != rb:
                parent[ra] = rb
        groups = OrderedDict()
        for at, key in zip(model.atom, keys):
            groups.setdefault(root(key), []).append((at, key))
        for members in groups.values():
            ids = [int(at.index) for at, _ in members]
            n_heavy = sum(1 for at, _ in members if _norm_elem(at.symbol, at.name) != "H")
            res = list(OrderedDict.fromkeys(key for _, key in members))
            segi, chain, resi, resn = res[0]
            # a hetero group that is one building block of a short peptide-like chain
            if chain or segi:
                n_poly = cmd.count_atoms("%s and polymer and name CA" % chain_sel(obj, segi, chain))
                if 0 < n_poly <= 25:
                    add_chain(obj, segi, chain)
                    continue
            sel = "(%%%s and index %s)" % (obj, "+".join(_ranges(ids)))
            n, _ = contacts(sel)
            tmp = "_bs_cand"
            cmd.select(tmp, sel, enable=0, quiet=1)
            linked = cmd.count_atoms("polymer and (neighbor %s)" % tmp)
            cmd.delete(tmp)
            resns = [r[3].upper() for r in res]
            if all(r in ADDITIVES for r in resns):
                kind = "additive"
            elif all(r in SUGARS for r in resns) and linked:
                kind = "glycan"
            elif n_heavy < 5:
                kind = "fragment"
            else:
                kind = "ligand"
            pretty = "%s and resn %s" % (obj, "+".join(OrderedDict.fromkeys(r[3] for r in res)))
            if chain:
                pretty += " and chain %s" % chain
            elif segi:
                pretty += " and segi %s" % segi
            pretty += " and resi %s" % "+".join(r[2].replace("-", "\\-") for r in res)
            found.append(dict(sel=sel, pretty=pretty, obj=obj, resn="-".join(OrderedDict.fromkeys(resns)),
                              resi=resi, chain=chain, natoms=n_heavy, contacts=n,
                              covalent=bool(linked), kind=kind))
    # peptides: short polymer chains bound to a longer one
    for obj in cmd.get_object_list("(%s) and polymer" % selection) or []:
        seen = OrderedDict()
        cmd.iterate("(%s) and polymer and %%%s and name CA" % (selection, obj),
                    "seen[(segi, chain)] = seen.get((segi, chain), 0) + 1", space={"seen": seen})
        if len(seen) < 2:
            continue
        for (segi, chain), n_res in seen.items():
            if n_res <= 25:
                add_chain(obj, segi, chain)
    rank = {"ligand": 0, "peptide": 0, "fragment": 1, "glycan": 2, "additive": 3}
    found.sort(key=lambda f: (rank[f["kind"]], -f["contacts"], -f["natoms"]))
    return found


# =============================================================================
#  analysis: selection -> site record
# =============================================================================
def _offset_for(offset, chain):
    if isinstance(offset, dict):
        return offset.get(chain, offset.get("*", 0))
    return offset


def _parse_offset(text):
    """'27' -> 27 ;  'A:27,B:-21' -> {'A': 27, 'B': -21}"""
    if isinstance(text, (int, float)):
        return int(text)
    if isinstance(text, dict):
        return text
    text = str(_unquote(str(text))).strip()
    if not text:
        return 0
    if ":" not in text:
        return int(float(text))
    out = {}
    for part in re.split(r"[,; ]+", text):
        if part:
            ch, val = part.split(":")
            out[ch.strip()] = int(float(val))
    return out


def _res_label(atoms, idxs, fmt, offset, multi_chain):
    a = atoms[idxs[0]]
    resn = a.resn.upper()
    canon = AA_ALIAS.get(resn, resn)
    m = re.match(r"(-?\d+)(.*)", a.resi or "")
    num, icode = (int(m.group(1)), m.group(2)) if m else (a.resv, "")
    num += _offset_for(offset, a.chain)
    is_aa = canon in AA1
    text = fmt.format(resn=canon.capitalize() if is_aa else resn, RESN=canon,
                      r=AA1.get(canon, resn), resi="%d%s" % (num, icode), chain=a.chain,
                      segi=a.segi, obj=a.obj)
    if multi_chain and "{chain" not in fmt and a.chain:
        text += " (%s)" % a.chain
    return text


def _analyse(lig_sel, recv_sel, state, opts):
    st = state if state > 0 else cmd.get_state()
    note = None
    try:
        cmd.select("_bs_lig", _not_ours("(%s)" % lig_sel), enable=0, quiet=1)
        if not cmd.count_atoms("_bs_lig"):
            raise CmdException("bindingsite: the ligand selection '%s' is empty" % lig_sel)
        lig_rows = _coords("_bs_lig and not hydro", st)
        if not lig_rows:
            raise CmdException("bindingsite: the ligand has no heavy atoms with coordinates "
                               "in state %d" % st)
        L = np.vstack([X for _, _, X in lig_rows])
        # Receptor: by default whatever shares the ligand's object, so that other
        # complexes loaded alongside stay out of the picture.  A ligand that is
        # alone in its object (docking poses) takes the other enabled objects.
        default = recv_sel in ("", "auto", None)
        scope = "(%s)" % " or ".join("%" + obj for obj, _, _ in lig_rows) if default else recv_sel
        seed = OrderedDict()
        for attempt in (0, 1):
            cmd.select("_bs_recv", "%s and not _bs_lig" % _not_ours(scope), enable=0, quiet=1)
            seed = _near(_coords("_bs_recv", st), L, 8.0)
            if seed or not default or attempt:
                break
            scope = "enabled"
            note = "the ligand is alone in its object: the other enabled objects serve as receptor."
        cmd.select("_bs_env", "byres ((byres (%s)) extend 1)" % _osel(seed), enable=0, quiet=1)
        cmd.select("_bs_env", "(_bs_env and not _bs_lig) or _bs_lig", enable=0, quiet=1)
        lig_ids = set()
        cmd.iterate("_bs_lig", "ids.add((model, index))", space={"ids": lig_ids})
        atoms, bo, spurious = _extract("_bs_env", st, lig_ids)
        # where each polymer chain starts: tells a real N-terminus from a chain break
        chain_start = {}
        for obj in OrderedDict.fromkeys(a.obj for a in atoms):
            cmd.iterate("%%%s and polymer and name CA" % obj,
                        "d[(model, segi, chain)] = min(d.get((model, segi, chain), resv), resv)",
                        space={"d": chain_start, "min": min})
        # the wider protein, for the camera and the cut-aways
        poly = _coords("_bs_recv and polymer and not hydro", st)
        wide = _near(poly, L, 14.0)
        prot_xyz = np.vstack([X for _, X in wide.values()]) if wide else np.zeros((0, 3))
        trace = []
        for obj, (idx, X) in _near(_coords("_bs_recv and polymer and name CA+C4'", st), L, 26.0).items():
            trace.extend((obj, int(i), float(x[0]), float(x[1]), float(x[2])) for i, x in zip(idx, X))
        shell = _near(poly, L, 11.0)
        surface = None
        if shell:
            sobj = max(shell, key=lambda k: len(shell[k][0]))
            surface = (sobj, shell[sobj][0])
    finally:
        for tmp in ("_bs_lig", "_bs_recv", "_bs_env"):
            cmd.delete(tmp)

    C = _perceive(atoms, bo, opts["his_positive"], chain_start)
    if note:
        C.notes.append(note)
    done = " and removed" if opts["fix_bonds"] else ""
    if len(spurious) > 3:
        C.notes.append("PyMOL had guessed %d ligand-protein bonds from distance alone (%.2f-%.2f A); "
                       "none has a plausible bond length: treated as contacts%s."
                       % (len(spurious), min(t[2] for t in spurious), max(t[2] for t in spurious), done))
    for i, j, d in spurious:
        a, b = (atoms[i], atoms[j]) if atoms[i].lig else (atoms[j], atoms[i])
        if len(spurious) <= 3:
            C.notes.append("PyMOL had guessed a bond between ligand %s and %s%s %s (%.2f A), which is not a "
                           "plausible bond length: treated as a contact%s."
                           % (a.name, b.resn, b.resi, b.name, d, done))
        if opts["fix_bonds"]:
            cmd.unbond(_isel(atoms, [i]), _isel(atoms, [j]))
    lig = [a.i for a in atoms if a.lig and a.elem != "H"]
    env = [a.i for a in atoms if not a.lig and a.elem != "H"]
    if not lig:
        raise CmdException("bindingsite: the ligand has no heavy atoms in state %d" % st)
    inter = _detect(C, lig, env, opts["types"])
    if any(atoms[i].hnb for i in lig):
        C.notes.append("the ligand has explicit hydrogens: its protonation state and tautomer are "
                       "taken from the file as they are.")
    clashes = sorted((r for r in inter if r["type"] == "clash"), key=lambda r: r["dist"])
    for r in clashes[12:]:
        r["draw"] = False               # the picture needs the worst dozen, the table has all
    overlaps = [r["dist"] for r in inter if r["type"] == "clash" and r["dist"] < 1.3]
    if len(overlaps) >= 2:
        C.notes.append("the ligand sits ON TOP of protein atoms (%d contacts below 1.3 A, closest "
                       "%.2f A). The coordinates were probably mis-read, e.g. from a PDB file with a "
                       "4-letter residue name; load the mmCIF instead." % (len(overlaps), min(overlaps)))
    wanted = opts["types"]
    inter = [r for r in inter if r["type"] in wanted or r["type"] == "water"]

    # ---- residues of the pocket ------------------------------------------------
    XL = np.array([atoms[i].xyz for i in lig])
    residues = OrderedDict()
    by_res = OrderedDict()
    for j in env:
        by_res.setdefault(atoms[j].rkey, []).append(j)
    for rkey, idxs in by_res.items():
        X = np.array([atoms[j].xyz for j in idxs])
        mind = float(np.sqrt(((X[:, None, :] - XL[None, :, :]) ** 2).sum(-1)).min())
        resn = rkey[4].upper()
        if any(j in C.waters for j in idxs):
            cls = "water"
        elif len(idxs) == 1 and idxs[0] in C.metals:
            cls = "metal"
        elif AA_ALIAS.get(resn, resn) in AA1 or any(atoms[j].name == "CA" for j in idxs) \
                and any(atoms[j].name == "N" for j in idxs):
            cls = "protein"
        else:
            cls = "het"
        residues[rkey] = dict(cls=cls, atoms=idxs, mind=mind, tier=0, bb=set(), types=[],
                              shown=[], label=None)
    for r in inter:
        if r["type"] == "water" or r["res"] not in residues:
            continue
        info = residues[r["res"]]
        info["types"].append(r["type"])
        if not r["draw"] and r["type"] != "hydrophobic":
            continue
        tier = 2 if r["type"] == "hydrophobic" else 1
        info["tier"] = tier if info["tier"] == 0 else min(info["tier"], tier)
        for j in ([r["ea"]] if r["ea"] is not None else r.get("env_atoms", [])):
            if atoms[j].name in ("N", "O", "C", "OXT") and info["cls"] == "protein":
                info["bb"].add(atoms[j].name)
        if r.get("water") is not None:
            w = residues.get(atoms[r["water"]].rkey)
            if w is not None:
                w["tier"] = 1
    for rkey, info in residues.items():
        additive = info["cls"] == "het" and rkey[4].upper() in ADDITIVES
        if additive and info["tier"] == 2:
            info["tier"] = 0
        if info["tier"] == 0 and info["mind"] <= opts["radius"] and info["cls"] in ("protein", "het") \
                and not additive:
            info["tier"] = 3
        if info["cls"] == "metal" and info["mind"] <= PARAM["metal_dist"] + 0.3 and info["tier"] == 0:
            info["tier"] = 1

    # ---- which atoms of each residue to draw -------------------------------------
    by_name = {}
    for a in atoms:
        by_name[(a.rkey, a.name)] = a.i
    for rkey, info in residues.items():
        if not info["tier"]:
            continue
        idxs = info["atoms"]
        if info["cls"] != "protein":
            info["shown"] = list(idxs)
            continue
        resn = AA_ALIAS.get(rkey[4].upper(), rkey[4].upper())
        shown = [j for j in idxs if atoms[j].name not in BACKBONE or (resn == "PRO" and atoms[j].name == "N")]
        if info["bb"]:
            shown = list(idxs)
            n, c = by_name.get((rkey, "N")), by_name.get((rkey, "C"))
            if "N" in info["bb"] and n is not None:
                for k in atoms[n].nb:
                    if atoms[k].rkey != rkey and atoms[k].name == "C":     # previous residue
                        shown.append(k)
                        shown.extend(x for x in atoms[k].nb if atoms[x].name in ("O", "CA")
                                     and atoms[x].rkey == atoms[k].rkey)
            if ("O" in info["bb"] or "C" in info["bb"]) and c is not None:
                for k in atoms[c].nb:
                    if atoms[k].rkey != rkey and atoms[k].name == "N":     # next residue
                        shown.append(k)
                        shown.extend(x for x in atoms[k].nb if atoms[x].name == "CA"
                                     and atoms[x].rkey == atoms[k].rkey)
        info["shown"] = shown

    chains = set(k[2] for k, v in residues.items() if v["tier"] in (1, 2) and v["cls"] == "protein")
    for rkey, info in residues.items():
        if info["tier"] and info["cls"] in ("protein", "het", "metal"):
            if info["cls"] == "metal":
                info["label"] = atoms[info["atoms"][0]].elem
            else:
                info["label"] = _res_label(atoms, info["atoms"], opts["label_format"],
                                           opts["offset"], len(chains) > 1)

    order = list(ITYPES)
    inter.sort(key=lambda r: (order.index(r["type"]) if r["type"] in order else 99,
                              atoms[r["ea"]].resv if r["ea"] is not None else 0, r["dist"]))
    return dict(atoms=atoms, chem=C, lig=lig, env=env, inter=inter, residues=residues,
                prot_xyz=prot_xyz, trace=trace, surface=surface, state=st, opts=opts,
                lig_sel=lig_sel)


# =============================================================================
#  camera
# =============================================================================
def _sphere(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * k
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def _rank_views(site, n_dirs=400):
    """Score viewing directions; the best ones see the ligand unobstructed, flat,
    and with its interactions lying in the picture plane."""
    atoms = site["atoms"]
    L = np.array([atoms[i].xyz for i in site["lig"]])
    shown = [j for info in site["residues"].values() if info["tier"] in (1, 2) for j in info["shown"]]
    Sx = np.array([atoms[j].xyz for j in shown]) if shown else np.zeros((0, 3))
    Pr = site["prot_xyz"]
    vecs, mids = [], []
    for r in site["inter"]:
        if r["draw"] and r["type"] in SPECIFIC and r["p_lig"] is not None and r["p_env"] is not None:
            vecs.append(_unit(r["p_env"] - r["p_lig"]))
            mids.append(0.5 * (r["p_env"] + r["p_lig"]))
    vecs, mids = np.array(vecs).reshape(-1, 3), np.array(mids).reshape(-1, 3)
    front = np.vstack([L, Sx]) if len(Sx) else L
    dirs = _sphere(n_dirs)
    # ligand atoms that are not bonded neighbours, for the self-overlap term
    lpos = {i: k for k, i in enumerate(site["lig"])}
    far = np.ones((len(L), len(L)), dtype=bool)
    np.fill_diagonal(far, False)
    for i, k in lpos.items():
        for j in atoms[i].nb:
            if j in lpos:
                far[k, lpos[j]] = False
    rows = []
    for v in dirs:
        tl = L @ v
        ll = L - np.outer(tl, v)
        d2 = ((ll[:, None, :] - ll[None, :, :]) ** 2).sum(-1)
        selfo = float(((d2 < 0.75 ** 2) & far).any(1).mean())
        # ligand hidden behind protein
        if len(Pr):
            tp = Pr @ v
            pl = Pr - np.outer(tp, v)
            d2 = ((ll[:, None, :] - pl[None, :, :]) ** 2).sum(-1)
            hit = (d2 < 1.8 ** 2) & (tp[None, :] > tl[:, None] + 1.0)
            occ = float(hit.any(1).mean())
            layers = min(1.0, float(hit.sum()) / (4.0 * len(L)))
        else:
            occ = layers = 0.0
        # ligand hidden behind the drawn residues
        if len(Sx):
            ts = Sx @ v
            sl = Sx - np.outer(ts, v)
            d2 = ((ll[:, None, :] - sl[None, :, :]) ** 2).sum(-1)
            occ_s = float(((d2 < 1.0 ** 2) & (ts[None, :] > tl[:, None] + 0.8)).any(1).mean())
            behind = float(((d2 < 0.9 ** 2) & (ts[None, :] <= tl[:, None] + 0.8)).any(1).mean())
        else:
            occ_s = behind = 0.0
        # interactions seen end-on or hidden
        fore = float(np.abs(vecs @ v).mean()) if len(vecs) else 0.0
        if len(mids):
            tm = mids @ v
            ml = mids - np.outer(tm, v)
            tf = front @ v
            fl = front - np.outer(tf, v)
            d2 = ((ml[:, None, :] - fl[None, :, :]) ** 2).sum(-1)
            hid = float(((d2 < 0.9 ** 2) & (tf[None, :] > tm[:, None] + 1.2)).any(1).mean())
        else:
            hid = 0.0
        # how spread out the ligand looks from here
        w = np.linalg.eigvalsh(np.cov((ll - ll.mean(0)).T)) if len(L) > 2 else np.ones(3)
        spread = math.sqrt(max(w[1], 0.0) * max(w[2], 0.0))
        rows.append([occ, layers, occ_s, fore, hid, spread, selfo, behind])
    rows = np.array(rows)
    smax = rows[:, 5].max() or 1.0
    cost = (1.0 * rows[:, 0] + 0.35 * rows[:, 1] + 1.5 * rows[:, 2] + 0.9 * rows[:, 3]
            + 1.2 * rows[:, 4] + 1.3 * (1.0 - rows[:, 5] / smax) + 1.0 * rows[:, 6]
            + 0.5 * rows[:, 7])
    best = []
    for k in np.argsort(cost):
        if all(float(np.dot(dirs[k], b)) < 0.70 for _, b in best):
            best.append((float(cost[k]), dirs[k]))
        if len(best) == 6:
            break
    return best


def _camera(site, z_axis):
    """Complete camera frame for a viewing direction: landscape, protein below."""
    atoms = site["atoms"]
    pts = [atoms[i].xyz for i in site["lig"]]
    for info in site["residues"].values():
        if info["tier"] in (1, 2):
            pts.extend(atoms[j].xyz for j in info["shown"])
    pts = np.array(pts)
    Z = _unit(np.array(z_axis, dtype=float))
    flat = pts - np.outer(pts @ Z, Z)
    flat = flat - flat.mean(0)
    X = np.zeros(3)
    if len(flat) >= 3:
        vec = np.linalg.eigh(np.cov(flat.T))[1]
        X = _unit(vec[:, 2] - np.dot(vec[:, 2], Z) * Z)
    if float(np.linalg.norm(X)) < 1e-6:
        X = _perp(Z)
    Y = np.cross(Z, X)
    if len(site["prot_xyz"]):
        c = np.array([atoms[i].xyz for i in site["lig"]]).mean(0)
        if float(np.dot(site["prot_xyz"].mean(0) - c, Y)) > 0:
            X, Y = -X, -Y
    return X, Y, Z


def _set_camera(X, Y, Z):
    v = list(cmd.get_view())
    R = np.column_stack([X, Y, Z])
    v[0:9] = [float(x) for x in R.flatten()]
    cmd.set_view(v)


def _fit(site, aspect=None, margin=1.2):
    """Zoom/centre on everything drawn and set the clipping slab + fog."""
    atoms = site["atoms"]
    v = list(cmd.get_view())
    R = np.array(v[:9]).reshape(3, 3)
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
    lig = np.array([atoms[i].xyz for i in site["lig"]])
    pts = [lig]
    for info in site["residues"].values():
        if info["tier"] and info["shown"]:
            pts.append(np.array([atoms[j].xyz for j in info["shown"]]))
    pts = np.vstack(pts)
    x, y, z = pts @ X, pts @ Y, pts @ Z
    xs, ys = [x.min() - 0.8, x.max() + 0.8], [y.min() - 0.8, y.max() + 0.8]
    for (bx0, bx1, by0, by1) in site.get("label_boxes", []):
        xs += [bx0, bx1]
        ys += [by0, by1]
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    half_w, half_h = 0.5 * (max(xs) - min(xs)) + margin, 0.5 * (max(ys) - min(ys)) + margin
    zc = float((lig @ Z).mean())
    vp = cmd.get_viewport()
    if aspect is None:
        aspect = float(vp[0]) / float(vp[1]) if vp[1] else 1.33
    fov = abs(float(cmd.get("field_of_view")))
    # the colour key sits in the lower-left corner: if the picture reaches into
    # it, make room below
    rows = site.get("legend_rows", 0)
    if rows:
        vh = float(vp[1]) or 750.0
        px = max(11.0, 0.021 * vh)
        key_h = (rows * 1.6 * px + 0.03 * vh) / vh          # fraction of the image height
        key_w = (site.get("legend_chars", 20) * 0.62 * px + 0.03 * vh) / (vh * aspect)
        boxes = [(x[k] - 0.5, x[k] + 0.5, y[k] - 0.5, y[k] + 0.5) for k in range(len(x))]
        boxes += list(site.get("label_boxes", []))
        for _ in range(4):
            hh = max(half_h, half_w / aspect)
            hw = hh * aspect
            bx = cx - hw + 2.0 * hw * key_w                  # right edge of the key
            by = cy - hh + 2.0 * hh * key_h                  # top edge of the key
            if not any(b[0] < bx and b[2] < by for b in boxes):
                break
            grow = 2.0 * hh * key_h * 0.5
            cy -= 0.5 * grow
            half_h += 0.5 * grow
    dist = max(half_h, half_w / aspect) / math.tan(math.radians(fov) / 2.0)
    z_front = max(float(z.max()) + 3.0, site.get("label_front", -1e9) + 1.0) - zc
    z_back = float(z.min()) - 12.0 - zc
    origin = cx * X + cy * Y + zc * Z
    v[9:12] = [0.0, 0.0, -dist]
    v[12:15] = [float(t) for t in origin]
    v[15] = max(dist - z_front, 0.5)
    v[16] = dist - z_back
    cmd.set_view(v)
    # fog begins behind the deepest interacting residue: the site stays crisp,
    # the rest of the protein recedes into the background
    key = [atoms[j].xyz for info in site["residues"].values() if info["tier"] in (1, 2)
           for j in info["shown"]]
    deep = float((np.array(key) @ Z).min()) if key else float((lig @ Z).min())
    deep = min(deep, float((lig @ Z).min())) - zc - 1.0
    if "fog_start" not in _SAVED:
        _SAVED["fog_start"] = cmd.get("fog_start")
    _set("fog_start", max(0.05, min(0.92, (z_front - deep) / (z_front - z_back))))
    site["aspect"] = aspect


# =============================================================================
#  labels
# =============================================================================
def _place(items, obstacles, radius=0.42, pad=0.17):
    """Greedy free-space placement in the screen plane: every label takes the
    spot nearest its residue that covers neither a drawn atom nor another label."""
    radii = np.arange(0.0, 12.0, 0.25)
    angles = np.radians(np.arange(0.0, 360.0, 10.0))
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    rr, aa = rr.ravel(), aa.ravel()
    boxes = []
    for k in sorted(range(len(items)), key=lambda k: items[k]["prio"]):
        it = items[k]
        hx, hy = it["half"][0] + pad, it["half"][1] + pad
        base = math.atan2(it["out"][1], it["out"][0])
        cand = it["anchor"][None, :] + np.column_stack([rr * np.cos(base + aa), rr * np.sin(base + aa)])
        cost = rr * (1.0 + 0.6 * (1.0 - np.cos(aa)))        # near the residue, preferably outward
        free = np.ones(len(cand), dtype=bool)
        if len(obstacles):
            dx = np.abs(cand[:, None, 0] - obstacles[None, :, 0])
            dy = np.abs(cand[:, None, 1] - obstacles[None, :, 1])
            free &= ~((dx < hx + radius) & (dy < hy + radius)).any(1)
        for bx, by, bhx, bhy in boxes:
            free &= ~((np.abs(cand[:, 0] - bx) < hx + bhx) & (np.abs(cand[:, 1] - by) < hy + bhy))
        idx = np.where(free)[0]
        best = int(idx[np.argmin(cost[idx])]) if len(idx) else int(np.argmax(rr))
        it["pos"] = cand[best]
        boxes.append((cand[best][0], cand[best][1], hx, hy))


def _make_labels(site):
    """Residue labels placed for the CURRENT camera, free of overlaps."""
    name, o, atoms = site["name"], site["opts"], site["atoms"]
    obj, obj3, lead = name + ".labels", name + ".labels_lining", name + ".leaders"
    for n in (obj, obj3, lead):
        cmd.delete(n)
    site["label_boxes"], site["label_front"] = [], -1e9
    if not o["labels"]:
        return
    v = cmd.get_view()
    R = np.array(v[:9]).reshape(3, 3)
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
    lig = np.array([atoms[i].xyz for i in site["lig"]])
    c2 = np.array([float((lig @ X).mean()), float((lig @ Y).mean())])
    zc = float((lig @ Z).mean())
    drawn = list(site["lig"])
    for info in site["residues"].values():
        if info["tier"] and (info["tier"] < 3 or o["lining"]):
            drawn.extend(info["shown"])
    Pd = np.array([atoms[i].xyz for i in drawn])
    obstacles = np.column_stack([Pd @ X, Pd @ Y])
    extra = []
    for r in site["inter"]:             # keep labels off the interaction dashes too
        if r["draw"] and r["p_lig"] is not None and r["p_env"] is not None:
            for t in (0.2, 0.4, 0.6, 0.8):
                m = r["p_lig"] + t * (r["p_env"] - r["p_lig"])
                extra.append([float(m @ X), float(m @ Y)])
    if extra:
        obstacles = np.vstack([obstacles, np.array(extra)])
    z_front = float((Pd @ Z).max())
    size = float(o["label_size"])
    n_key = sum(1 for info in site["residues"].values() if info["label"] and info["tier"] in (1, 2))
    n_lin = sum(1 for info in site["residues"].values() if info["label"] and info["tier"] == 3)
    lining_labels = o["lining"] and (o["label_lining"] == "1" or
                                     (o["label_lining"] == "auto" and n_key + n_lin <= 18))
    n_all = n_key + (n_lin if lining_labels else 0)
    if n_all > 16:                      # keep a busy site legible
        size *= max(0.78, 1.0 - 0.012 * (n_all - 16))
    items = []
    for rkey, info in site["residues"].items():
        if not info["label"] or not info["shown"]:
            continue
        if info["tier"] == 3 and not lining_labels:
            continue
        own = [j for j in info["shown"] if atoms[j].rkey == rkey] or info["shown"]
        Q = np.array([atoms[j].xyz for j in own])
        q2 = np.column_stack([Q @ X, Q @ Y])
        anchor = q2.mean(0)
        u = anchor - c2
        n = float(np.linalg.norm(u))
        u = u / n if n > 0.4 else np.array([0.0, 1.0])
        s = size * (0.82 if info["tier"] == 3 else 1.0)
        half = np.array([0.5 * len(info["label"]) * CHAR_W * s, 0.5 * CHAR_H * s])
        items.append(dict(text=info["label"], half=half, anchor=anchor, out=u, prio=n, own=q2, own3=Q,
                          z=min(max(float((Q @ Z).mean()), zc - 2.5), zc + 2.5),
                          tier=info["tier"], scale=s))
    if o["ligand_label"]:
        text = o["ligand_label"]
        half = np.array([0.5 * len(text) * CHAR_W * size, 0.5 * CHAR_H * size])
        l2 = np.column_stack([lig @ X, lig @ Y])
        top = l2[int(np.argmax(l2[:, 1]))]
        items.append(dict(text=text, half=half, anchor=top, out=np.array([0.0, 1.0]), prio=-1.0,
                          own=l2, own3=lig, z=zc, tier=0, scale=size))
    if not items:
        return
    _place(items, obstacles)
    lift = z_front - min(it["z"] for it in items) + 1.5
    n_lead = 0
    for k, it in enumerate(items):
        p3 = it["pos"][0] * X + it["pos"][1] * Y + it["z"] * Z
        nm = "L%d" % (k + 1)
        target = obj3 if it["tier"] == 3 else obj
        cmd.pseudoatom(target, pos=[float(t) for t in p3], name=nm, resn="LBL", resi=str(k + 1),
                       chain="", segi="", label=it["text"], state=site["state"], quiet=1)
        _set("label_color", {0: "bs_ligand", 1: "bs_label", 2: "bs_label",
                             3: "bs_label_dim"}.get(it["tier"], "bs_label"),
             "%%%s and name %s" % (target, nm))
        hx, hy = it["half"]
        site["label_boxes"].append((it["pos"][0] - hx, it["pos"][0] + hx, it["pos"][1] - hy, it["pos"][1] + hy))
        # a leader line when the label had to move away from its residue
        edge = np.column_stack([np.clip(it["own"][:, 0], it["pos"][0] - hx, it["pos"][0] + hx),
                                np.clip(it["own"][:, 1], it["pos"][1] - hy, it["pos"][1] + hy)])
        gaps = np.linalg.norm(edge - it["own"], axis=1)
        j = int(np.argmin(gaps))
        if gaps[j] > 1.3 and it["tier"] != 0:
            n_lead += 1
            a3 = it["own3"][j]
            away = _unit(np.append(edge[j] - it["own"][j], 0.0))
            start = a3 + 0.35 * (away[0] * X + away[1] * Y)
            end = (edge[j][0] - 0.12 * away[0]) * X + (edge[j][1] - 0.12 * away[1]) * Y + float(a3 @ Z) * Z
            ends = name + ".leader_ends"
            for tag, p in (("A", start), ("B", end)):
                cmd.pseudoatom(ends, pos=[float(t) for t in p], name="%s%d" % (tag, n_lead), resn="LDR",
                               resi=str(n_lead), chain="", segi="", state=site["state"], quiet=1)
            cmd.distance(lead, "%%%s and name A%d" % (ends, n_lead), "%%%s and name B%d" % (ends, n_lead),
                         mode=0, reset=0, quiet=1, state=site["state"])
    site["label_size_used"] = size
    for target, s in ((obj, size), (obj3, 0.82 * size)):
        if target not in cmd.get_names("objects"):
            continue
        cmd.hide("everything", target)
        cmd.show("labels", target)
        _set("label_size", -s, target)
        _set("label_font_id", int(o["label_font"]), target)
        _set("label_position", (0.0, 0.0, lift + 1.0), target)
    if n_lead:
        cmd.delete(name + ".leader_ends")       # the measurement keeps its coordinates
        cmd.hide("labels", lead)
        cmd.color("bs_label_dim", lead)
        for key, val in (("dash_color", "bs_label_dim"), ("dash_gap", 0.0), ("dash_length", 0.5),
                         ("dash_radius", 0.012), ("dash_width", 1.0), ("dash_round_ends", 0)):
            _set(key, val, lead)
    site["label_front"] = max(it["z"] for it in items) + lift


def _make_legend(site):
    """Colour key for the interaction types present, pinned to the lower left."""
    name, o = site["name"], site["opts"]
    obj, guard = name + ".legend", "_%s_legend_end" % name
    cmd.delete(obj)
    cmd.delete(guard)
    site["legend_rows"] = 0
    if not o["legend"]:
        return
    present = []
    for key in ITYPES:
        drawn = [r for r in site["inter"] if r["type"] == key and r["draw"]]
        if key == "covalent":
            drawn = [r for r in drawn if r["note"]]
        if key == "hydrophobic" and not o["hydrophobic_lines"]:
            drawn = []
        if drawn:
            present.append(key)
    if not present:
        return
    # anchor the key just behind the front clipping plane: never clipped, never fogged
    v = cmd.get_view()
    Z = np.array(v[:9]).reshape(3, 3)[:, 2]
    pos = [float(t) for t in np.array(v[12:15]) + Z * (-v[11] - v[15] - 0.6)]
    vh = float(cmd.get_viewport()[1]) or 750.0
    px = max(11.0, 0.021 * vh)
    step = 2.0 * 1.6 * px / vh
    for k, key in enumerate(reversed(present)):
        text = ITYPES[key]["label"].replace("pi-stacking", u"\u03c0\u2013\u03c0 stacking")
        text = text.replace("Cation-pi", u"Cation\u2013\u03c0")
        cmd.pseudoatom(obj, pos=pos, name="E%d" % (k + 1), resn="KEY", resi=str(k + 1), chain="",
                       segi="", label=u"\u2013 \u2013  " + text, state=site["state"], quiet=1)
        sel = "%%%s and name E%d" % (obj, k + 1)
        _set("label_screen_point", (-0.965, -0.945 + step * k, 0.0), sel)
        _set("label_color", "bs_" + key, sel)
    cmd.hide("everything", obj)
    cmd.show("labels", obj)
    _set("label_relative_mode", 1, obj)
    _set("label_position", (1.0, 0.0, 0.0), obj)
    _set("label_size", px, obj)
    _set("label_font_id", 7, obj)
    # PyMOL keeps the screen position of the last pinned label and applies it to
    # the measurement labels drawn next (also in the next frame).  An ordinary,
    # blank label rendered right after the key resets that state.
    cmd.pseudoatom(guard, pos=pos, name="Z", resn="KEY", resi="0", chain="", segi="", label=" ",
                   state=site["state"], quiet=1)
    cmd.hide("everything", guard)
    cmd.show("labels", guard)
    site["legend_rows"] = len(present)
    site["legend_chars"] = max(len(ITYPES[k]["label"]) for k in present) + 5


def _cut_surface(site):
    """Show the pocket surface only behind the ligand, so it never veils it."""
    sobj = site["name"] + ".surface"
    if sobj not in cmd.get_names("objects"):
        return
    atoms = site["atoms"]
    Z = np.array(cmd.get_view()[:9]).reshape(3, 3)[:, 2]
    zl = np.array([atoms[i].xyz for i in site["lig"]]) @ Z
    rows = []
    cmd.iterate_state(1, sobj, "rows.append((index, x, y, z))", space={"rows": rows})
    if not rows:
        return
    idx = np.array([r[0] for r in rows])
    z = np.array([r[1:] for r in rows]) @ Z
    keep = idx[z <= float(zl.mean()) + 0.5]
    cmd.hide("surface", sobj)
    if len(keep):
        r = _ranges(keep.tolist())
        for k in range(0, len(r), 200):
            cmd.show("surface", "%%%s and index %s" % (sobj, "+".join(r[k:k + 200])))


def _cut_cartoon(site):
    """Hide the stretch of cartoon that passes between the camera and the site,
    so that the ribbon is a backdrop and never a veil."""
    o, trace = site["opts"], site.get("trace")
    if not (o["cartoon"] and o["restyle"] and o["cutaway"]) or not trace:
        return
    atoms = site["atoms"]
    R = np.array(cmd.get_view()[:9]).reshape(3, 3)
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
    lig = np.array([atoms[i].xyz for i in site["lig"]])
    pts = [lig] + [np.array([atoms[j].xyz for j in info["shown"]])
                   for info in site["residues"].values() if info["tier"] and info["shown"]]
    pts = np.vstack(pts)
    xs, ys = list(pts @ X), list(pts @ Y)
    for bx0, bx1, by0, by1 in site.get("label_boxes", []):
        xs += [bx0, bx1]
        ys += [by0, by1]
    zl = lig @ Z
    z_cut = 0.5 * (float(zl.min()) + float(zl.mean()))
    P = np.array([t[2:] for t in trace])
    x, y, z = P @ X, P @ Y, P @ Z
    front = (z > z_cut) & (x > min(xs) - 2.5) & (x < max(xs) + 2.5) & (y > min(ys) - 2.5) & (y < max(ys) + 2.5)
    for obj in OrderedDict.fromkeys(t[0] for t in trace):
        cmd.show("cartoon", "%%%s and polymer" % obj)
        ids = [trace[k][1] for k in np.where(front)[0] if trace[k][0] == obj]
        r = _ranges(ids)
        for k in range(0, len(r), 200):
            cmd.hide("cartoon", "byres (%%%s and index %s)" % (obj, "+".join(r[k:k + 200])))


def _for_this_view(site, fit=True):
    """Everything that depends on the camera: labels, frame, cut-aways, key."""
    name = site["name"]
    _make_labels(site)
    _make_legend(site)
    if fit:
        _fit(site)
        _make_legend(site)          # again: its anchor follows the clipping plane
    _cut_surface(site)
    _cut_cartoon(site)
    have = set(cmd.get_names("objects"))
    members = [n for n in _site_names(name) if n in have]
    if members:
        cmd.group(name, " ".join(members))
    cmd.deselect()


# =============================================================================
#  drawing
# =============================================================================
def _site_names(name):
    """Every object and selection a site called `name` can own, and nothing else."""
    order = list(ITYPES) + ["centers", "surface", "labels_lining", "labels", "leaders",
                            "leader_ends", "legend"]
    names = ["%s.%s" % (name, suffix) for suffix in order]
    names.append("_%s_legend_end" % name)      # must stay right behind the legend
    names += ["%s_%s" % (name, suffix) for suffix in ("ligand", "pocket", "interacting")]
    return names


def _delete_site(name):
    have = set(cmd.get_names("all"))
    for n in _site_names(name):
        if n in have:
            cmd.delete(n)
    if name in have and cmd.get_type(name) == "object:group":
        try:
            cmd.group(name, action="empty")     # anything else in there is not ours to delete
        except Exception:
            pass
        cmd.delete(name)


def _get_site(name, who):
    """The stored site, unless the session has been reset under it."""
    name = str(_unquote(name)).strip()
    site = _SITES.get(name)
    if site is not None and name + "_ligand" not in cmd.get_names("selections"):
        _SITES.pop(name, None)
        site = None
    if site is None:
        raise CmdException("%s: run bindingsite first (there is no site named '%s')" % (who, name))
    return site


def _apply_style(style_name, quiet=True):
    style = STYLES[style_name]
    for key, val in style["settings"].items():
        if key not in _SAVED:
            try:
                _SAVED[key] = cmd.get(key)
            except Exception:
                continue
        _set(key, val)
    if "bg_rgb" not in _SAVED:
        _SAVED["bg_rgb"] = cmd.get("bg_rgb")
    _color("bs_bg", style["bg"])
    cmd.bg_color("bs_bg")
    for key in ("ligand", "residue", "het", "cartoon", "surface", "label", "label_dim"):
        _color("bs_" + key, style[key])
    for key, spec in ITYPES.items():
        rgb = spec["rgb"]
        if style_name == "dark":        # lift the dark hues so they read on black
            rgb = tuple(min(255, int(40 + 0.9 * c)) for c in rgb)
            if key == "covalent":
                rgb = (235, 235, 235)
        _color("bs_" + key, rgb)


def _draw(site):
    name, o, atoms, C = site["name"], site["opts"], site["atoms"], site["chem"]
    style = STYLES[o["style"]]
    st = site["state"]
    objs = list(OrderedDict.fromkeys(a.obj for a in atoms))
    site["objs"] = objs
    lig_sel = _isel(atoms, [a.i for a in atoms if a.lig])
    cmd.select(name + "_ligand", lig_sel, enable=0, quiet=1)
    ligand = name + "_ligand"

    # ---- base representation ------------------------------------------------
    if o["restyle"]:
        for obj in objs:
            cmd.hide("everything", "%" + obj)
            _set("valence", 0, obj)
            _set("cartoon_side_chain_helper", 0, obj)
            if o["cartoon"]:
                cmd.show("cartoon", "%%%s and polymer" % obj)
                _set("cartoon_transparency", float(o["cartoon_transparency"])
                     if o["cartoon_transparency"] >= 0 else style["cartoon_transparency"], obj)
                if o["recolor"]:
                    cmd.color("bs_cartoon", "%%%s and polymer" % obj)

    # ---- residues -------------------------------------------------------------
    radii = {1: 0.17, 2: 0.13, 3: 0.075}
    tier_sel = {}
    for tier in (3, 2, 1):
        idx = [j for info in site["residues"].values() if info["tier"] == tier
               and info["cls"] in ("protein", "het") for j in info["shown"]]
        if tier == 3 and not o["lining"]:
            idx = []
        tier_sel[tier] = idx
        if not idx:
            continue
        sel = _isel(atoms, idx)
        cmd.show("sticks", sel)
        cmd.set_bond("stick_radius", radii[tier], sel)
        if o["recolor"]:
            cmd.color("bs_residue", "(%s) and elem C" % sel)
            cmd.color("atomic", "(%s) and not elem C" % sel)
    het = [j for info in site["residues"].values() if info["tier"] and info["cls"] == "het"
           for j in info["shown"]]
    if het and o["recolor"]:
        cmd.color("bs_het", "(%s) and elem C" % _isel(atoms, het))
    pocket = [j for info in site["residues"].values() if info["tier"] for j in info["atoms"]]
    cmd.select(name + "_pocket", "byres (%s)" % _isel(atoms, pocket) if pocket else "none",
               enable=0, quiet=1)
    key = [j for info in site["residues"].values() if info["tier"] in (1, 2) for j in info["atoms"]]
    cmd.select(name + "_interacting", "byres (%s)" % _isel(atoms, key) if key else "none",
               enable=0, quiet=1)
    waters = [info["atoms"][0] for info in site["residues"].values()
              if info["cls"] == "water" and info["tier"]]
    if waters:
        sel = _isel(atoms, waters)
        cmd.show("spheres", sel)
        _set("sphere_scale", 0.22, sel)
        cmd.color("bs_waterbridge", sel)
    metals = [j for j in C.metals if (atoms[j].lig or site["residues"].get(atoms[j].rkey, {}).get("tier"))]
    if metals:
        sel = _isel(atoms, metals)
        cmd.show("spheres", sel)
        _set("sphere_scale", 0.45, sel)

    # ---- ligand -----------------------------------------------------------------
    cmd.show("sticks", ligand)
    if o["ligand_style"] == "ballstick":
        cmd.show("spheres", ligand)
        _set("sphere_scale", 0.24, ligand)
        _set("sphere_scale", 0.15, "%s and hydro" % ligand)
        cmd.set_bond("stick_radius", 0.14, ligand)
    else:
        cmd.set_bond("stick_radius", 0.23, ligand)
    lig_c = o["ligand_color"] if o["ligand_color"] != "auto" else "bs_ligand"
    cmd.color(lig_c, "%s and elem C" % ligand)
    cmd.color("atomic", "%s and not elem C" % ligand)
    shown_all = "%s or %s" % (ligand, _isel(atoms, [j for t in tier_sel.values() for j in t]))
    cmd.hide("everything", "(%s) and hydro and (neighbor elem C)" % shown_all)
    cmd.show("sticks", "hydro and (neighbor ((%s) and elem N+O+S))" % shown_all)

    # ---- interactions -------------------------------------------------------------
    centers = name + ".centers"
    n_center = [0]

    def endpoint(atom_idx, xyz):
        """(selection, state) of one end of a dash; ring centres get a pseudoatom."""
        if atom_idx is not None:
            return _isel(atoms, [atom_idx]), _object_state(atoms[atom_idx].obj, st)
        n_center[0] += 1
        nm = "C%d" % n_center[0]
        cmd.pseudoatom(centers, pos=[float(t) for t in xyz], name=nm, resn="CEN",
                       resi=str(n_center[0]), chain="", segi="", state=st, quiet=1)
        return "%%%s and name %s" % (centers, nm), st

    def dash(dobj, a, b):
        try:
            cmd.distance(dobj, a[0], b[0], mode=0, reset=0, quiet=1, state=st, state1=a[1], state2=b[1])
        except TypeError:               # a PyMOL without per-selection states
            cmd.distance(dobj, a[0], b[0], mode=0, reset=0, quiet=1, state=st)

    made = []
    for r in site["inter"]:
        if not r["draw"] or r["type"] not in ITYPES:
            continue
        if r["type"] == "hydrophobic" and not o["hydrophobic_lines"]:
            continue
        if r["res"] in site["residues"] and not site["residues"][r["res"]]["tier"]:
            continue
        dobj = "%s.%s" % (name, r["type"])
        if r["type"] == "covalent" and not r["note"]:
            continue                    # a real bond: the sticks already show it
        if r.get("water") is not None:
            w = endpoint(r["water"], None)
            dash(dobj, endpoint(r["la"], r["p_lig"]), w)
            dash(dobj, w, endpoint(r["ea"], r["p_env"]))
        else:
            dash(dobj, endpoint(r["la"], r["p_lig"]), endpoint(r["ea"], r["p_env"]))
        if dobj not in made:
            made.append(dobj)
    for dobj in made:
        spec = ITYPES[dobj.split(".")[-1]]
        gap, length, radius = spec["dash"]
        cmd.color("bs_" + dobj.split(".")[-1], dobj)
        _set("dash_color", "bs_" + dobj.split(".")[-1], dobj)
        _set("dash_gap", gap, dobj)
        _set("dash_length", length, dobj)
        _set("dash_radius", radius, dobj)
        _set("dash_width", max(1.5, radius * 55.0), dobj)
        if o["distances"] and dobj.split(".")[-1] != "hydrophobic":
            _set("label_size", -0.62 * float(o["label_size"]), dobj)
            _set("label_color", "bs_" + dobj.split(".")[-1], dobj)
            _set("label_font_id", int(o["label_font"]), dobj)
            _set("label_distance_digits", 1, dobj)
            _set("label_position", (0.0, 1.4, 4.0), dobj)
        else:
            cmd.hide("labels", dobj)
    if n_center[0]:
        cmd.hide("everything", centers)
        cmd.show("spheres", centers)
        _set("sphere_scale", 0.12, centers)
        _set("sphere_transparency", 0.25, centers)
        cmd.color("bs_pistack", centers)

    # ---- pocket surface ---------------------------------------------------------
    if o["surface"] != "none" and site.get("surface"):
        sobj = name + ".surface"
        src, idx = site["surface"]
        near = " or ".join("index " + "+".join(r) for r in
                           (_ranges(idx.tolist())[k:k + 200] for k in range(0, len(idx), 200)) if r)
        cmd.create(sobj, "(byres (%%%s and (%s))) and polymer and not %s" % (src, near, ligand),
                   _object_state(src, st), 1, quiet=1)
        cmd.hide("everything", sobj)
        cmd.show("surface", sobj)
        cmd.color("bs_surface", sobj)
        if o["surface"] == "polar":
            cmd.color("bs_surface", "%s and elem C" % sobj)
            _color("bs_surf_n", (0.62, 0.72, 0.95))
            _color("bs_surf_o", (0.96, 0.66, 0.64))
            _color("bs_surf_s", (0.93, 0.86, 0.55))
            cmd.color("bs_surf_n", "%s and elem N" % sobj)
            cmd.color("bs_surf_o", "%s and elem O" % sobj)
            cmd.color("bs_surf_s", "%s and elem S" % sobj)
        _set("surface_carve_selection", ligand, sobj)
        _set("surface_carve_cutoff", float(o["surface_cutoff"]), sobj)
        _set("transparency", float(o["surface_transparency"]), sobj)
        _set("surface_quality", 1, sobj)

    # ---- camera, labels -----------------------------------------------------------
    if o["view"]:
        views = site.get("views") or _rank_views(site)
        site["views"] = views
        k = min(int(o.get("view_index", 0)), len(views) - 1)
        _set_camera(*_camera(site, views[k][1]))
    _for_this_view(site, fit=o["view"])


# =============================================================================
#  report
# =============================================================================
def _rows(site):
    atoms, o = site["atoms"], site["opts"]
    rows = []
    for r in site["inter"]:
        if r["type"] == "water":
            continue
        info = site["residues"].get(r["res"]) if r["res"] else None
        if info is not None:
            res = _res_label(atoms, info["atoms"], o["label_format"], o["offset"], False)
            chain = r["res"][2]
        else:
            res, chain = "-", ""
        if r.get("shell"):
            la = "(%s)" % atoms[r["la"]].elem
        elif r["la"] is not None:
            la = atoms[r["la"]].name
        else:
            names = [atoms[i].name for i in r.get("lig_atoms", ())]
            ring_first = r["type"] == "pistack" or r["note"].startswith("ligand ring")
            la = ("ring " if ring_first else "") + "-".join(names[:6])
        if r["ea"] is not None:
            ea = atoms[r["ea"]].name
        else:
            ea = "ring" if "ring" in r["note"].split("...")[-1] or r["type"] == "pistack" else "centre"
        rows.append(dict(type=ITYPES[r["type"]]["label"], residue=res, chain=chain, res_atom=ea,
                         lig_atom=la, dist=r["dist"], angle=r["angle"], note=r["note"],
                         drawn=bool(r["draw"])))
    return rows


def _print_report(site):
    rows = _rows(site)
    atoms = site["atoms"]
    lig = [atoms[i] for i in site["lig"]]
    resn = "+".join(OrderedDict.fromkeys(a.resn for a in lig))
    print("")
    print(" Binding site of %s (%d heavy atoms) in %s" % (resn, len(lig), lig[0].obj))
    off = site["opts"]["offset"]
    if off:
        print(" Residue numbers shifted by %s for display." % (off,))
    if not rows:
        print("   no interactions detected")
    else:
        fmt = "   %-20s %-14s %-9s %-24s %6s %6s  %s"
        print(fmt % ("Interaction", "Residue", "Res.atom", "Ligand atom", "d (A)", "angle", "note"))
        print("   " + "-" * 100)
        clash_label, n_shown = ITYPES["clash"]["label"], 0
        for r in sorted(rows, key=lambda r: r["dist"] if r["type"] == clash_label else 0.0):
            if r["type"] == clash_label:
                n_shown += 1
                if n_shown > 8:
                    continue
            print(fmt % (r["type"], r["residue"] + (" " + r["chain"] if r["chain"] else ""),
                         r["res_atom"], r["lig_atom"][:24], "%.2f" % r["dist"],
                         "%.0f" % r["angle"] if r["angle"] is not None else "",
                         r["note"] + ("" if r["drawn"] else "  [not drawn]")))
        if n_shown > 8:
            print("   ... and %d more clashes (bindingsite_report file.csv lists them all)" % (n_shown - 8))
    near = [info["label"] or _res_label(atoms, info["atoms"], site["opts"]["label_format"], off, False)
            for info in site["residues"].values() if info["tier"] == 3]
    if near:
        print("   Lining the pocket without a specific contact (<= %.1f A): %s"
              % (site["opts"]["radius"], ", ".join(near)))
    n_clash = sum(1 for r in rows if r["type"] == ITYPES["clash"]["label"])
    if n_clash:
        print("   WARNING: %d steric clash%s between ligand and pocket: check this pose."
              % (n_clash, "" if n_clash == 1 else "es"))
    for note in site["chem"].notes:
        print("   note: " + note)
    print("")


# =============================================================================
#  PyMOL commands
# =============================================================================
def bindingsite(ligand="auto", receptor="auto", radius=4.0, style="publication",
                labels=1, distances=0, surface="none", lining=1, hydrophobic=1,
                offset=0, label_format="{resn}{resi}", label_size=0.7, ligand_label="",
                ligand_color="auto", ligand_style="sticks", cartoon=1,
                cartoon_transparency=-1, surface_transparency=0.35, surface_cutoff=5.0,
                interactions="all", his_positive=0, view=1, restyle=1, recolor=1,
                label_lining="auto", legend=1, cutaway=1, fix_bonds=1, scene=1, state=-1,
                name="bs", quiet=0, _self=cmd):
    """
DESCRIPTION

    Draw a finished binding-site view: ligand, pocket residues, colour-coded
    interactions, residue labels that do not overlap and a camera that looks
    into the pocket.

USAGE

    bindingsite [ligand [, receptor [, radius [, style ]]]] [, option=value ...]

ARGUMENTS

    ligand = selection of the ligand, or "auto" (default) to pick the most
             buried non-additive organic molecule in the enabled objects
    receptor = everything the ligand may interact with.  Default: the rest of
             the ligand's own object (protein, waters, ions, cofactors), so
             that other complexes loaded alongside stay out of the picture.
             A ligand that is alone in its object (docking poses) takes
             the other enabled objects instead.
    radius = also show residues within this distance of the ligand that make
             no specific contact, as thin sticks {default: 4.0}
    style  = publication | soft | flat | dark  {default: publication}

OPTIONS

    labels = 0/1         residue labels {1}
    legend = 0/1         colour key in the lower-left corner {1}
    distances = 0/1      print the distance on every interaction {0}
    surface = none | pocket | polar
                         semi-transparent pocket surface behind the ligand;
                         "polar" tints N blue and O red {none}
    lining = 0/1         show the non-interacting lining residues {1}
    label_lining = auto | 0 | 1
                         label them too; "auto" drops these labels when the
                         site gets crowded {auto}
    hydrophobic = 0/1    draw hydrophobic contacts (one per residue) {1}
    offset = int | "A:27,B:-21"
                         add this to the residue numbers in labels and reports,
                         e.g. to print precursor numbering on a mature-chain
                         model; the structure itself is not renumbered {0}
    label_format         {resn}{resi} -> Ser174 ; {r}{resi} -> S174 ;
                         also {RESN} {chain} {segi} {obj}
    label_size           label height in Angstrom (scales with the zoom) {0.7}
    ligand_label         text placed next to the ligand {none}
    ligand_color         any PyMOL colour for the ligand carbons {style default}
    ligand_style         sticks | ballstick {sticks}
    cartoon = 0/1        keep the cartoon as a faint backdrop {1}
    cutaway = 0/1        hide the stretch of cartoon in front of the site {1}
    cartoon_transparency, surface_transparency, surface_cutoff
    interactions         "all" or a list such as "hbond,saltbridge,pistack";
                         types: hbond saltbridge pistack pication halogen metal
                         waterbridge hydrophobic covalent nearattack clash
    his_positive = 0/1   treat histidines of unknown protonation as cationic,
                         as at lysosomal / acidic pH {0}
    view = 0/1           choose the camera automatically {1}; 0 keeps yours
    restyle = 0/1        0 leaves the existing representation alone and only
                         adds the site on top {1}
    recolor = 0/1        0 keeps your atom colours {1}
    fix_bonds = 0/1      remove ligand-protein bonds that PyMOL guessed from
                         distance but that are too long/short to be real {1}
    scene = 0/1          store the result as a scene called <name> {1}
    state                state to analyse {current}
    name                 prefix of everything created {bs}

NOTES

    Hydrogens are not required.  Donors, acceptors, charges, aromatic rings and
    electrophilic warheads are perceived from geometry; where the file has
    explicit hydrogens they are used instead.  Amines, amidines and guanidines
    are taken as protonated, carboxylates/phosphates/sulfonates/tetrazoles as
    deprotonated.  Steric clashes are drawn and reported, so that a pose that
    clashes does not pass unnoticed.

    Labels, the colour key and the cut-aways are laid out for the camera at
    the time of the call.  After rotating, run  bindingsite_relabel .

RESULT

    Objects  <name>.hbond, <name>.saltbridge, ... (toggle them in the panel),
             <name>.labels, <name>.legend, optionally <name>.surface
    Selections <name>_ligand, <name>_pocket, <name>_interacting
    A scene called <name>.

SEE ALSO

    bindingsite_report, bindingsite_png, bindingsite_relabel, bindingsite_view,
    bindingsite_ligands, bindingsite_all, bindingsite_clear, bindingsite_param,
    bindingsite_gui
    """
    quiet = int(quiet)
    ligand, receptor = _unquote(ligand), _unquote(receptor)
    style = str(_unquote(style)).strip().lower()
    if style not in STYLES:
        raise CmdException("bindingsite: unknown style '%s' (choose from %s)" % (style, ", ".join(STYLES)))
    interactions = str(_unquote(interactions)).strip().lower()
    if interactions in ("all", ""):
        types = set(ITYPES)
    else:
        types = set(t for t in re.split(r"[,+ ]+", interactions) if t)
        bad = types - set(ITYPES)
        if bad:
            raise CmdException("bindingsite: unknown interaction type(s): %s" % ", ".join(sorted(bad)))
    if not _truthy(hydrophobic):
        types.discard("hydrophobic")
    surface = str(_unquote(surface)).strip().lower()
    if surface in ("", "0", "none", "off", "no", "false"):
        surface = "none"
    elif surface in ("1", "yes", "on", "true"):
        surface = "pocket"
    elif surface not in ("pocket", "polar"):
        raise CmdException("bindingsite: surface must be none, pocket or polar")
    ligand_color = str(_unquote(ligand_color)).strip()
    if ligand_color.lower() in ("", "auto"):
        ligand_color = "auto"
    elif cmd.get_color_index(ligand_color) == -1:
        raise CmdException("bindingsite: '%s' is not a PyMOL colour" % ligand_color)
    ligand_style = str(_unquote(ligand_style)).strip().lower()
    if ligand_style not in ("sticks", "ballstick"):
        raise CmdException("bindingsite: ligand_style must be sticks or ballstick")
    label_format = str(_unquote(label_format))
    try:
        label_format.format(resn="Ser", RESN="SER", r="S", resi="1", chain="A", segi="", obj="")
    except (KeyError, IndexError, ValueError):
        raise CmdException("bindingsite: label_format '%s' is not valid; the fields are {resn} {RESN} "
                           "{r} {resi} {chain} {segi} {obj}" % label_format)
    name = str(_unquote(name)).strip()
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise CmdException("bindingsite: name must consist of letters, digits and underscores")
    if name in cmd.get_names("all"):
        ours = name in _SITES or name + "_ligand" in cmd.get_names("selections")
        if cmd.get_type(name) != "object:group" or not ours:
            raise CmdException("bindingsite: '%s' is already the name of an object, selection or group "
                               "in this session; pass another name=" % name)
    state = int(state)
    opts = dict(radius=float(radius), style=style, labels=_truthy(labels), distances=_truthy(distances),
                surface=surface, lining=_truthy(lining),
                hydrophobic_lines=_truthy(hydrophobic), offset=_parse_offset(offset),
                label_format=label_format, label_size=float(label_size),
                ligand_label=str(_unquote(ligand_label)), ligand_color=ligand_color,
                ligand_style=ligand_style, cartoon=_truthy(cartoon),
                cartoon_transparency=float(cartoon_transparency),
                surface_transparency=float(surface_transparency), surface_cutoff=float(surface_cutoff),
                types=types, his_positive=_truthy(his_positive), view=_truthy(view),
                restyle=_truthy(restyle), recolor=_truthy(recolor), label_font=7, quiet=quiet,
                view_index=0, fix_bonds=_truthy(fix_bonds), cutaway=_truthy(cutaway),
                label_lining="auto" if str(label_lining).strip().lower() == "auto"
                else ("1" if _truthy(label_lining) else "0"),
                legend=_truthy(legend))

    lig_sel = str(ligand).strip()
    others = []
    if lig_sel.lower() in ("auto", ""):
        cands = find_ligands("enabled", state)
        if not cands:
            raise CmdException("bindingsite: no ligand found in the enabled objects. "
                               "Name one, e.g.  bindingsite resn LIG  or  bindingsite chain B")
        lig_sel = cands[0]["sel"]
        others = cands[1:]
        _say(quiet, "ligand = %s  (%d heavy atoms, %s)" % (cands[0]["pretty"], cands[0]["natoms"], cands[0]["kind"]))

    site = _analyse(lig_sel, receptor, state, opts)
    site["name"] = name
    _delete_site(name)
    _SITES.pop(name, None)
    _apply_style(style)
    cmd.set("suspend_updates", 1, quiet=1)
    try:
        _draw(site)
    finally:
        cmd.set("suspend_updates", 0, quiet=1)
    _SITES[name] = site
    if _truthy(scene):
        try:
            cmd.scene(name, "store", quiet=1)
        except Exception:
            pass
    if not quiet:
        _print_report(site)
        real = [c for c in others if c["kind"] in ("ligand", "peptide")]
        if real:
            print(" bindingsite: %d more ligand%s here, e.g.  bindingsite %s"
                  % (len(real), "" if len(real) == 1 else "s", real[0]["pretty"]))
    return site


def bindingsite_ligands(selection="enabled", state=-1, quiet=0, _self=cmd):
    """
DESCRIPTION

    List the ligands that `bindingsite` can find, best candidate first.

USAGE

    bindingsite_ligands [selection]
    """
    cands = find_ligands(str(_unquote(selection)), int(state))
    if not int(quiet):
        if not cands:
            print(" bindingsite: no ligands found in '%s'" % selection)
        for k, c in enumerate(cands):
            print(" %2d. %-9s %-12s %3d atoms, %3d protein atoms within 4.5 A   ->  bindingsite %s"
                  % (k + 1, c["kind"], c["resn"][:12], c["natoms"], c["contacts"], c["pretty"]))
    return cands


def bindingsite_report(filename="", name="bs", quiet=0, _self=cmd):
    """
DESCRIPTION

    Print the interaction table of the last `bindingsite` call, and optionally
    write it to a .csv / .tsv / .md file.

USAGE

    bindingsite_report [filename [, name ]]
    """
    site = _get_site(name, "bindingsite_report")
    filename = str(_unquote(filename))
    if not int(quiet):
        _print_report(site)
    rows = _rows(site)
    if filename:
        cols = ("type", "residue", "chain", "res_atom", "lig_atom", "dist", "angle", "note", "drawn")
        ext = os.path.splitext(filename)[1].lower()

        def cell(r, c):
            v = r[c]
            if v is None:
                return ""
            if isinstance(v, float):
                return "%.2f" % v if c == "dist" else "%.0f" % v
            return str(v)

        with open(os.path.expanduser(filename), "w") as fh:
            if ext == ".md":
                fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
                for r in rows:
                    fh.write("| " + " | ".join(cell(r, c) for c in cols) + " |\n")
            else:
                sep = "\t" if ext in (".tsv", ".txt") else ","
                fh.write(sep.join(cols) + "\n")
                for r in rows:
                    fh.write(sep.join('"%s"' % cell(r, c) if sep in cell(r, c) else cell(r, c)
                                      for c in cols) + "\n")
        _say(int(quiet), "wrote %s" % filename)
    return rows


def bindingsite_relabel(name="bs", fit=0, _self=cmd):
    """
DESCRIPTION

    Re-place the residue labels for the camera as it is now.  Rotate the scene
    to taste, then run this to get non-overlapping labels for that view.

USAGE

    bindingsite_relabel [name [, fit ]]

    fit = 1 also re-centres and re-zooms on the site.
    """
    _for_this_view(_get_site(name, "bindingsite_relabel"), fit=_truthy(fit))


def bindingsite_view(index="next", name="bs", _self=cmd):
    """
DESCRIPTION

    Switch between the automatically ranked camera directions.  The first is
    the default; "next" cycles through the alternatives.

USAGE

    bindingsite_view [index | next [, name ]]
    """
    site = _get_site(name, "bindingsite_view")
    views = site.get("views") or _rank_views(site)
    site["views"] = views
    cur = site["opts"].get("view_index", 0)
    k = (cur + 1) % len(views) if str(index).strip().lower() == "next" else (int(index) - 1) % len(views)
    site["opts"]["view_index"] = k
    _set_camera(*_camera(site, views[k][1]))
    _for_this_view(site, fit=True)
    print(" bindingsite: view %d of %d" % (k + 1, len(views)))


# =============================================================================
#  pictures larger than the window
# =============================================================================
# PyMOL's ray-tracer enlarges labels sized in pixels when the picture is larger
# than the window, but not labels sized in Angstrom: TextRenderRay() multiplies
# by the magnification only where label_size >= 0 (the same in PyMOL 2.5, 3.1
# and the current source).  The labels drawn here are sized in Angstrom, so that
# they keep their proportions when the view is zoomed.  Left alone they are
# traced at their on-screen pixel size, which on a 4000-pixel picture is too
# small to read.  Their size is therefore corrected while a picture is traced:
# by bindingsite_png, and by PyMOL's own `ray` and `png`, which are wrapped
# below for this purpose alone.
_STOCK = {}                 # PyMOL's own ray and png
_tracing = []               # not empty while a corrected picture is being traced


def _angstrom_labels(everything=False):
    """(object, label_size) of the objects labelled in Angstrom.

    Only those of binding-site scenes, recognised by name so that a reloaded
    session works too; with `everything`, any object with a size of its own
    and the global setting ("") as well.
    """
    found = []
    ours = set(ITYPES) | {"labels", "labels_lining"}
    try:
        common = float(cmd.get("label_size"))
    except Exception:
        common = 0.0
    if everything and common < 0:
        found.append(("", common))
    for obj in cmd.get_names("objects"):
        base, _, suffix = obj.rpartition(".")
        mine = bool(base) and suffix in ours
        if not (mine or everything):
            continue
        try:
            size = float(cmd.get("label_size", obj))
        except Exception:
            continue
        if size < 0 and (mine or abs(size - common) > 1e-6):
            found.append((obj, size))
    return found


@contextlib.contextmanager
def _labels_for_picture(height, everything=False):
    """Size the Angstrom labels for a ray-traced picture `height` pixels high."""
    touched, scale = [], 1.0
    if not _tracing:
        scale = float(height) / float(cmd.get_viewport()[1])
        if abs(scale - 1.0) > 1e-6:
            touched = _angstrom_labels(everything)
    _tracing.append(height)
    try:
        # updates=0 on the way in and out: the tracer reads the size as it runs,
        # and nothing on screen is invalidated, so the picture that `ray` leaves
        # in the window is still there for a following `png`.
        for obj, size in touched:
            cmd.set("label_size", size * scale, obj, updates=0, quiet=1)
        yield
    finally:
        for obj, size in touched:
            cmd.set("label_size", size, obj, updates=0, quiet=1)
        _tracing.pop()


def _pixels(value, dpi):
    """A `png` size in pixels; PyMOL also takes `10cm`, `4in` or `80mm` with a dpi."""
    text = str(value).strip().lower()
    m = re.match(r"^([0-9.]+)\s*(in|mm|cm)$", text)
    if not m:
        return int(float(text))
    dpi = float(dpi)
    if dpi < 0:
        dpi = float(cmd.get("image_dots_per_inch"))
    return int(float(m.group(1)) * dpi / {"in": 1.0, "mm": 25.4, "cm": 2.54}[m.group(2)] + 0.5)


def _to_correct(width, height):
    """(width, height) of a picture whose labels need correcting, else None."""
    try:
        if _tracing:
            return None
        vw, vh = cmd.get_viewport()
        w, h = max(0, int(width)), max(0, int(height))
        if not (w or h):
            return None
        h = h or (w * vh) // vw         # a missing side follows the window, as in PyMOL
        w = w or (h * vw) // vh
        if h == vh or not _angstrom_labels():
            return None
        return w, h
    except Exception:
        return None


def _sides_cut(size, quiet):
    """PyMOL keeps the height of the view; a narrower picture loses its sides."""
    try:
        vw, vh = cmd.get_viewport()
        if not int(quiet) and size[0] * vh < 0.98 * size[1] * vw:
            print(" bindingsite: this picture is narrower than the window, so its sides are cut off."
                  "  `bindingsite_png file.png, %d, %d` frames the site for the picture." % size)
    except Exception:
        pass


def _ray(width=0, height=0, antialias=-1, angle=0.0, shift=0.0, renderer=-1, quiet=1,
         async_=0, _self=cmd, **kwargs):
    stock = _STOCK["ray"]
    size = _to_correct(width, height) if _self is cmd else None
    if size is None:
        return stock(width, height, antialias, angle, shift, renderer, quiet, async_,
                     _self=_self, **kwargs)
    kwargs.pop("async", None)           # the labels are put back once the picture is done
    with _labels_for_picture(size[1]):
        result = stock(width, height, antialias, angle, shift, renderer, quiet, 0,
                       _self=_self, **kwargs)
    _sides_cut(size, quiet)
    return result


def _png(filename, width=0, height=0, dpi=-1.0, ray=0, quiet=1, prior=0, format=0, *, _self=cmd):
    stock = _STOCK["png"]
    size = None
    try:
        # without OpenGL (pymol -c) PyMOL ray-traces every picture of a given size
        traced = int(ray) or not any(cmd.get_renderer())
        if _self is cmd and traced and int(prior) <= 0:
            size = _to_correct(_pixels(width, dpi), _pixels(height, dpi))
    except Exception:
        size = None
    if size is None:
        return stock(filename, width, height, dpi, ray, quiet, prior, format, _self=_self)
    with _labels_for_picture(size[1]):
        result = stock(filename, width, height, dpi, ray, quiet, prior, format, _self=_self)
    _sides_cut(size, quiet)
    return result


def _wrap_ray_and_png():
    """Put the label correction around PyMOL's `ray` and `png`.

    A PyMOL whose commands do not have the arguments known here is left as it is.
    """
    import inspect
    known = {"ray": ["width", "height", "antialias", "angle", "shift", "renderer", "quiet",
                     "async_", "_self", "kwargs"],
             "png": ["filename", "width", "height", "dpi", "ray", "quiet", "prior", "format",
                     "_self"]}
    for name, wrapper in (("ray", _ray), ("png", _png)):
        current = getattr(cmd, name, None)
        stock = getattr(current, "_bindingsite_stock", current)    # the plugin may be loaded twice
        try:
            if list(inspect.signature(stock).parameters) != known[name]:
                continue
        except Exception:
            continue
        _STOCK[name] = stock
        wrapper.__doc__ = stock.__doc__
        wrapper._bindingsite_stock = stock
        setattr(cmd, name, wrapper)
        entry = cmd.keyword.get(name)
        if entry and entry[0] in (stock, current):
            entry[0] = wrapper


def bindingsite_png(filename, width=2400, height=0, dpi=300, name="bs", transparent=0,
                    fit=1, quiet=0, _self=cmd):
    """
DESCRIPTION

    Ray-trace the current view to a PNG, with the frame re-fitted to the
    aspect ratio of the picture.  A plain `ray` or `png` keeps the camera as
    it is, so a picture narrower than the window loses its sides.  The
    labels come out as on screen either way.

USAGE

    bindingsite_png filename [, width [, height [, dpi ]]]

    height = 0 keeps the aspect ratio of the viewport.
    transparent = 1 writes a transparent background.
    fit = 0 keeps the camera exactly as it is.
    """
    width, height, dpi = int(width), int(height), float(dpi)
    filename, name = str(_unquote(filename)), str(_unquote(name)).strip()
    vw, vh = cmd.get_viewport()
    if height <= 0:
        height = int(round(width * float(vh) / float(vw)))
    view = cmd.get_view()
    site = _SITES.get(name) if name + "_ligand" in cmd.get_names("selections") else None
    if site is not None and _truthy(fit):
        _fit(site, aspect=float(width) / float(height))
    old_bg = cmd.get("ray_opaque_background")
    cmd.set("ray_opaque_background", 0 if _truthy(transparent) else 1, quiet=1)
    try:
        with _labels_for_picture(height, everything=True):
            cmd.png(os.path.expanduser(filename), width=width, height=height, dpi=dpi, ray=1, quiet=1)
    finally:
        cmd.set("ray_opaque_background", old_bg, quiet=1)
        cmd.set_view(view)              # the frame was fitted for the picture only
    _say(int(quiet), "wrote %s (%d x %d px)" % (filename, width, height))


def _site_objects(name):
    have = set(cmd.get_names("objects"))
    return [n for n in [name] + _site_names(name) if n in have]


def bindingsite_all(selection="enabled", align=1, quiet=0, _self=cmd, **kwargs):
    """
DESCRIPTION

    One binding-site view per loaded complex, all from the same camera, each
    stored as a scene.  Page through a ligand series with PgDn / PgUp.

USAGE

    bindingsite_all [selection [, align ]] [, any bindingsite option ...]

    align = 1 superposes every complex on the first one beforehand.
    The camera is chosen for the first complex and reused for the others.
    """
    selection = str(_unquote(selection))
    objs = [o for o in cmd.get_object_list("(%s) and polymer" % _not_ours(selection)) or []
            if find_ligands("%" + o)]
    if not objs:
        raise CmdException("bindingsite_all: no complexes with a ligand in '%s'" % selection)
    for key in ("name", "scene", "view"):
        kwargs.pop(key, None)
    ref = objs[0]
    if _truthy(align):
        for obj in objs[1:]:
            try:
                cmd.super("%%%s and polymer" % obj, "%%%s and polymer" % ref, quiet=1)
            except Exception:
                pass
    names = dict((obj, "bs_" + re.sub(r"[^A-Za-z0-9_]", "_", obj)) for obj in objs)
    view, sites = None, {}
    for k, obj in enumerate(objs):
        for other in objs:
            cmd.disable(other)
        cmd.enable(obj)
        cand = find_ligands("%" + obj)[0]
        sites[obj] = bindingsite(cand["sel"], "%" + obj, name=names[obj], quiet=1, scene=0,
                                 view=1 if k == 0 else 0, **kwargs)
        if k == 0:
            view = cmd.get_view()
    # every scene must know every object, so store them only now
    for obj in objs:
        for other in objs:
            cmd.disable(other)
            for n in _site_objects(names[other]):
                cmd.disable(n)
        cmd.enable(obj)
        for n in _site_objects(names[obj]):
            cmd.enable(n)
        cmd.set_view(view)
        _for_this_view(sites[obj], fit=False)
        cmd.scene(obj, "store", quiet=1)
        _say(int(quiet), "%-40s %s" % (obj, _summary(sites[obj])))
    _GALLERY["scenes"] = list(objs)
    _GALLERY["objs"] = list(objs)
    cmd.scene(objs[0], "recall", quiet=1)
    _say(int(quiet), "%d scenes stored; PgDn / PgUp pages through them" % len(objs))


def bindingsite_clear(name="", restore=1, _self=cmd):
    """
DESCRIPTION

    Delete what `bindingsite` created: dashes, labels, colour key, surface,
    selections, scenes.  With restore=1 the settings it changed (background,
    lighting, fog, per-object cartoon transparency ...) are put back as well.
    The molecule keeps the sticks-and-cartoon representation it was given;
    use `as lines` or a preset if you want something else.

USAGE

    bindingsite_clear [name [, restore ]]
    """
    name = str(_unquote(name)).strip()
    names = [name] if name else (list(_SITES) or ["bs"])
    have = set(cmd.get_names("objects"))
    for nm in names:
        site = _SITES.pop(nm, None)
        _delete_site(nm)
        try:
            cmd.scene(nm, "clear", quiet=1)
        except Exception:
            pass
        if site is not None and _truthy(restore):
            for obj in site.get("objs", []):
                if obj in have:
                    for key in ("valence", "cartoon_side_chain_helper", "cartoon_transparency"):
                        try:
                            cmd.unset(key, obj)
                        except Exception:
                            pass
    if not _SITES:
        for sc in _GALLERY["scenes"]:
            try:
                cmd.scene(sc, "clear", quiet=1)
            except Exception:
                pass
        for obj in _GALLERY["objs"]:
            if obj in have:
                cmd.enable(obj)
        _GALLERY["scenes"], _GALLERY["objs"] = [], []
        if _truthy(restore):
            for key, val in list(_SAVED.items()):
                _set(key, val)
            _SAVED.clear()


def bindingsite_param(key="", value="", _self=cmd):
    """
DESCRIPTION

    Show or change a detection threshold.

USAGE

    bindingsite_param                  # list them all
    bindingsite_param hbond_dist, 3.3
    """
    if not key:
        for k, v in PARAM.items():
            print(" %-20s %s" % (k, v))
        return PARAM
    if key not in PARAM:
        raise CmdException("bindingsite_param: unknown key '%s'" % key)
    if value != "":
        PARAM[key] = float(value)
    print(" %s = %s" % (key, PARAM[key]))
    return PARAM[key]


for _fn in (bindingsite, bindingsite_ligands, bindingsite_report, bindingsite_relabel,
            bindingsite_view, bindingsite_png, bindingsite_all, bindingsite_clear, bindingsite_param):
    cmd.extend(_fn.__name__, _fn)
try:
    _wrap_ray_and_png()
except Exception:                       # PyMOL's own commands then stay as they are
    pass
try:
    cmd.auto_arg[0]["bindingsite"] = cmd.auto_arg[0]["zoom"]
    cmd.auto_arg[1]["bindingsite"] = cmd.auto_arg[0]["zoom"]
    cmd.auto_arg[0]["bindingsite_ligands"] = cmd.auto_arg[0]["zoom"]
    cmd.auto_arg[0]["bindingsite_all"] = cmd.auto_arg[0]["zoom"]
except Exception:
    pass


# =============================================================================
#  graphical front end  (Plugin > Binding Site)
# =============================================================================
_dialog = None


def _summary(site):
    counts = OrderedDict()
    for r in site["inter"]:
        if r["type"] in ITYPES and r["draw"]:
            if r["type"] == "metal" and r.get("shell"):
                continue
            counts[r["type"]] = counts.get(r["type"], 0) + 1
    if not counts:
        return "no interactions found"
    return ", ".join("%d %s" % (n, ITYPES[k]["label"].lower() + ("" if n == 1 else
                                 ("es" if ITYPES[k]["label"].endswith("sh") else "s")))
                     for k, n in counts.items())


def _build_dialog():
    from pymol.Qt import QtWidgets, QtCore

    try:
        from pymol import gui as _gui
        parent = _gui.get_qtwindow()
    except Exception:
        parent = None
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("Binding Site  %s" % __version__)
    dlg.setWindowFlags(dlg.windowFlags() | QtCore.Qt.Tool)
    outer = QtWidgets.QVBoxLayout(dlg)
    form = QtWidgets.QFormLayout()
    outer.addLayout(form)

    w = {}
    w["ligand"] = QtWidgets.QComboBox()
    w["ligand"].setEditable(True)
    w["ligand"].setMinimumWidth(300)
    w["ligand"].setToolTip("Pick a detected ligand or type any PyMOL selection")
    refresh = QtWidgets.QPushButton("Find")
    refresh.setToolTip("Search the enabled objects for ligands")
    row = QtWidgets.QHBoxLayout()
    row.addWidget(w["ligand"], 1)
    row.addWidget(refresh)
    form.addRow("Ligand", row)

    w["style"] = QtWidgets.QComboBox()
    for key, st in STYLES.items():
        w["style"].addItem("%s  -  %s" % (key, st["doc"]), key)
    form.addRow("Style", w["style"])

    w["surface"] = QtWidgets.QComboBox()
    for key, text in (("none", "none"), ("pocket", "pocket surface behind the ligand"),
                      ("polar", "pocket surface, N blue / O red")):
        w["surface"].addItem(text, key)
    form.addRow("Surface", w["surface"])

    w["label_format"] = QtWidgets.QComboBox()
    w["label_format"].setEditable(True)
    w["label_format"].addItems(["{resn}{resi}", "{r}{resi}", "{resn}{resi}{chain}", "{chain}:{resn}{resi}"])
    w["label_format"].setToolTip("{resn} Ser   {RESN} SER   {r} S   {resi}   {chain}")
    form.addRow("Label format", w["label_format"])

    w["offset"] = QtWidgets.QLineEdit("0")
    w["offset"].setToolTip("Added to residue numbers in labels and reports, e.g. 27, or per chain: A:27,B:-21")
    form.addRow("Numbering offset", w["offset"])

    w["ligand_label"] = QtWidgets.QLineEdit()
    w["ligand_label"].setPlaceholderText("optional, e.g. the compound name")
    form.addRow("Ligand label", w["ligand_label"])

    def spin(lo, hi, step, val, tip):
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(2)
        sb.setValue(val)
        sb.setToolTip(tip)
        return sb

    w["radius"] = spin(0.0, 10.0, 0.5, 4.0, "Residues within this distance of the ligand are shown as thin sticks")
    w["label_size"] = spin(0.2, 3.0, 0.05, 0.70, "Label height in Angstrom; scales with the zoom")
    row = QtWidgets.QHBoxLayout()
    row.addWidget(QtWidgets.QLabel("lining radius"))
    row.addWidget(w["radius"])
    row.addSpacing(12)
    row.addWidget(QtWidgets.QLabel("label size"))
    row.addWidget(w["label_size"])
    row.addStretch(1)
    form.addRow(u"Sizes (Å)", row)

    grid = QtWidgets.QGridLayout()
    checks = (("labels", "Residue labels", True, ""),
              ("legend", "Colour key", True, ""),
              ("distances", "Distances on the dashes", False, ""),
              ("lining", "Lining residues", True, "Residues near the ligand without a specific contact"),
              ("hydrophobic", "Hydrophobic contacts", True, ""),
              ("cartoon", "Cartoon backdrop", True, ""),
              ("his_positive", "Histidines charged (acidic pH)", False,
               "Treat histidines of unknown protonation as cationic, as in a lysosome"),
              ("view", "Choose the camera for me", True, "Off: keep the current camera"))
    for k, (key, text, on, tip) in enumerate(checks):
        w[key] = QtWidgets.QCheckBox(text)
        w[key].setChecked(on)
        if tip:
            w[key].setToolTip(tip)
        grid.addWidget(w[key], k // 2, k % 2)
    outer.addLayout(grid)

    status = QtWidgets.QLabel("")
    status.setWordWrap(True)
    outer.addWidget(status)

    buttons = QtWidgets.QGridLayout()
    b_show = QtWidgets.QPushButton("Show binding site")
    b_show.setDefault(True)
    b_view = QtWidgets.QPushButton("Next view")
    b_relabel = QtWidgets.QPushButton("Re-place labels")
    b_relabel.setToolTip("Rotate the scene to taste, then lay the labels out again for that view")
    b_report = QtWidgets.QPushButton("Save report...")
    b_png = QtWidgets.QPushButton("Save image...")
    b_clear = QtWidgets.QPushButton("Clear")
    buttons.addWidget(b_show, 0, 0, 1, 3)
    for k, b in enumerate((b_view, b_relabel, b_clear, b_png, b_report)):
        buttons.addWidget(b, 1 + k // 3, k % 3)
    outer.addLayout(buttons)

    def guarded(fn):
        def run(*_):
            try:
                fn()
            except CmdException as exc:
                status.setText(str(exc).replace("Error:", "").strip())
            except Exception as exc:      # keep PyMOL alive whatever happens
                status.setText("%s: %s" % (type(exc).__name__, exc))
        return run

    def find():
        current = w["ligand"].currentText()
        w["ligand"].clear()
        w["ligand"].addItem("auto  (most buried ligand)", "auto")
        for c in find_ligands("enabled"):
            w["ligand"].addItem("%s  [%s, %d atoms]" % (c["pretty"], c["kind"], c["natoms"]), c["sel"])
        k = w["ligand"].findText(current)
        if k >= 0:
            w["ligand"].setCurrentIndex(k)
        elif current and not current.startswith("auto"):
            w["ligand"].setEditText(current)
        status.setText("%d ligand candidate%s" % (w["ligand"].count() - 1,
                                                  "" if w["ligand"].count() == 2 else "s"))

    def ligand_text():
        idx = w["ligand"].currentIndex()
        if idx >= 0 and w["ligand"].itemText(idx) == w["ligand"].currentText():
            return w["ligand"].itemData(idx)
        return w["ligand"].currentText().strip() or "auto"

    def show():
        site = bindingsite(ligand=ligand_text(), style=w["style"].currentData(),
                           surface=w["surface"].currentData(), radius=w["radius"].value(),
                           label_size=w["label_size"].value(), offset=w["offset"].text() or 0,
                           label_format=w["label_format"].currentText() or "{resn}{resi}",
                           ligand_label=w["ligand_label"].text(),
                           **dict((key, int(w[key].isChecked())) for key, _, _, _ in checks))
        text = _summary(site)
        n_clash = sum(1 for r in site["inter"] if r["type"] == "clash")
        if n_clash:
            text += ".  Check this pose: the ligand clashes with the pocket"
        status.setText(text)

    def save_png():
        fn = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save ray-traced image", "binding_site.png",
                                                   "PNG image (*.png)")[0]
        if fn:
            status.setText("ray tracing...")
            QtWidgets.QApplication.processEvents()
            bindingsite_png(fn, width=2400, dpi=300)
            status.setText("wrote %s" % fn)

    def save_report():
        fn = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save interaction table", "interactions.csv",
                                                   "CSV (*.csv);;Tab-separated (*.tsv);;Markdown (*.md)")[0]
        if fn:
            bindingsite_report(fn)
            status.setText("wrote %s" % fn)

    def clear():
        bindingsite_clear()
        status.setText("cleared")

    refresh.clicked.connect(guarded(find))
    b_show.clicked.connect(guarded(show))
    b_view.clicked.connect(guarded(lambda: bindingsite_view("next")))
    b_relabel.clicked.connect(guarded(lambda: bindingsite_relabel()))
    b_report.clicked.connect(guarded(save_report))
    b_png.clicked.connect(guarded(save_png))
    b_clear.clicked.connect(guarded(clear))
    dlg._bs = dict(widgets=w, status=status, find=find, show=show)
    guarded(find)()
    return dlg


def bindingsite_gui(_self=cmd):
    """
DESCRIPTION

    Open the Binding Site dialog (also under Plugin > Binding Site).
    """
    global _dialog
    if _dialog is None:
        _dialog = _build_dialog()
    else:
        _dialog._bs["find"]()
    _dialog.show()
    _dialog.raise_()
    return _dialog


cmd.extend("bindingsite_gui", bindingsite_gui)


def __init_plugin__(app=None):
    """Called by PyMOL's plugin manager: add the menu entry."""
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("Binding Site", bindingsite_gui)
