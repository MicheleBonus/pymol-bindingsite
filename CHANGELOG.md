# Changelog

## 1.0.1 - 2026-11-27

- Fixed: in a picture ray-traced larger than the window with PyMOL's own `ray`
  or `png` (for example `ray 4000, 4000`), the residue and distance labels kept
  their on-screen size in pixels and were too small to read. PyMOL's ray-tracer enlarges labels sized in pixels for a
  larger picture but not labels sized in Å, which the plugin uses. `ray` and
  `png` are now wrapped, and the label size of binding-site objects is corrected
  while a picture is traced. `bindingsite_png` did this already.
- When labels have to be corrected, `ray ..., async=1` (the *Draw/Ray* button)
  runs synchronously.
- A plain `ray` or `png` narrower than the window prints a hint that the sides
  are cut off and that `bindingsite_png` frames the site for the picture.

## 1.0.0 - 2026-11-27

First public release.

- `bindingsite`: finds the ligand, perceives donors, acceptors, charges, aromatic
  rings and electrophilic warheads from heavy-atom geometry, and draws the site
  with an automatic camera, residue labels that do not overlap and a colour key.
- Interactions: hydrogen bonds, salt bridges, π–π stacking, cation–π, halogen
  bonds, metal coordination, water bridges, hydrophobic contacts, covalent
  links, near-attack geometry of covalent warheads, steric clashes.
- Companion commands: `bindingsite_report` (console, `.csv`, `.tsv`, `.md`),
  `bindingsite_png`, `bindingsite_relabel`, `bindingsite_view`,
  `bindingsite_ligands`, `bindingsite_all`, `bindingsite_clear`,
  `bindingsite_param`, and the dialog `bindingsite_gui` under
  *Plugin › Binding Site*.
- Four styles: `publication`, `soft`, `flat`, `dark`.
- Tests on ten reference complexes from the PDB and a set of edge cases.
