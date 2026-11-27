# Changelog

## 1.0.0 — 2026-11-27

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
