"""FE-aware MCS atom mapping between two ligands."""

import sys
from pathlib import Path

from .parameterize import _infer_net_charge


def _atom_names_from_mol(mol) -> dict[int, str]:
    """Return {rdkit_atom_idx: tripos_atom_name} preserving mol2 names."""
    names: dict[int, str] = {}
    for atom in mol.GetAtoms():
        mi = atom.GetMonomerInfo()
        names[atom.GetIdx()] = mi.GetName().strip() if mi else f"X{atom.GetIdx()}"
    return names


def _fe_score(mol_old, mol_new,
              match_old: tuple[int, ...],
              match_new: tuple[int, ...]) -> float:
    """
    Score an atom-atom mapping for free energy suitability.

    Lower = better.  Penalties:
      +10 per pair with different atomic numbers
      + 5 per pair where ring membership differs
      + 3 per pair with different formal charges
      + 1 per unmatched atom on either side
    """
    score = 0.0
    ri_old = mol_old.GetRingInfo()
    ri_new = mol_new.GetRingInfo()
    for io, in_ in zip(match_old, match_new):
        ao = mol_old.GetAtomWithIdx(io)
        an = mol_new.GetAtomWithIdx(in_)
        if ao.GetAtomicNum() != an.GetAtomicNum():
            score += 10
        if (ri_old.NumAtomRings(io) > 0) != (ri_new.NumAtomRings(in_) > 0):
            score += 5
        if ao.GetFormalCharge() != an.GetFormalCharge():
            score += 3
    score += (mol_old.GetNumAtoms() - len(match_old) +
              mol_new.GetNumAtoms() - len(match_new))
    return score


def compute_fe_mapping(pdb_old: Path, pdb_new: Path) -> dict:
    """
    Find the FE-optimal atom mapping between two ligands.

    Algorithm
    ---------
    1. Load both PDB files with RDKit (keep explicit H); assign bond orders
       via DetermineBonds using the auto-detected net charge.
    2. Strip H and run FindMCS on heavy atoms with ring-consistent constraints.
    3. Score all MCS matches on both molecules; pick the lowest-penalty pair.
    4. Extend the heavy-atom assignment to bonded hydrogens.

    Returns
    -------
    dict with keys:
        matched_old   atom names (old mol) in the common core
        matched_new   atom names (new mol, same ordering as matched_old)
        unique_old    names unique to old ligand → soft-core (scmask1)
        unique_new    names unique to new ligand → soft-core (scmask2)
        score         float penalty (lower = better mapping)
        n_common_heavy int
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS, rdDetermineBonds
    except ImportError:
        sys.exit("rdkit required:  conda install -c conda-forge rdkit")

    def _load(pdb: Path):
        mol = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
        if mol is None:
            sys.exit(f"RDKit could not parse {pdb} — check the PDB format.")
        charge = _infer_net_charge(pdb)
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.SanitizeMol(mol)
        return mol

    mol_old_h = _load(pdb_old)
    mol_new_h = _load(pdb_new)

    names_old_h = _atom_names_from_mol(mol_old_h)
    names_new_h = _atom_names_from_mol(mol_new_h)

    mol_old = Chem.RemoveHs(mol_old_h)
    mol_new = Chem.RemoveHs(mol_new_h)
    names_old = _atom_names_from_mol(mol_old)
    names_new = _atom_names_from_mol(mol_new)

    mcs = rdFMCS.FindMCS(
        [mol_old, mol_new],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=60,
    )

    if mcs.numAtoms == 0:
        print("  WARNING: no common heavy-atom substructure found.")
        print("           All atoms will be soft-core. Convergence may be slow.")
        return {
            "matched_old": [],
            "matched_new": [],
            "unique_old": list(names_old_h.values()),
            "unique_new": list(names_new_h.values()),
            "score": float("inf"),
            "n_common_heavy": 0,
        }

    query = Chem.MolFromSmarts(mcs.smartsString)
    matches_old = mol_old.GetSubstructMatches(query, uniquify=False)
    matches_new = mol_new.GetSubstructMatches(query, uniquify=False)

    best_score = float("inf")
    best_io: tuple = matches_old[0]
    best_in: tuple = matches_new[0]
    for mo in matches_old:
        for mn in matches_new:
            s = _fe_score(mol_old, mol_new, mo, mn)
            if s < best_score:
                best_score, best_io, best_in = s, mo, mn

    def _h_neighbors(mol_h, names_h: dict) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for idx, name in names_h.items():
            if mol_h.GetAtomWithIdx(idx).GetAtomicNum() == 1:
                continue
            result[name] = [
                names_h[n.GetIdx()]
                for n in mol_h.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetAtomicNum() == 1
            ]
        return result

    h_nbrs_old = _h_neighbors(mol_old_h, names_old_h)
    h_nbrs_new = _h_neighbors(mol_new_h, names_new_h)

    matched_old: list[str] = []
    matched_new: list[str] = []
    unique_old:  list[str] = []
    unique_new:  list[str] = []

    for io, in_ in zip(best_io, best_in):
        old_name = names_old[io]
        new_name = names_new[in_]
        matched_old.append(old_name)
        matched_new.append(new_name)

        old_hs = h_nbrs_old.get(old_name, [])
        new_hs = h_nbrs_new.get(new_name, [])
        n = min(len(old_hs), len(new_hs))
        matched_old.extend(old_hs[:n])
        matched_new.extend(new_hs[:n])
        unique_old.extend(old_hs[n:])
        unique_new.extend(new_hs[n:])

    matched_old_heavy = {names_old[i] for i in best_io}
    matched_new_heavy = {names_new[i] for i in best_in}

    for idx, name in names_old.items():
        if name not in matched_old_heavy:
            unique_old.append(name)
            unique_old.extend(h_nbrs_old.get(name, []))

    for idx, name in names_new.items():
        if name not in matched_new_heavy:
            unique_new.append(name)
            unique_new.extend(h_nbrs_new.get(name, []))

    return {
        "matched_old":    matched_old,
        "matched_new":    matched_new,
        "unique_old":     unique_old,
        "unique_new":     unique_new,
        "score":          best_score,
        "n_common_heavy": len(best_io),
    }
