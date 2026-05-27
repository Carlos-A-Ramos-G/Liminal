#!/usr/bin/env python3
"""
rbfe_runner.py — AmberTools-native Relative Binding Free Energy workflow.

Automates the full RBFE pipeline starting from raw ligand mol2 files and a
protein PDB, eliminating all manual atom-mask specification.

Pipeline
--------
  prepare   Parameterise ligands (antechamber + parmchk2), find the optimal
            atom mapping (FE-aware MCS), build solvated dual-residue AMBER
            systems (tleap), derive TI masks automatically (parmed), and
            write all production AMBER input files.
  submit    Submit the generated SLURM / local job scripts.
  analyse   Compute ΔΔG via TI (Gauss-Legendre quadrature) and, when
            mbar: true, via MBAR (pymbar >= 4).

What makes this less error-prone than manual setup
---------------------------------------------------
  * timask1/2 and scmask1/2 are derived automatically from a scored MCS that
    penalises element changes, ring/non-ring switches, and charge changes —
    not just raw atom count.
  * Net charge change (ΔQ ≠ 0) is detected and reported before any
    simulation files are written.
  * parmchk2 ATTN warnings abort the run; partial parameter sets never reach
    the cluster.
  * The atom mapping is written to prep/mapping.json for user inspection and
    optional manual override before production inputs are generated.
  * The protein PDB is screened for alternate locations and non-standard
    residues before tleap is invoked.

Environment (AmberTools25 conda env)
-------------------------------------
  conda activate AmberTools25
  conda install -c conda-forge rdkit pymbar pyyaml

Usage
-----
  python rbfe_runner.py prepare [--dry-run]
  python rbfe_runner.py submit  [--mode serial|parallel|local]
  python rbfe_runner.py analyse [--tail N]
  python rbfe_runner.py --config my_config.yaml prepare
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required:  conda install -c conda-forge pyyaml")


# =============================================================================
# Gauss-Legendre quadrature
# =============================================================================

def compute_gl_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """GL nodes on [0, 1] and weights summing to 1."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (nodes + 1) / 2, weights / 2


def _middle(n: int) -> int:
    return (n + 1) // 2


# =============================================================================
# MBAR helpers
# =============================================================================

_KB_KCAL: float = 0.001987204258   # kcal mol⁻¹ K⁻¹


def _mbar_beta(temp: float) -> float:
    """β = 1/(k_B T) in mol kcal⁻¹.  Temperature from config."""
    return 1.0 / (_KB_KCAL * temp)


def _format_mbar_block(lambdas: np.ndarray) -> str:
    """
    AMBER &cntrl fragment for ifmbar=1.

    Adds λ=0 and λ=1 as unsampled endpoint states (N_k = 0 for those rows)
    so pymbar.MBAR spans the full alchemical interval.
    """
    all_lam = np.concatenate([[0.0], lambdas, [1.0]])
    lam_str = ", ".join(f"{l:.5f}" for l in all_lam)
    return (
        f"\n   ifmbar = 1, mbar_states = {len(all_lam)},"
        f"\n   mbar_lambda = {lam_str},"
    )


# =============================================================================
# Environment check
# =============================================================================

def check_environment() -> None:
    """Verify required tools and Python packages are reachable. Exit clearly."""
    missing: list[str] = []
    for tool in ("antechamber", "parmchk2", "tleap"):
        if not shutil.which(tool):
            missing.append(f"  {tool:<14} → activate the AmberTools25 conda env")
    for pkg, install in (("rdkit", "rdkit"), ("yaml", "pyyaml")):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(
                f"  {pkg:<14} → conda install -c conda-forge {install}"
            )
    if missing:
        sys.exit(
            "Missing dependencies — run inside 'conda activate AmberTools25':\n"
            + "\n".join(missing)
        )


# =============================================================================
# Config validation
# =============================================================================

def validate_config(cfg: dict, require_ligands: bool = True) -> None:
    """Raise SystemExit with a precise message on the first config error."""

    def _req(d: dict, *keys: str, label: str = "config") -> None:
        for k in keys:
            if k not in d:
                sys.exit(f"Config error: '{label}' is missing required key '{k}'")

    _req(cfg, "temperature", "n_lambdas", "replicates",
         "forcefield", "system", "simulation", "amber", "slurm")

    if cfg["n_lambdas"] < 3:
        sys.exit("Config error: n_lambdas must be >= 3")
    if cfg["temperature"] <= 0:
        sys.exit("Config error: temperature must be > 0 K")
    if cfg["replicates"] < 1:
        sys.exit("Config error: replicates must be >= 1")

    if require_ligands:
        _req(cfg, "ligands")
        _req(cfg["ligands"], "old", "new", label="ligands")
        for side in ("old", "new"):
            lig = cfg["ligands"][side]
            _req(lig, "pdb", "name", label=f"ligands.{side}")
            pdb = Path(lig["pdb"])
            if not pdb.exists():
                sys.exit(f"Config error: ligands.{side}.pdb not found: {pdb}")
            if len(lig["name"]) > 4:
                sys.exit(
                    f"Config error: ligands.{side}.name must be ≤ 4 characters "
                    f"(AMBER limit); got '{lig['name']}'"
                )

    if "protein" in cfg:
        pdb = Path(cfg["protein"]["pdb"])
        if not pdb.exists():
            sys.exit(f"Config error: protein.pdb not found: {pdb}")

    _req(cfg["forcefield"], "protein", "ligand", "water",
         label="forcefield")
    _req(cfg["system"], "box_padding", "charge_method", "ion_concentration",
         label="system")

    valid_gaff  = {"gaff", "gaff2"}
    valid_water = {"tip3p", "tip4pew", "opc"}
    ff = cfg["forcefield"]
    if ff["ligand"] not in valid_gaff:
        sys.exit(f"Config error: forcefield.ligand must be one of {valid_gaff}")
    if ff["water"].lower() not in valid_water:
        sys.exit(f"Config error: forcefield.water must be one of {valid_water}")

    for stage in ("min", "heating", "equil", "prod"):
        if stage not in cfg["simulation"]:
            sys.exit(f"Config error: simulation.{stage} section is missing")


# =============================================================================
# Subprocess helpers
# =============================================================================

def _run(cmd: list, cwd: Path | None = None, desc: str = "") -> str:
    """Run a subprocess, print what was run, return stdout.  Exit on failure."""
    label = desc or Path(cmd[0]).name
    print(f"    → {label}: {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        sys.exit(
            f"\nERROR: {label} failed (exit {res.returncode})\n"
            f"stdout (last 40 lines):\n"
            + "\n".join((res.stdout or "").splitlines()[-40:])
            + "\nstderr:\n"
            + "\n".join((res.stderr or "").splitlines()[-20:])
        )
    return res.stdout


def _check_frcmod(frcmod: Path) -> None:
    """Abort if parmchk2 produced any ATTN (missing parameter) lines."""
    attn = [l for l in frcmod.read_text().splitlines() if "ATTN" in l]
    if attn:
        sys.exit(
            f"parmchk2 found missing parameters in {frcmod}:\n"
            + "\n".join(f"  {l}" for l in attn)
            + "\n\nAdd a custom frcmod for these terms or choose a different "
              "forcefield / charge method."
        )


# =============================================================================
# Ligand parameterisation
# =============================================================================

def _patch_sqm_out(sqm_out: Path) -> bool:
    """Reformat sqm.out from new-style header to the format antechamber expects.

    New sqm (AmberTools >= 24) writes:
        Atom    Element       Mulliken Charge
        ...
    Old antechamber parser expects:
        Mulliken charges:
        ...

    Returns True if the file was patched, False if already in expected format.
    """
    text = sqm_out.read_text()
    if "Mulliken charges:" in text:
        return False
    # Match the new-style table header and convert
    patched = re.sub(
        r"Atom\s+Element\s+Mulliken Charge\s*\n",
        "Mulliken charges:\n          1\n",
        text,
    )
    # Reformat data rows:  "   1  C     -0.123456" → "   1  C    -0.123456"
    # (antechamber expects the charge in column 3, which already matches — no
    #  column shift needed, just the header swap above is sufficient)
    if patched == text:
        return False
    sqm_out.write_text(patched)
    return True


def _run_antechamber(cmd: list, cwd: Path, desc: str) -> str:
    """Run antechamber, retrying once after patching sqm.out if charge parsing fails.

    Some AmberTools builds ship a sqm that writes a different Mulliken charge
    header than the antechamber parser expects. When that mismatch is detected
    we patch sqm.out and re-invoke antechamber with a no-op sqm shim so sqm
    does not overwrite the patched file.
    """
    label = desc or "antechamber"
    print(f"    → {label}: {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        capture_output=True, text=True,
    )
    if res.returncode == 0:
        return res.stdout

    stderr_lower = (res.stderr or "").lower() + (res.stdout or "").lower()
    sqm_out = cwd / "sqm.out"
    if "mulliken" not in stderr_lower or not sqm_out.exists():
        sys.exit(
            f"\nERROR: {label} failed (exit {res.returncode})\n"
            f"stdout (last 40 lines):\n"
            + "\n".join((res.stdout or "").splitlines()[-40:])
            + "\nstderr:\n"
            + "\n".join((res.stderr or "").splitlines()[-20:])
        )

    if not _patch_sqm_out(sqm_out):
        sys.exit(
            f"\nERROR: {label} failed (exit {res.returncode}) and sqm.out "
            f"could not be patched.\nstdout:\n"
            + "\n".join((res.stdout or "").splitlines()[-40:])
            + "\nstderr:\n"
            + "\n".join((res.stderr or "").splitlines()[-20:])
        )

    print(f"    → sqm/antechamber version mismatch detected — patched sqm.out, retrying")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a no-op sqm shim so antechamber won't re-run sqm
        shim = Path(tmpdir) / "sqm"
        shim.write_text("#!/bin/sh\n# no-op shim: sqm already ran\nexit 0\n")
        shim.chmod(0o755)
        env = {**os.environ, "PATH": f"{tmpdir}:{os.environ.get('PATH', '')}"}
        res2 = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd),
            capture_output=True, text=True,
            env=env,
        )

    if res2.returncode != 0:
        sys.exit(
            f"\nERROR: {label} failed after sqm.out patch (exit {res2.returncode})\n"
            f"stdout (last 40 lines):\n"
            + "\n".join((res2.stdout or "").splitlines()[-40:])
            + "\nstderr:\n"
            + "\n".join((res2.stderr or "").splitlines()[-20:])
        )
    return res2.stdout


def _infer_net_charge(pdb_path: Path) -> int:
    """Infer net formal charge from a ligand PDB using RDKit.

    Tries candidate total charges until DetermineBonds produces a self-consistent
    assignment. Raises RuntimeError if none converge (bad geometry or missing H).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        sys.exit("rdkit required:  conda install -c conda-forge rdkit")
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if mol is None:
        raise RuntimeError(f"RDKit could not parse {pdb_path}")
    for charge in [0, -1, 1, -2, 2, -3, 3]:
        try:
            mol_try = Chem.RWMol(Chem.Mol(mol))
            rdDetermineBonds.DetermineBonds(mol_try, charge=charge)
            Chem.SanitizeMol(mol_try)
            if Chem.GetFormalCharge(mol_try) == charge:
                return charge
        except Exception:
            continue
    raise RuntimeError(
        f"Could not determine net charge for {pdb_path.name}. "
        "Check that the PDB contains all hydrogens and has correct geometry, "
        "or set net_charge to an explicit integer in config.yaml."
    )


def _resolve_charge(pdb_path: Path, cfg: dict) -> int:
    raw = cfg["system"].get("net_charge", "auto")
    if str(raw).lower() == "auto":
        return _infer_net_charge(pdb_path)
    return int(raw)


def parameterize_ligand(
    pdb: Path, resname: str, cfg: dict,
) -> tuple[Path, Path]:
    """
    Run antechamber → parmchk2 → tleap for one ligand.

    Outputs are written to parameters/{resname}/:
        {resname}.mol2    GAFF atom types + AM1-BCC charges
        {resname}.frcmod  missing GAFF parameters
        {resname}.lib     AMBER off-library file

    Returns (lib, frcmod).  Aborts on any parmchk2 ATTN warning.
    """
    work_dir = Path("parameters") / resname
    work_dir.mkdir(parents=True, exist_ok=True)
    gaff   = cfg["forcefield"]["ligand"]        # gaff | gaff2
    method = cfg["system"]["charge_method"]     # bcc  | mul
    mult   = cfg["system"].get("multiplicity", 1)
    charge = _resolve_charge(pdb, cfg)

    # Use absolute paths throughout so subprocess cwd never interferes
    out_mol2 = (work_dir / f"{resname}.mol2").resolve()
    frcmod   = (work_dir / f"{resname}.frcmod").resolve()
    lib      = (work_dir / f"{resname}.lib").resolve()

    _run_antechamber(
        ["antechamber",
         "-i",  str(pdb.resolve()), "-fi", "pdb",
         "-o",  str(out_mol2),      "-fo", "mol2",
         "-c",  method, "-s", "2",
         "-nc", str(charge), "-m", str(mult),
         "-rn", resname, "-at", gaff],
        cwd=work_dir.resolve(),
        desc=f"antechamber ({resname})",
    )
    _run(
        ["parmchk2",
         "-i", str(out_mol2), "-f", "mol2",
         "-o", str(frcmod), "-s", gaff],
        cwd=work_dir.resolve(),
        desc=f"parmchk2 ({resname})",
    )
    _check_frcmod(frcmod)

    tleap_in = (work_dir / "tleap_lib.in").resolve()
    tleap_in.write_text(
        f"source {_FF_SOURCES.get(gaff, 'leaprc.' + gaff)}\n"
        f"{resname} = loadmol2 {out_mol2}\n"
        f"check {resname}\n"
        f"loadamberparams {frcmod}\n"
        f"saveoff {resname} {lib}\n"
        "quit\n"
    )
    _run(["tleap", "-f", str(tleap_in)], cwd=work_dir.resolve(),
         desc=f"tleap lib ({resname})")

    if not lib.exists():
        sys.exit(f"tleap did not produce {lib} — check {work_dir}/leap.log")

    return lib, frcmod


# =============================================================================
# FE-aware atom mapping
# =============================================================================

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
        in_ring_old = ri_old.NumAtomRings(io) > 0
        in_ring_new = ri_new.NumAtomRings(in_) > 0
        if in_ring_old != in_ring_new:
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
    3. Score *all* MCS matches on both molecules; pick the lowest-penalty pair.
    4. Extend the heavy-atom assignment to bonded hydrogens.

    Returns
    -------
    dict with keys:
        matched_old   atom names (old mol) in the common core
        matched_new   atom names (new mol, same ordering as matched_old)
        unique_old    names unique to old ligand → will be soft-core (scmask1)
        unique_new    names unique to new ligand → will be soft-core (scmask2)
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

    # No common substructure — all atoms soft-core (valid but slow to converge)
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

    # Build {heavy_atom_name: [H_neighbor_names]} for each molecule ----------
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

    # Matched heavy-atom pairs: compare H counts per pair.
    # Excess H on either side go to that side's unique list — this correctly
    # handles fused-ring cases (e.g. benzene→naphthalene) where a matched
    # carbon loses its H upon becoming a ring-junction atom.
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

    # Unique heavy atoms and all their hydrogens
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


# =============================================================================
# Net charge detection
# =============================================================================

def detect_formal_charges(pdb_old: Path, pdb_new: Path) -> tuple[int, int]:
    """Return (q_old, q_new) formal charges from PDB files via RDKit."""
    return _infer_net_charge(pdb_old), _infer_net_charge(pdb_new)


# =============================================================================
# PDB pre-flight check
# =============================================================================

def check_pdb(pdb: Path) -> None:
    """Screen a protein PDB for issues that cause silent tleap failures."""
    text = pdb.read_text()

    if re.search(r"^(?:ATOM|HETATM).{10}[AB] ", text, re.MULTILINE):
        print(
            f"  NOTE [{pdb.name}]: alternate locations (ALTLOC) detected.\n"
            "  tleap will use whichever conformation appears first in the file.\n"
            "  If this causes issues, clean with: "
            "pdb4amber -i protein.pdb -o protein_clean.pdb"
        )

    known_het = {"HOH", "WAT", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE"}
    lig_names = {cfg_lig_name for cfg_lig_name in []}   # populated later in prepare()
    non_std = (
        set(re.findall(r"^HETATM.{8}(\S+)", text, re.MULTILINE)) - known_het
    )
    if non_std:
        print(
            f"  NOTE [{pdb.name}]: non-standard HETATM residues found: "
            f"{', '.join(sorted(non_std))}\n"
            "  These are fine if they are your TI ligands; otherwise they need "
            "separate parameterisation."
        )



# =============================================================================
# tleap system building
# =============================================================================

_FF_SOURCES: dict[str, str] = {
    "ff14SB":  "leaprc.protein.ff14SB",
    "ff19SB":  "leaprc.protein.ff19SB",
    "gaff":    "leaprc.gaff",
    "gaff2":   "leaprc.gaff2",
    "tip3p":   "leaprc.water.tip3p",
    "tip4pew": "leaprc.water.tip4pew",
    "opc":     "leaprc.water.opc",
    # Note: ion parameters (ionsjc_tip3p etc.) are loaded automatically
    # by the water leaprc — there is no separate leaprc.water.ionsjc_* file.
}

_BOX_NAME: dict[str, str] = {
    "tip3p":  "TIP3PBOX",
    "tip4pew":"TIP4PEWBOX",
    "opc":    "OPCBOX",
}


def _estimate_ion_counts(total_solute_charge: int,
                         conc_M: float,
                         padding: float) -> tuple[int, int]:
    """
    Return (n_Na, n_Cl) for charge neutralisation + physiological NaCl.

    Water count is estimated from box volume assuming ~30 Å³/water and a
    cubic box with edge ≈ 2 * padding + 50 Å (generous for most complexes).
    The salt count is rounded to the nearest integer.
    """
    edge_A   = 2 * padding + 50.0
    vol_L    = (edge_A * 1e-10) ** 3 * 1e3          # m³ → L
    n_salt   = max(0, round(conc_M * 6.022e23 * vol_L))

    if total_solute_charge < 0:
        n_Na = -total_solute_charge + n_salt
        n_Cl = n_salt
    elif total_solute_charge > 0:
        n_Na = n_salt
        n_Cl = total_solute_charge + n_salt
    else:
        n_Na = n_Cl = n_salt

    return n_Na, n_Cl


def _ff_source(key: str, val: str) -> str:
    """Return the full leaprc name for a forcefield key/value pair."""
    if val in _FF_SOURCES:
        return _FF_SOURCES[val]
    if key == "protein":
        return f"leaprc.protein.{val}"
    if key == "water":
        return f"leaprc.water.{val}"
    return f"leaprc.{val}"


def _tleap_load(resname: str, path: Path) -> str:
    """Return the tleap load line for a mol2 or lib file."""
    if path.suffix == ".lib":
        return f"loadOff {path.resolve()}"
    return f"{resname} = loadMol2 {path.resolve()}"


def build_system(
    leg: str,
    lig_old_struct: Path, lig_old_frcmod: Path,
    lig_new_struct: Path, lig_new_frcmod: Path,
    old_resname: str, new_resname: str,
    q_old: int, q_new: int,
    cfg: dict, work_dir: Path,
) -> tuple[Path, Path]:
    """
    Write and execute a tleap script to build the solvated dual-residue system.

    leg : 'unbound'  — two ligands in water only
          'bound'    — protein + two ligands in water

    Both ligands are loaded as separate residues so AMBER TI can use
    timask1/timask2 to select them independently.

    Net charge change (ΔQ ≠ 0)
    ---------------------------
    Both ligand charges contribute to the combined topology charge.  tleap's
    addions command neutralises this sum.  When ΔQ ≠ 0 the neutralisation
    ion count differs between λ=0 and λ=1; this introduces a small systematic
    bias (same as FESetup's default behaviour).  For rigorous ΔQ ≠ 0 handling,
    build separate topologies for each endpoint — see AMBER TI tutorial 3.

    Returns (parm7_path, rst7_path).
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    ff      = cfg["forcefield"]
    sys_cfg = cfg["system"]
    padding = float(sys_cfg["box_padding"])
    conc    = float(sys_cfg["ion_concentration"])
    water   = ff["water"].lower()
    box     = _BOX_NAME.get(water, "TIP3PBOX")

    # Both ligands are present simultaneously → sum their charges
    combined_charge = q_old + q_new
    n_Na, n_Cl = _estimate_ion_counts(combined_charge, conc, padding)

    sources = [
        f"source {_ff_source(key, val)}"
        for key in ("protein", "ligand", "water")
        for val in [ff.get(key)]
        if val
    ]
    lines: list[str] = sources + ["",
        f"loadAmberParams {lig_old_frcmod.resolve()}",
        _tleap_load(old_resname, lig_old_struct),
        "",
        f"loadAmberParams {lig_new_frcmod.resolve()}",
        _tleap_load(new_resname, lig_new_struct),
        "",
    ]

    if leg == "bound" and "protein" in cfg:
        pdb = Path(cfg["protein"]["pdb"]).resolve()
        lines += [
            f"protein = loadPdb {pdb}",
            f"solute  = combine {{protein {old_resname} {new_resname}}}",
        ]
    else:
        lines += [f"solute = combine {{{old_resname} {new_resname}}}"]

    lines += [
        "",
        f"solvatebox solute {box} {padding}",
        "addions solute Na+ 0",
        "addions solute Cl- 0",
    ]
    if n_Na:
        lines.append(f"addionsrand solute Na+ {n_Na}")
    if n_Cl:
        lines.append(f"addionsrand solute Cl- {n_Cl}")

    out_parm = work_dir / "ti.parm7"
    out_rst  = work_dir / "ti.rst7"
    lines += [
        "",
        f"saveAmberParm solute {out_parm.resolve()} {out_rst.resolve()}",
        "quit",
    ]

    tleap_in = work_dir / "tleap.in"
    tleap_in.write_text("\n".join(lines) + "\n")

    stdout = _run(["tleap", "-f", str(tleap_in.resolve())],
                  cwd=work_dir, desc=f"tleap ({leg})")
    (work_dir / "tleap.out").write_text(stdout)

    for f in (out_parm, out_rst):
        if not f.exists():
            sys.exit(
                f"tleap did not produce {f}.\n"
                f"Check {work_dir / 'tleap.out'} for details."
            )

    return out_parm, out_rst


# =============================================================================
# Mask derivation
# =============================================================================

def derive_masks(
    parm7: Path,
    old_resname: str, new_resname: str,
    unique_old_names: list[str], unique_new_names: list[str],
) -> tuple[str, str, str, str]:
    """
    Derive AMBER TI mask strings by parsing the parm7 RESIDUE_LABEL section.

    timask selects whole residues (`:N` format).
    scmask selects soft-core atoms by name within a residue (`:N@n1,n2,...`).

    Returns (timask1, timask2, scmask1, scmask2).
    """
    m = re.search(
        r'%FLAG RESIDUE_LABEL\s+%FORMAT\([^)]+\)\s+(.*?)(?=%FLAG|\Z)',
        parm7.read_text(), re.DOTALL,
    )
    if not m:
        sys.exit(f"Cannot parse RESIDUE_LABEL section from {parm7}")

    labels = m.group(1).split()
    old_resnum = new_resnum = None
    for i, label in enumerate(labels, 1):
        if label == old_resname and old_resnum is None:
            old_resnum = i
        elif label == new_resname and new_resnum is None:
            new_resnum = i

    if old_resnum is None:
        sys.exit(
            f"Residue '{old_resname}' not found in {parm7}.\n"
            "Check that ligands.old.name matches the residue name in the PDB."
        )
    if new_resnum is None:
        sys.exit(
            f"Residue '{new_resname}' not found in {parm7}.\n"
            "Check that ligands.new.name matches the residue name in the PDB."
        )

    timask1 = f'":{old_resnum}"'
    timask2 = f'":{new_resnum}"'

    def _scmask(resnum: int, names: list[str]) -> str:
        if not names:
            return '""'      # no soft-core atoms (pure charge/type perturbation)
        return f'":{resnum}@{",".join(names)}"'

    scmask1 = _scmask(old_resnum, unique_old_names)
    scmask2 = _scmask(new_resnum, unique_new_names)

    return timask1, timask2, scmask1, scmask2


# =============================================================================
# AMBER input templates
# =============================================================================

_MIN_TEMPLATE = """\
minimisation
 &cntrl
   imin = 1, ntmin = 2, maxcyc = {maxcyc},
   ntpr = {ntpr}, ntwe = 20, dx0 = 1.0D-7,
   ntb = 1, ntxo = 1,
   icfe = 1, ifsc = 1, clambda = 0.5, scalpha = 0.5, scbeta = 12.0,
   logdvdl = 0,
   timask1 = {timask1}, timask2 = {timask2},
   scmask1 = {scmask1}, scmask2 = {scmask2},
 /
"""

_HEATING_TEMPLATE = """\
NVT heating
 &cntrl
   nstlim = {nstlim}, irest = 0, ntx = 1, dt = {dt},
   nmropt = 1, ntt = 3, tempi = {tempi}, temp0 = {temp0},
   gamma_ln = 2.0, ig = -1,
   ntc = 1, ntf = 1, ntb = 1, ntp = 0,
   ntwe = {ntwx}, ntwx = {ntwx}, ntpr = {ntwx}, ntwr = {ntwx},
   icfe = 1, ifsc = 1, clambda = 0.5, scalpha = 0.5, scbeta = 12.0,
   logdvdl = 0,
   timask1 = {timask1}, timask2 = {timask2},
   scmask1 = {scmask1}, scmask2 = {scmask2},
 /
 &wt type='TEMP0', istep1=0,            istep2={ramp_end},    value1={tempi}, value2={temp0} /
 &wt type='TEMP0', istep1={ramp_end_p1}, istep2={nstlim},     value1={temp0}, value2={temp0} /
 &wt type='END' /
 /
"""

_EQUIL_TEMPLATE = """\
TI equilibration
 &cntrl
   imin = 0, nstlim = {nstlim}, irest = 0, ntx = 1, dt = {dt},
   ntt = 3, temp0 = {temp0}, gamma_ln = 2.0, ig = -1,
   ntc = 1, ntf = 1,
   ntb = 2, ntp = 1, pres0 = 1.01325, taup = 2.0, barostat = 2,
   ntwe = {ntwx}, ntwx = {ntwx}, ntpr = {ntwx}, ntwr = {ntwx},
   icfe = 1, ifsc = 1, clambda = 0.5, scalpha = 0.5, scbeta = 12.0,
   logdvdl = 0,
   timask1 = {timask1}, timask2 = {timask2},
   scmask1 = {scmask1}, scmask2 = {scmask2},
 /
"""

_PROD_TEMPLATE = """\
TI production   lambda = {clambda:.5f}
 &cntrl
   imin = 0, nstlim = {nstlim}, irest = 0, ntx = 1, dt = {dt},
   ntt = 3, temp0 = {temp0}, gamma_ln = 2.0, ig = -1,
   ntc = 1, ntf = 1,
   ntb = 2, ntp = 1, pres0 = 1.01325, taup = 2.0, barostat = 2,
   ntwe = {ntwe}, ntwx = {ntwx}, ntpr = {ntwe}, ntwr = {ntwe},
   icfe = 1, ifsc = 1, clambda = {clambda:.5f}, scalpha = 0.5, scbeta = 12.0,
   logdvdl = 0,
   timask1 = {timask1}, timask2 = {timask2},
   scmask1 = {scmask1}, scmask2 = {scmask2},{mbar_block}
 /
"""


# =============================================================================
# SLURM / local script helpers  (adapted from fep_runner.py)
# =============================================================================

def _sbatch_header(job_name: str, resources: dict) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=mpi_%j.out",
        "#SBATCH --error=mpi_%j.err",
        f"#SBATCH --ntasks={resources['ntasks']}",
    ]
    for k, v in resources.items():
        if k != "ntasks":
            lines.append(f"#SBATCH --{k}={v}")
    return "\n".join(lines)


def _module_block(module: str) -> str:
    return f"\nmodule purge\nmodule load {module}\n"


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def _write_exe(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _equil_cpptraj_params(cfg: dict, n_replicas: int) -> tuple[int, int, int]:
    equil = cfg["simulation"]["equil"]
    total = equil["nstlim"] // equil["ntwx"]
    start = total // 2
    if n_replicas == 1:
        return total, total, 1
    step = (total - start) // (n_replicas - 1)
    return start, total, step


def _gen_equilibration_cmd(
    leg: str, n_replicas: int, mid: int, cfg: dict, mode: str
) -> str:
    res   = cfg["slurm"]["gpu"]
    gpu   = cfg["execution_command"]["gpu"]
    start, end, step = _equil_cpptraj_params(cfg, n_replicas)

    lines = [
        _sbatch_header(f"equil-{leg}", res),
        _module_block(cfg["amber"]["cuda_module"]),
        "# ---- Minimisation",
        f"{gpu} -i min.in -c ti.rst7 -p ti.parm7 -O \\",
        "    -o min.out -inf min.info -e min.en -r min.rst7 -l min.log",
        "",
        "# ---- NVT heating",
        f"{gpu} -i heating.in -c min.rst7 -p ti.parm7 -O \\",
        "    -o heating.out -inf heating.info -e heating.en -r heating.rst7 -x heating.nc -l heating.log",
        "",
        "# ---- Equilibration",
        f"{gpu} -i equil.in -c heating.rst7 -p ti.parm7 -O \\",
        "    -o equil.out -inf equil.info -e equil.en -r equil.rst7 -x equil.nc -l equil.log",
        "",
        f"# Extract {n_replicas} restart(s) from the second half of equilibration",
    ]
    for r in range(1, n_replicas + 1):
        frame = start + (r - 1) * step if n_replicas > 1 else end
        lines += [
            "cpptraj <<_EOF",
            "parm ti.parm7",
            f"trajin equil.nc {frame} {frame} 1",
            f"trajout equil.rst7.{r} restart",
            "_EOF", "",
        ]

    top = "top=$(pwd)"
    if mode == "parallel":
        lines += [
            top,
            f"for r in $(seq 1 {n_replicas}); do",
            f'    (cd "$top/replica_${{r}}/{mid}" && sbatch FEP_PROD_{mid}.cmd)',
            "done", "",
        ]
    else:
        lines += [top, f"cd $top/replica_1/{mid}", f"sbatch FEP_PROD_{mid}.cmd", ""]

    return "\n".join(lines)


def _prod_submissions(window: int, replica: int,
                      n_windows: int, n_replicas: int,
                      mode: str) -> list[tuple[str, str]]:
    mid = _middle(n_windows)
    if mode == "parallel":
        if window == mid:
            return [(f"../{mid-1}", f"FEP_PROD_{mid-1}.cmd"),
                    (f"../{mid+1}", f"FEP_PROD_{mid+1}.cmd")]
        if 1 < window < mid:
            return [(f"../{window-1}", f"FEP_PROD_{window-1}.cmd")]
        if mid < window < n_windows:
            return [(f"../{window+1}", f"FEP_PROD_{window+1}.cmd")]
        return []
    # serial
    if window == mid:
        return [(f"../{mid-1}", f"FEP_PROD_{mid-1}.cmd")]
    if 1 < window < mid:
        return [(f"../{window-1}", f"FEP_PROD_{window-1}.cmd")]
    if window == 1:
        return [(f"../{mid+1}", f"FEP_PROD_{mid+1}.cmd")]
    if mid < window < n_windows:
        return [(f"../{window+1}", f"FEP_PROD_{window+1}.cmd")]
    if replica < n_replicas:
        return [(f"../../replica_{replica+1}/{mid}", f"FEP_PROD_{mid}.cmd")]
    return []


def _gen_prod_cmd(
    window: int, replica: int, n_windows: int, n_replicas: int,
    leg: str, cfg: dict, mode: str
) -> str:
    mid   = _middle(n_windows)
    res   = cfg["slurm"]["gpu"]
    gpu   = cfg["execution_command"]["gpu"]

    coords = (
        f"../../equil.rst7.{replica}" if window == mid
        else f"../{window+1}/ti{replica}_{window+1}.rst7" if window < mid
        else f"../{window-1}/ti{replica}_{window-1}.rst7"
    )

    subs  = _prod_submissions(window, replica, n_windows, n_replicas, mode)
    lines = [
        _sbatch_header(f"R{replica}.{window}-{leg}", res),
        _module_block(cfg["amber"]["cuda_module"]),
        f"{gpu} -i ti_{window}.in -c {coords} -p ti.parm7 -O \\",
        f"    -o ti{replica}_{window}.out -inf ti{replica}_{window}.info "
        f"-e ti{replica}_{window}.en \\",
        f"    -r ti{replica}_{window}.rst7 -x ti{replica}_{window}.nc "
        f"-l ti{replica}_{window}.log",
        "",
    ]
    if len(subs) > 1:
        for d, c in subs:
            lines.append(f"(cd {d} && sbatch {c})")
        lines.append("")
    elif subs:
        d, c = subs[0]
        lines += [f"cd {d}", f"sbatch {c}", ""]

    return "\n".join(lines)


# =============================================================================
# Simulation input generation
# =============================================================================

def generate_leg_inputs(
    leg: str,
    prep_parm7: Path, prep_rst7: Path,
    timask1: str, timask2: str, scmask1: str, scmask2: str,
    lambdas: np.ndarray, weights: np.ndarray,
    cfg: dict, mode: str,
) -> None:
    """
    Write all AMBER input files and job scripts for one leg (unbound / bound).

    Directory layout mirrors fep_runner.py so the same analyse command works.
    """
    mutation_dir = Path(
        f"{cfg['ligands']['old']['name']}_to_{cfg['ligands']['new']['name']}"
    )
    leg_dir = mutation_dir / leg
    leg_dir.mkdir(parents=True, exist_ok=True)

    sim   = cfg["simulation"]
    temp  = float(cfg["temperature"])
    n_lam = len(lambdas)
    n_rep = cfg["replicates"]
    mid   = _middle(n_lam)
    mbar  = bool(cfg.get("mbar", True))

    mbar_blk  = _format_mbar_block(lambdas) if mbar else ""
    mask_kw   = dict(timask1=timask1, timask2=timask2,
                     scmask1=scmask1, scmask2=scmask2)

    # Symlinks to topology and coordinates at the leg level
    _symlink(prep_parm7.resolve(), leg_dir / "ti.parm7")
    _symlink(prep_rst7.resolve(),  leg_dir / "ti.rst7")

    # Shared stage inputs -------------------------------------------------------
    (leg_dir / "min.in").write_text(
        _MIN_TEMPLATE.format(
            maxcyc=sim["min"]["maxcyc"], ntpr=sim["min"]["ntpr"], **mask_kw)
    )
    _heat = sim["heating"]
    ramp  = int(0.8 * _heat["nstlim"])
    (leg_dir / "heating.in").write_text(
        _HEATING_TEMPLATE.format(
            nstlim=_heat["nstlim"], dt=_heat["dt"], ntwx=_heat["ntwx"],
            tempi=_heat["tempi"], temp0=temp,
            ramp_end=ramp, ramp_end_p1=ramp + 1, **mask_kw)
    )
    (leg_dir / "equil.in").write_text(
        _EQUIL_TEMPLATE.format(
            nstlim=sim["equil"]["nstlim"], dt=sim["equil"]["dt"],
            ntwx=sim["equil"]["ntwx"], temp0=temp, **mask_kw)
    )

    # Mode-specific job scripts -------------------------------------------------
    if mode == "local":
        _write_exe(leg_dir / "run_local.sh",
                   _gen_local_script(leg, n_lam, n_rep, lambdas, cfg,
                                     timask1, timask2, scmask1, scmask2))
    else:
        _write_exe(leg_dir / "EQUILIBRATION.cmd",
                   _gen_equilibration_cmd(leg, n_rep, mid, cfg, mode))

    # Per-replica / per-window --------------------------------------------------
    for replica in range(1, n_rep + 1):
        for w_idx, clambda in enumerate(lambdas, 1):
            win_dir = leg_dir / f"replica_{replica}" / str(w_idx)
            win_dir.mkdir(parents=True, exist_ok=True)
            _symlink(prep_parm7.resolve(), win_dir / "ti.parm7")

            (win_dir / f"ti_{w_idx}.in").write_text(
                _PROD_TEMPLATE.format(
                    nstlim=sim["prod"]["nstlim"], dt=sim["prod"]["dt"],
                    ntwe=sim["prod"]["ntwe"], ntwx=sim["prod"]["ntwx"],
                    clambda=clambda, temp0=temp,
                    mbar_block=mbar_blk, **mask_kw)
            )

            if mode != "local":
                _write_exe(
                    win_dir / f"FEP_PROD_{w_idx}.cmd",
                    _gen_prod_cmd(w_idx, replica, n_lam, n_rep,
                                  leg, cfg, mode),
                )


def _gen_local_script(
    leg: str, n_windows: int, n_replicas: int,
    lambdas: np.ndarray, cfg: dict,
    timask1: str, timask2: str, scmask1: str, scmask2: str,
) -> str:
    mid  = _middle(n_windows)
    sim  = cfg["simulation"]
    temp = cfg["temperature"]
    start, end, step = _equil_cpptraj_params(cfg, n_replicas)
    order = list(range(mid, 0, -1)) + list(range(mid + 1, n_windows + 1))

    cuda_lib = cfg.get("amber", {}).get("cuda_lib_path", "").strip()

    L = [
        "#!/bin/bash",
        f"# Sequential TI run for {leg} — no SLURM required.",
        "set -euo pipefail",
        'SYSDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$SYSDIR"',
        ': "${AMBERHOME:?Set AMBERHOME before running}"',
        'AMBER="$AMBERHOME/bin/pmemd.cuda"',
        'CPPTRAJ="$AMBERHOME/bin/cpptraj"',
        "",
    ]
    if cuda_lib:
        L.append(f'export LD_LIBRARY_PATH="{cuda_lib}:${{LD_LIBRARY_PATH:-}}"')

    L += [
        'log() { echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*"; }',
        "",
        'log "Minimisation"',
        '$AMBER -i min.in -c ti.rst7 -p ti.parm7 -O \\',
        '    -o min.out -inf min.info -e min.en -r min.rst7 -l min.log',
        "",
        'log "Heating"',
        '$AMBER -i heating.in -c min.rst7 -p ti.parm7 -O \\',
        '    -o heating.out -inf heating.info -e heating.en -r heating.rst7 -x heating.nc -l heating.log',
        "",
        'log "Equilibration"',
        '$AMBER -i equil.in -c heating.rst7 -p ti.parm7 -O \\',
        '    -o equil.out -inf equil.info -e equil.en -r equil.rst7 -x equil.nc -l equil.log',
        "",
    ]
    for r in range(1, n_replicas + 1):
        frame = start + (r - 1) * step if n_replicas > 1 else end
        L += [
            '$CPPTRAJ <<_EOF',
            "parm ti.parm7",
            f"trajin equil.nc {frame} {frame} 1",
            f"trajout equil.rst7.{r} restart",
            "_EOF", "",
        ]
    for replica in range(1, n_replicas + 1):
        L.append(f'log "Replica {replica}/{n_replicas}"')
        for window in order:
            clambda = lambdas[window - 1]
            coords = (
                f"../../equil.rst7.{replica}" if window == mid
                else f"../{window+1}/ti{replica}_{window+1}.rst7" if window < mid
                else f"../{window-1}/ti{replica}_{window-1}.rst7"
            )
            L += [
                f'log "  window {window}/{n_windows}  lambda={clambda:.5f}"',
                f'cd "$SYSDIR/replica_{replica}/{window}"',
                f'$AMBER -i ti_{window}.in -c {coords} -p ti.parm7 -O \\',
                f'    -o ti{replica}_{window}.out -inf ti{replica}_{window}.info \\',
                f'    -e ti{replica}_{window}.en   -r ti{replica}_{window}.rst7 \\',
                f'    -x ti{replica}_{window}.nc   -l ti{replica}_{window}.log',
                "",
            ]
    L.append('log "All done."')
    return "\n".join(L)


# =============================================================================
# Analysis  (TI + MBAR — same logic as fep_runner.py)
# =============================================================================

def _extract_dvdl(en_file: Path, tail: int) -> np.ndarray:
    values: list[float] = []
    with open(en_file) as fh:
        for line in fh:
            if line.startswith(" L9") or line.startswith("L9"):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        values.append(float(parts[5]))
                    except ValueError:
                        pass
    if not values:
        raise RuntimeError(f"No L9 records in {en_file}")
    return np.array(values[-tail:])


def _extract_mbar_energies(en_file: Path, n_states: int) -> np.ndarray:
    frames: list[list[float]] = []
    with open(en_file) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("MBAR") and len(s) > 4:
                parts = s.split()
                try:
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                if len(vals) == n_states:
                    frames.append(vals)
    if not frames:
        raise RuntimeError(
            f"No MBAR records in {en_file}.\n"
            "  Ensure the simulation was run with mbar: true in the config."
        )
    return np.array(frames)


def _ti_system_dg(
    leg: str, base: Path, n_replicas: int, n_lambdas: int,
    weights: np.ndarray, tail: int,
) -> tuple[float, float]:
    replica_dg: list[float] = []
    for replica in range(1, n_replicas + 1):
        means = [
            _extract_dvdl(
                base / leg / f"replica_{replica}" / str(w) /
                f"ti{replica}_{w}.en", tail
            ).mean()
            for w in range(1, n_lambdas + 1)
        ]
        dg = float(np.dot(means, weights))
        print(f"    replica {replica}: ΔG(TI) = {dg:9.3f} kcal/mol")
        replica_dg.append(dg)
    mean = float(np.mean(replica_dg))
    std  = float(np.std(replica_dg))
    print(f"    mean          : ΔG(TI) = {mean:9.3f} ± {std:.3f} kcal/mol")
    return mean, std


def _mbar_system_dg(
    leg: str, base: Path, n_replicas: int, n_lambdas: int,
    tail: int, temp: float,
) -> tuple[float, float]:
    try:
        from pymbar import MBAR
    except ImportError:
        raise ImportError("pymbar >= 4 required:  conda install -c conda-forge pymbar")

    beta     = _mbar_beta(temp)
    n_states = n_lambdas + 2   # GL nodes + λ=0 + λ=1 endpoints

    replica_dg: list[float] = []
    for replica in range(1, n_replicas + 1):
        window_E: list[np.ndarray] = []
        for w in range(1, n_lambdas + 1):
            en = (base / leg / f"replica_{replica}" / str(w) /
                  f"ti{replica}_{w}.en")
            window_E.append(_extract_mbar_energies(en, n_states))

        n_frames = min(min(E.shape[0], tail) for E in window_E)
        window_E = [E[-n_frames:] for E in window_E]

        N_k = np.zeros(n_states, dtype=int)
        N_k[1:n_lambdas + 1] = n_frames

        u_kn = np.empty((n_states, n_lambdas * n_frames))
        for wi, E_w in enumerate(window_E):
            c = wi * n_frames
            u_kn[:, c:c + n_frames] = beta * E_w.T

        result = MBAR(u_kn, N_k).compute_free_energy_differences()
        dg  = float(result["Delta_f"][0, -1])  / beta
        ddg = float(result["dDelta_f"][0, -1]) / beta
        print(f"    replica {replica}: ΔG(MBAR) = {dg:9.3f} ± {ddg:.3f} kcal/mol")
        replica_dg.append(dg)

    mean = float(np.mean(replica_dg))
    std  = float(np.std(replica_dg))
    print(f"    mean          : ΔG(MBAR) = {mean:9.3f} ± {std:.3f} kcal/mol")
    return mean, std


# =============================================================================
# Top-level commands
# =============================================================================

def prepare(cfg: dict, mode: str = "serial", dry_run: bool = False,
            skip_param: bool = False) -> None:
    """
    Full preparation pipeline:
      1. Parameterise both ligands.
      2. Compute FE-aware atom mapping; write mapping.json.
      3. Detect net charge change; warn if ΔQ ≠ 0.
      4. Build solvated systems with tleap (unbound + bound legs).
      5. Derive TI masks from the parm7 RESIDUE_LABEL section.
      6. Write AMBER input files and job scripts for all λ windows.
    """
    check_environment()
    validate_config(cfg)

    old_cfg  = cfg["ligands"]["old"]
    new_cfg  = cfg["ligands"]["new"]
    old_name = old_cfg["name"]
    new_name = new_cfg["name"]
    pdb_old = Path(old_cfg["pdb"])
    pdb_new = Path(new_cfg["pdb"])

    mutation = f"{old_name}_to_{new_name}"
    prep_dir = Path(mutation) / "prep"

    lambdas, weights = compute_gl_quadrature(cfg["n_lambdas"])
    mid  = _middle(cfg["n_lambdas"])
    mbar = bool(cfg.get("mbar", True))

    print(f"\n{'═' * 62}")
    print(f"  Mutation : {mutation}")
    print(f"  Mode     : {mode}   Windows : {cfg['n_lambdas']}   "
          f"Replicas : {cfg['replicates']}")
    print(f"  Temp     : {cfg['temperature']} K")
    print(f"  MBAR     : {'enabled' if mbar else 'disabled'}")
    if dry_run:
        print("  DRY RUN — no files will be written.")
    print(f"{'═' * 62}\n")

    # ── Step 1: parameterise ligands ─────────────────────────────────────
    print("[1/6] Parameterising ligands...")

    def _param_dir(name: str) -> Path:
        return Path("parameters") / name

    if skip_param:
        old_struct = Path(old_cfg["lib"])    if "lib"    in old_cfg else _param_dir(old_name) / f"{old_name}.lib"
        old_frcmod = Path(old_cfg["frcmod"]) if "frcmod" in old_cfg else _param_dir(old_name) / f"{old_name}.frcmod"
        new_struct = Path(new_cfg["lib"])    if "lib"    in new_cfg else _param_dir(new_name) / f"{new_name}.lib"
        new_frcmod = Path(new_cfg["frcmod"]) if "frcmod" in new_cfg else _param_dir(new_name) / f"{new_name}.frcmod"
        for f in (old_struct, old_frcmod, new_struct, new_frcmod):
            if not f.exists():
                sys.exit(f"  --skip-param requested but file not found: {f}")
        print("  Skipping — using existing parameter files.")
    elif not dry_run:
        old_struct, old_frcmod = parameterize_ligand(pdb_old, old_name, cfg)
        new_struct, new_frcmod = parameterize_ligand(pdb_new, new_name, cfg)
    else:
        old_struct = _param_dir(old_name) / f"{old_name}.lib"
        old_frcmod = _param_dir(old_name) / f"{old_name}.frcmod"
        new_struct = _param_dir(new_name) / f"{new_name}.lib"
        new_frcmod = _param_dir(new_name) / f"{new_name}.frcmod"

    # ── Step 2: atom mapping ─────────────────────────────────────────────
    override_mapping = cfg.pop("_override_mapping", None)
    if override_mapping is not None:
        mapping = override_mapping
        print("\n[2/6] Using manually supplied atom mapping (--override-mapping).")
    else:
        print("\n[2/6] Computing FE-aware atom mapping...")
        mapping = compute_fe_mapping(pdb_old, pdb_new)

    print(f"  Common heavy atoms : {mapping['n_common_heavy']}")
    print(f"  Soft-core (old)    : {len(mapping['unique_old'])} atoms  "
          f"→ {mapping['unique_old']}")
    print(f"  Soft-core (new)    : {len(mapping['unique_new'])} atoms  "
          f"→ {mapping['unique_new']}")
    print(f"  Mapping FE score   : {mapping['score']:.1f}  (lower is better)")

    if not dry_run:
        map_path = prep_dir / "mapping.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(mapping, indent=2))
        print(f"\n  Mapping written to {map_path}")
        if override_mapping is None:
            print("  Review it and re-run with --override-mapping if needed.")

    # ── Step 3: charge check ─────────────────────────────────────────────
    print("\n[3/6] Checking formal charges...")
    q_old, q_new = detect_formal_charges(pdb_old, pdb_new)
    delta_q = q_new - q_old
    print(f"  Q(old) = {q_old:+d}    Q(new) = {q_new:+d}    ΔQ = {delta_q:+d}")
    if delta_q != 0:
        print(
            f"\n  WARNING: ΔQ = {delta_q:+d} — the net charge changes between end states.\n"
            "  The ion count in the topology neutralises the combined (old + new)\n"
            "  ligand charge. This is a pragmatic approximation; for rigorous\n"
            "  ΔQ ≠ 0 handling build separate topologies per endpoint (see\n"
            "  AMBER TI tutorial 3 / Rocklin et al. JCTC 2013)."
        )

    # ── Step 4: build systems ────────────────────────────────────────────
    print("\n[4/6] Building solvated systems with tleap...")
    has_protein = "protein" in cfg

    if has_protein:
        print(f"  Checking protein PDB: {cfg['protein']['pdb']}")
        if not dry_run:
            check_pdb(Path(cfg["protein"]["pdb"]))

    legs = (["unbound", "bound"] if has_protein else ["unbound"])
    parm7: dict[str, Path] = {}
    rst7:  dict[str, Path] = {}

    for leg in legs:
        print(f"\n  Building {leg} system...")
        if not dry_run:
            p7, r7 = build_system(
                leg, old_struct, old_frcmod, new_struct, new_frcmod,
                old_name, new_name, q_old, q_new, cfg,
                prep_dir / leg,
            )
            parm7[leg] = p7
            rst7[leg]  = r7
        else:
            parm7[leg] = prep_dir / leg / "ti.parm7"
            rst7[leg]  = prep_dir / leg / "ti.rst7"

    # ── Step 5: derive masks ─────────────────────────────────────────────
    print("\n[5/6] Deriving AMBER TI masks...")
    masks: dict[str, tuple] = {}

    for leg in legs:
        if dry_run:
            masks[leg] = (f'":{old_name}"', f'":{new_name}"',
                          f'"soft-core-old"', f'"soft-core-new"')
            print(f"  [{leg}] (dry run — masks not derived)")
            continue

        tm1, tm2, sm1, sm2 = derive_masks(
            parm7[leg], old_name, new_name,
            mapping["unique_old"], mapping["unique_new"],
        )
        masks[leg] = (tm1, tm2, sm1, sm2)
        print(f"  [{leg}]")
        print(f"    timask1 = {tm1}")
        print(f"    timask2 = {tm2}")
        print(f"    scmask1 = {sm1}")
        print(f"    scmask2 = {sm2}")

    # ── Step 6: write simulation inputs ──────────────────────────────────
    print(f"\n[6/6] Writing simulation inputs ({mode} mode)...")
    if not dry_run:
        for leg in legs:
            tm1, tm2, sm1, sm2 = masks[leg]
            generate_leg_inputs(
                leg, parm7[leg], rst7[leg],
                tm1, tm2, sm1, sm2,
                lambdas, weights, cfg, mode,
            )
            print(f"  [{leg}] → {Path(mutation) / leg}/")

    print(f"\n{'═' * 62}")
    if dry_run:
        print("  Dry run complete — no files were written.")
    else:
        print("  Preparation complete.")
        print(f"  Submit with:  python rbfe_runner.py submit --mode {mode}")
    print(f"{'═' * 62}\n")


def submit(cfg: dict, mode: str = "serial") -> None:
    """Submit the equilibration job for each leg."""
    mutation = f"{cfg['ligands']['old']['name']}_to_{cfg['ligands']['new']['name']}"
    legs = (["unbound", "bound"] if "protein" in cfg else ["unbound"])

    if mode == "local":
        for leg in legs:
            script = Path(mutation) / leg / "run_local.sh"
            if not script.exists():
                sys.exit(f"Script not found: {script}\nRun 'prepare --mode local' first.")
            log = Path(mutation) / leg / "run.log"
            import subprocess as _sp
            proc = _sp.Popen(
                f"bash {script.resolve()} > {log.resolve()} 2>&1",
                shell=True, start_new_session=True,
            )
            print(f"  [{leg}] launched in background (PID {proc.pid}) → {log}")
        return

    for leg in legs:
        leg_dir = Path(mutation) / leg
        cmd_file = leg_dir / "EQUILIBRATION.cmd"
        if not cmd_file.exists():
            sys.exit(f"Script not found: {cmd_file}\nRun 'prepare' first.")
        import subprocess as _sp
        res = _sp.run(
            ["sbatch", "EQUILIBRATION.cmd"],
            capture_output=True, text=True, cwd=str(leg_dir),
        )
        if res.returncode == 0:
            print(f"  [{leg}] {res.stdout.strip()}")
        else:
            print(f"  [{leg}] sbatch failed: {res.stderr.strip()}", file=sys.stderr)


def analyse(cfg: dict, tail: int = 4000) -> None:
    """Compute ΔΔG = ΔG(bound) − ΔG(unbound) via TI and, if enabled, MBAR."""
    validate_config(cfg)

    mutation = f"{cfg['ligands']['old']['name']}_to_{cfg['ligands']['new']['name']}"
    base     = Path(mutation)
    n_lam    = cfg["n_lambdas"]
    n_rep    = cfg["replicates"]
    temp     = float(cfg["temperature"])
    mbar     = bool(cfg.get("mbar", True))
    _, weights = compute_gl_quadrature(n_lam)
    legs     = (["unbound", "bound"] if "protein" in cfg else ["unbound"])

    print(f"\n{'═' * 62}")
    print(f"  Mutation : {mutation}")
    print(f"  Windows  : {n_lam}    Replicas : {n_rep}")
    print(f"  Temp     : {temp} K    β = {_mbar_beta(temp):.4f} mol/kcal")
    print(f"  Records  : last {tail} dV/dλ frames per window")
    if mbar:
        print("  MBAR     : enabled")
    print(f"{'═' * 62}")

    ti_results:   dict[str, tuple[float, float]] = {}
    mbar_results: dict[str, tuple[float, float]] = {}

    for leg in legs:
        print(f"\n  [{leg}]  TI (Gauss-Legendre)")
        try:
            ti_results[leg] = _ti_system_dg(
                leg, base, n_rep, n_lam, weights, tail)
        except (FileNotFoundError, RuntimeError) as exc:
            sys.exit(f"  ERROR: {exc}")

        if mbar:
            print(f"\n  [{leg}]  MBAR")
            try:
                mbar_results[leg] = _mbar_system_dg(
                    leg, base, n_rep, n_lam, tail, temp)
            except (FileNotFoundError, RuntimeError, ImportError) as exc:
                print(f"  WARNING: MBAR failed — {exc}")

    print(f"\n{'═' * 62}")
    if "bound" in ti_results and "unbound" in ti_results:
        print("  ΔΔG = ΔG(bound) − ΔG(unbound)")
        ub_ti, b_ti = ti_results["unbound"], ti_results["bound"]
        ddG_ti = b_ti[0] - ub_ti[0]
        err_ti = (b_ti[1] ** 2 + ub_ti[1] ** 2) ** 0.5
        print(f"\n  TI / Gauss-Legendre:")
        print(f"    ΔΔG = {ddG_ti:+.3f} ± {err_ti:.3f} kcal/mol")

        if "bound" in mbar_results and "unbound" in mbar_results:
            ub_mb, b_mb = mbar_results["unbound"], mbar_results["bound"]
            ddG_mb = b_mb[0] - ub_mb[0]
            err_mb = (b_mb[1] ** 2 + ub_mb[1] ** 2) ** 0.5
            print(f"\n  MBAR (pymbar):")
            print(f"    ΔΔG = {ddG_mb:+.3f} ± {err_mb:.3f} kcal/mol")
    else:
        # Single-leg run (ligand hydration FEP)
        for leg, (dg, std) in ti_results.items():
            print(f"  [{leg}]  ΔG(TI) = {dg:+.3f} ± {std:.3f} kcal/mol")

    print(f"{'═' * 62}\n")

    return {"ti": ti_results, "mbar": mbar_results}


# =============================================================================
# Perturbation network  (high-throughput mode)
# =============================================================================

_SOLVENT_RESNAMES = {"HOH", "WAT", "SOL", "TIP", "T3P", "Na+", "Cl-", "NA", "CL"}


def _read_resname_from_pdb(pdb: Path) -> str:
    """Read residue name from the first non-solvent HETATM or ATOM record."""
    with open(pdb) as fh:
        for line in fh:
            if line.startswith(("HETATM", "ATOM  ")):
                resname = line[17:20].strip()
                if resname and resname not in _SOLVENT_RESNAMES:
                    return resname
    raise RuntimeError(
        f"No valid residue name found in {pdb}. "
        "Ensure the file contains HETATM/ATOM records with a residue name."
    )


def _stem_to_resname(pdb: Path) -> str:
    """
    Derive a unique AMBER-safe residue name (≤ 4 chars) from a PDB filename stem.

    AMBER residue names may not start with a digit, so a leading 'L' is prepended
    when the stem starts with a number (e.g. '7g' → 'L7G', '12b' → 'L12B').
    The result is always uppercased and truncated to 4 characters.
    """
    stem = pdb.stem.upper()
    if stem and stem[0].isdigit():
        stem = "L" + stem
    return stem[:4]


def _kruskal_mst(
    n: int, edges: list[tuple[float, int, int]]
) -> list[tuple[int, int, float]]:
    """
    Kruskal's MST via union-find with path compression.

    edges : list of (weight, i, j) with i < j
    Returns list of (i, j, weight) MST edges.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    mst: list[tuple[int, int, float]] = []
    for w, i, j in sorted(edges):
        if union(i, j):
            mst.append((i, j, w))
        if len(mst) == n - 1:
            break
    return mst


def discover_ligands(
    structures_dir: Path, protein_stem: str = "protein"
) -> list[Path]:
    """
    Return all .pdb files in structures_dir sorted alphabetically,
    excluding the file whose stem matches protein_stem (case-insensitive).
    """
    pdbs = sorted(
        p for p in structures_dir.glob("*.pdb")
        if p.stem.lower() != protein_stem.lower()
    )
    if len(pdbs) < 2:
        sys.exit(
            f"Need at least 2 ligand PDB files in {structures_dir} "
            f"(found {len(pdbs)} after excluding '{protein_stem}.pdb')."
        )
    return pdbs


def build_perturbation_network(pdbs: list[Path]) -> dict:
    """
    Build a minimum spanning tree over all ligands.

    Similarity metric : Morgan radius-2 Tanimoto on heavy atoms (ECFP4-style).
    Edge weight       : 1 − Tanimoto  (lower = more similar = preferred by MST).
    Edge ordering     : MST edges sorted by ascending average heavy-atom count
                        of their endpoints (small → large traversal).

    Returns a dict with keys 'ligands' and 'edges' (written to network.json).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs, rdDetermineBonds
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        sys.exit("rdkit required:  conda install -c conda-forge rdkit")

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    n = len(pdbs)
    names:   list[str] = []
    heavies: list[int] = []
    fps:     list      = []

    print(f"  Loading {n} ligands and computing fingerprints...")
    for pdb in pdbs:
        # Use filename stem as the AMBER residue name so each ligand is unique
        # even when PDB files share the same residue name (e.g. all labelled INH).
        names.append(_stem_to_resname(pdb))

        mol_h = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
        if mol_h is None:
            sys.exit(f"RDKit could not parse {pdb}")
        charge = _infer_net_charge(pdb)
        rdDetermineBonds.DetermineBonds(mol_h, charge=charge)
        Chem.SanitizeMol(mol_h)
        mol = Chem.RemoveHs(mol_h)

        heavies.append(mol.GetNumAtoms())
        fps.append(mfpgen.GetFingerprint(mol))

    n_pairs = n * (n - 1) // 2
    print(f"  Computing {n_pairs} pairwise Tanimoto similarities...")
    sim: np.ndarray = np.zeros((n, n))
    raw_edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            sim[i, j] = sim[j, i] = s
            raw_edges.append((1.0 - s, i, j))

    mst = _kruskal_mst(n, raw_edges)
    mst.sort(key=lambda e: (heavies[e[0]] + heavies[e[1]]) / 2)

    ligands = [
        {"name": names[i], "pdb": str(pdbs[i]), "n_heavy": heavies[i]}
        for i in range(n)
    ]
    edges = [
        {
            "old":        names[i],
            "old_pdb":    str(pdbs[i]),
            "new":        names[j],
            "new_pdb":    str(pdbs[j]),
            "similarity": round(float(sim[i, j]), 4),
        }
        for i, j, _ in mst
    ]
    return {"ligands": ligands, "edges": edges}


def _print_ascii_tree(network: dict) -> None:
    """Render the MST as an ASCII tree rooted at the smallest ligand."""
    adj: dict[str, list[tuple[str, float]]] = {
        lig["name"]: [] for lig in network["ligands"]
    }
    for edge in network["edges"]:
        adj[edge["old"]].append((edge["new"], edge["similarity"]))
        adj[edge["new"]].append((edge["old"], edge["similarity"]))

    n_heavy: dict[str, int] = {
        lig["name"]: lig["n_heavy"] for lig in network["ligands"]
    }
    root = min(network["ligands"], key=lambda x: x["n_heavy"])["name"]

    def _walk(node: str, parent: str | None, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        sim_str = ""
        if parent is not None:
            for nb, sim in adj[parent]:
                if nb == node:
                    sim_str = f"  [sim={sim:.3f}]"
                    break
        print(f"{prefix}{connector}{node} ({n_heavy[node]}){sim_str}")
        children = sorted(
            [(nb, sim) for nb, sim in adj[node] if nb != parent],
            key=lambda x: n_heavy[x[0]],
        )
        ext = "    " if is_last else "│   "
        for k, (child, _) in enumerate(children):
            _walk(child, node, prefix + ext, k == len(children) - 1)

    print(f"\n  Transformation tree  (heavy atoms in parentheses):\n")
    print(f"  {root} ({n_heavy[root]})")
    children = sorted(adj[root], key=lambda x: n_heavy[x[0]])
    for k, (child, _) in enumerate(children):
        _walk(child, root, "  ", k == len(children) - 1)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive HTML network visualization
# ─────────────────────────────────────────────────────────────────────────────

def _sim_to_hex(sim: float) -> str:
    """Tanimoto → hex color: red (0.45) → yellow (0.65) → green (0.90+)."""
    t = max(0.0, min(1.0, (sim - 0.45) / 0.50))
    if t < 0.5:
        f = t * 2.0
        r, g, b = 210, int(40 + f * 175), 40
    else:
        f = (t - 0.5) * 2.0
        r, g, b = int(210 * (1 - f) + 40 * f), 215, 40
    return f"#{r:02x}{g:02x}{b:02x}"


def _set_2d_coords(
    mol,
    ref_mol=None,
) -> None:
    """
    Compute 2D coordinates for mol, optionally aligning to ref_mol via MCS.

    When ref_mol is provided the shared heavy atoms are pinned to ref_mol's 2D
    positions so the common scaffold appears in the same orientation across all
    depictions.  Falls back to an independent layout if the MCS is too small.
    """
    from rdkit.Chem import rdDepictor, rdFMCS
    from rdkit.Geometry.rdGeometry import Point2D

    if ref_mol is None or not ref_mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
        return

    mcs = rdFMCS.FindMCS(
        [ref_mol, mol],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        timeout=3,
    )
    if mcs.numAtoms < 4:
        rdDepictor.Compute2DCoords(mol)
        return

    from rdkit import Chem
    mcs_mol   = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    mol_match = mol.GetSubstructMatch(mcs_mol)
    if not ref_match or not mol_match:
        rdDepictor.Compute2DCoords(mol)
        return

    conf = ref_mol.GetConformer()
    coord_map = {
        mol_i: Point2D(conf.GetAtomPosition(ref_i).x,
                       conf.GetAtomPosition(ref_i).y)
        for ref_i, mol_i in zip(ref_match, mol_match)
    }
    rdDepictor.Compute2DCoords(mol, coordMap=coord_map)


def _mol_to_svg(mol, width: int = 160, height: int = 120) -> str:
    """Render mol to SVG. Caller must set 2D coordinates first via _set_2d_coords."""
    from rdkit.Chem.Draw import rdMolDraw2D
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
    try:
        opts.clearBackground = False
    except AttributeError:
        pass
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    idx = svg.find("<svg")
    return svg[idx:] if idx != -1 else svg


def _html_tree_layout(
    root:    str,
    adj:     dict[str, list[tuple[str, float]]],
    n_heavy: dict[str, int],
) -> dict[str, tuple[float, int]]:
    """
    Return {node: (float_x, int_depth)} grid positions for a top-down tree.
    x is in "slot units" (each leaf occupies 1 unit).
    """
    def _width(node: str, parent: str | None) -> int:
        kids = [nb for nb, _ in adj[node] if nb != parent]
        return max(1, sum(_width(k, node) for k in kids)) if kids else 1

    positions: dict[str, tuple[float, int]] = {}

    def _place(node: str, parent: str | None, x_left: float, depth: int) -> float:
        kids = sorted(
            [nb for nb, _ in adj[node] if nb != parent],
            key=lambda k: n_heavy[k],
        )
        if not kids:
            positions[node] = (x_left + 0.5, depth)
            return x_left + 1.0
        x = x_left
        for kid in kids:
            x = _place(kid, node, x, depth + 1)
        positions[node] = (
            (positions[kids[0]][0] + positions[kids[-1]][0]) / 2.0,
            depth,
        )
        return x

    _place(root, None, 0.0, 0)
    return positions


def visualize_network(
    network_path: Path = Path("network.json"),
    output_path:  Path = Path("network.html"),
    structures_dir: Path | None = None,
    cfg:          dict | None = None,
) -> None:
    """
    Generate a self-contained interactive HTML tree visualization of the MST.

    Reads network.json produced by 'network'.  If network.json is absent but
    structures_dir + cfg are supplied the network is built on-the-fly.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        sys.exit("rdkit required:  conda install -c conda-forge rdkit")

    # ── Load or build network ────────────────────────────────────────────────
    if network_path.exists():
        print(f"\n  Reading {network_path}...")
        network = json.loads(network_path.read_text())
    elif structures_dir is not None and cfg is not None:
        print("\n  network.json not found — building network from structures/...")
        protein_stem = Path(cfg.get("protein", {}).get("pdb", "protein.pdb")).stem
        pdbs         = discover_ligands(structures_dir.resolve(), protein_stem)
        network      = build_perturbation_network(pdbs)
    else:
        sys.exit(
            f"{network_path} not found.\n"
            "Run 'network' first, or pass --structures-dir to build on-the-fly."
        )

    n_heavy: dict[str, int] = {lig["name"]: lig["n_heavy"] for lig in network["ligands"]}

    # ── Load molecules ───────────────────────────────────────────────────────
    MOL_W, MOL_H = 160, 120
    print("  Loading molecules...")
    mols: dict[str, object] = {}
    for lig in network["ligands"]:
        pdb   = Path(lig["pdb"])
        mol_h = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
        if mol_h is None:
            sys.exit(f"RDKit could not parse {pdb}")
        charge = _infer_net_charge(pdb)
        rdDetermineBonds.DetermineBonds(mol_h, charge=charge)
        Chem.SanitizeMol(mol_h)
        mols[lig["name"]] = Chem.RemoveHs(mol_h)

    # ── Align 2D coords: root first, then all others pinned to root's scaffold
    root_tmp = min(network["ligands"], key=lambda x: x["n_heavy"])["name"]
    print("  Computing aligned 2D layouts...")
    _set_2d_coords(mols[root_tmp])                                # free layout for root
    for name, mol in mols.items():
        if name != root_tmp:
            _set_2d_coords(mol, ref_mol=mols[root_tmp])          # aligned to root

    # ── Render SVGs ──────────────────────────────────────────────────────────
    print("  Rendering molecule depictions...")
    mol_svgs: dict[str, str] = {
        name: _mol_to_svg(mol, MOL_W, MOL_H) for name, mol in mols.items()
    }

    # ── Adjacency & root ─────────────────────────────────────────────────────
    adj: dict[str, list[tuple[str, float]]] = {
        lig["name"]: [] for lig in network["ligands"]
    }
    for edge in network["edges"]:
        adj[edge["old"]].append((edge["new"], edge["similarity"]))
        adj[edge["new"]].append((edge["old"], edge["similarity"]))

    root = min(network["ligands"], key=lambda x: x["n_heavy"])["name"]

    # ── Pixel layout ─────────────────────────────────────────────────────────
    NODE_W, NODE_H = 174, 162   # node div outer size
    COL_W,  ROW_H  = 204, 240   # grid slot dimensions
    MARGIN         = 80

    grid = _html_tree_layout(root, adj, n_heavy)

    def to_px(gx: float, gy: int) -> tuple[float, float]:
        return (MARGIN + gx * COL_W - NODE_W / 2, MARGIN + gy * ROW_H)

    px_pos: dict[str, tuple[float, float]] = {
        name: to_px(gx, gy) for name, (gx, gy) in grid.items()
    }
    canvas_w = int(max(x + NODE_W for x, _ in px_pos.values()) + MARGIN)
    canvas_h = int(max(y + NODE_H for _, y in px_pos.values()) + MARGIN)

    # ── BFS to determine parent direction per edge ───────────────────────────
    parent: dict[str, str | None] = {root: None}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nb, _ in adj[node]:
            if nb not in parent:
                parent[nb] = node
                queue.append(nb)

    # ── SVG edge elements ─────────────────────────────────────────────────────
    arrow_defs = ""
    edge_elems = ""
    for k, edge in enumerate(network["edges"]):
        old_n, new_n = edge["old"], edge["new"]
        sim   = edge["similarity"]
        color = _sim_to_hex(sim)
        sw    = round(1.5 + sim * 2.5, 1)

        src, dst = (old_n, new_n) if parent.get(new_n) == old_n else (new_n, old_n)

        sx, sy = px_pos[src];  sx += NODE_W / 2;  sy += NODE_H      # bottom-center
        dx, dy = px_pos[dst];  dx += NODE_W / 2                      # top-center

        # cubic bezier
        gap = dy - sy
        c1x, c1y = sx, sy + gap * 0.45
        c2x, c2y = dx, dy - gap * 0.45
        mx,  my  = (sx + dx) / 2, (sy + dy) / 2

        arrow_defs += (
            f'\n    <marker id="a{k}" markerWidth="9" markerHeight="7" '
            f'refX="8" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 9 3.5, 0 7" fill="{color}"/></marker>'
        )
        edge_elems += f"""
  <g class="eg" data-tip="sim = {sim:.3f}  |  {src} → {dst}">
    <path d="M{sx:.1f},{sy:.1f} C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {dx:.1f},{dy:.1f}"
          stroke="transparent" stroke-width="14" fill="none"/>
    <path d="M{sx:.1f},{sy:.1f} C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {dx:.1f},{dy:.1f}"
          stroke="{color}" stroke-width="{sw}" fill="none" opacity="0.80"
          marker-end="url(#a{k})"/>
    <text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" dy="-5"
          style="font-size:11px;font-weight:700;fill:{color};pointer-events:none;">{sim:.3f}</text>
  </g>"""

    # ── Node divs ─────────────────────────────────────────────────────────────
    node_divs = ""
    for lig in network["ligands"]:
        name = lig["name"]
        x, y = px_pos[name]
        node_divs += (
            f'\n  <div class="node" style="left:{x:.1f}px;top:{y:.1f}px;'
            f'width:{NODE_W}px;">'
            f'\n    {mol_svgs[name]}'
            f'\n    <div class="nl">{name}</div>'
            f'\n    <div class="ns">{lig["n_heavy"]} heavy atoms'
            f' · {Path(lig["pdb"]).name}</div>'
            f'\n  </div>'
        )

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    n_lig   = len(network["ligands"])
    n_edges = len(network["edges"])
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RBFE Network</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#1a1a2e;font-family:'Segoe UI',Helvetica,sans-serif;overflow:hidden;color:#e0e0e0}}
#bar{{position:fixed;top:0;left:0;right:0;height:46px;background:#16213e;display:flex;
  align-items:center;padding:0 20px;gap:14px;z-index:100;
  box-shadow:0 2px 10px rgba(0,0,0,.6);font-size:13px}}
#bar b{{color:#a8d8ff}}
#bar .hint{{margin-left:auto;color:#666;font-size:11px}}
#leg{{position:fixed;bottom:16px;right:16px;background:rgba(22,33,62,.92);
  border:1px solid #333;border-radius:6px;padding:10px 14px;z-index:100;font-size:11px;color:#aaa}}
#leg h4{{font-size:12px;color:#ccc;margin-bottom:6px}}
.lr{{display:flex;align-items:center;gap:8px;margin-top:4px}}
#vp{{position:fixed;top:46px;left:0;right:0;bottom:0;overflow:hidden;cursor:grab}}
#vp.drag{{cursor:grabbing}}
#cv{{position:absolute;transform-origin:0 0}}
#esvg{{position:absolute;top:0;left:0;overflow:visible;pointer-events:none}}
.eg{{pointer-events:all;cursor:default}}
.eg:hover path:last-of-type{{opacity:1!important;stroke-width:5!important}}
.node{{position:absolute;background:#fff;border-radius:8px;border:2px solid #ccc;
  cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.4);
  transition:border-color .15s,box-shadow .15s,transform .1s;overflow:hidden;user-select:none}}
.node:hover{{border-color:#4a9eff;z-index:20;
  box-shadow:0 6px 20px rgba(74,158,255,.5);transform:translateY(-2px)}}
.node svg{{display:block;background:#fff}}
.nl{{text-align:center;font-size:12px;font-weight:700;color:#222;
  padding:3px 4px 1px;background:#f0f4ff;border-top:1px solid #dde}}
.ns{{text-align:center;font-size:10px;color:#888;padding:1px 4px 4px;background:#f0f4ff}}
#tt{{position:fixed;background:rgba(10,10,30,.92);color:#fff;padding:5px 10px;
  border-radius:4px;font-size:12px;pointer-events:none;opacity:0;
  transition:opacity .1s;z-index:200;white-space:nowrap;border:1px solid #444}}
</style>
</head>
<body>
<div id="bar">
  <b>RBFE Perturbation Network</b>
  <span style="color:#444">|</span>
  <span><b>{n_lig}</b> ligands · <b>{n_edges}</b> edges (MST)</span>
  <span style="color:#444">|</span>
  <span>Root: <b>{root}</b> ({n_heavy[root]} heavy atoms)</span>
  <span class="hint">Scroll = zoom &nbsp;·&nbsp; Drag = pan &nbsp;·&nbsp; Hover edge = similarity</span>
</div>
<div id="leg">
  <h4>Tanimoto similarity</h4>
  <div class="lr">
    <div style="width:70px;height:6px;background:linear-gradient(to right,#d22828,#d2a828,#28d228);border-radius:3px"></div>
    <span>low → high</span>
  </div>
  <div class="lr"><div style="width:14px;height:4px;background:{_sim_to_hex(0.50)};border-radius:2px"></div><span>0.50</span></div>
  <div class="lr"><div style="width:14px;height:4px;background:{_sim_to_hex(0.65)};border-radius:2px"></div><span>0.65</span></div>
  <div class="lr"><div style="width:14px;height:4px;background:{_sim_to_hex(0.80)};border-radius:2px"></div><span>0.80</span></div>
  <div class="lr"><div style="width:14px;height:4px;background:{_sim_to_hex(0.92)};border-radius:2px"></div><span>0.90+</span></div>
</div>
<div id="vp">
  <div id="cv" style="width:{canvas_w}px;height:{canvas_h}px">
    <svg id="esvg" width="{canvas_w}" height="{canvas_h}">
      <defs>{arrow_defs}
      </defs>
      {edge_elems}
    </svg>
    {node_divs}
  </div>
</div>
<div id="tt"></div>
<script>
const vp=document.getElementById('vp'),cv=document.getElementById('cv');
let sc=1,tx=0,ty=0,drag=false,lx=0,ly=0;
function upd(){{cv.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{sc}})`;}}
vp.addEventListener('wheel',e=>{{
  e.preventDefault();
  const d=e.deltaY>0?.9:1.1,r=vp.getBoundingClientRect(),
        mx=e.clientX-r.left,my=e.clientY-r.top;
  tx=mx-(mx-tx)*d; ty=my-(my-ty)*d;
  sc=Math.max(.15,Math.min(4,sc*d)); upd();
}},{{passive:false}});
vp.addEventListener('mousedown',e=>{{if(e.target.closest('.node'))return;drag=true;lx=e.clientX;ly=e.clientY;vp.classList.add('drag');}});
document.addEventListener('mousemove',e=>{{if(!drag)return;tx+=e.clientX-lx;ty+=e.clientY-ly;lx=e.clientX;ly=e.clientY;upd();}});
document.addEventListener('mouseup',()=>{{drag=false;vp.classList.remove('drag');}});
const tt=document.getElementById('tt');
document.querySelectorAll('.eg').forEach(el=>{{
  el.addEventListener('mouseenter',()=>{{tt.textContent=el.dataset.tip;tt.style.opacity=1;}});
  el.addEventListener('mousemove',e=>{{tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY-32)+'px';}});
  el.addEventListener('mouseleave',()=>{{tt.style.opacity=0;}});
}});
// fit to screen on load
const vw=vp.clientWidth,vh=vp.clientHeight,cw={canvas_w},ch={canvas_h};
sc=Math.min(.92,vw/cw,vh/ch); tx=(vw-cw*sc)/2; ty=(vh-ch*sc)/2; upd();
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\n  Visualization written → {output_path}")
    print(f"  Open with: open {output_path}")


def prepare_network(
    cfg:            dict,
    structures_dir: Path,
    mode:           str  = "serial",
    dry_run:        bool = False,
    skip_param:     bool = False,
) -> None:
    """
    High-throughput RBFE preparation.

    1. Discover all ligand PDBs in structures_dir (excluding the protein).
    2. Build MST perturbation network via Morgan fingerprint Tanimoto.
    3. Parameterise each unique ligand exactly once.
    4. Run the prepare() pipeline for every MST edge.
    """
    check_environment()
    validate_config(cfg, require_ligands=False)

    structures_dir = structures_dir.resolve()
    protein_stem   = Path(cfg.get("protein", {}).get("pdb", "protein.pdb")).stem
    pdbs           = discover_ligands(structures_dir, protein_stem=protein_stem)
    n              = len(pdbs)
    mbar           = bool(cfg.get("mbar", True))

    print(f"\n{'═' * 62}")
    print(f"  Network RBFE preparation")
    print(f"  Structures : {structures_dir}  ({n} ligands)")
    print(f"  Edges      : {n - 1}  (minimum spanning tree)")
    print(f"  Mode       : {mode}   Windows : {cfg['n_lambdas']}   "
          f"Replicas : {cfg['replicates']}")
    print(f"  Temp       : {cfg['temperature']} K")
    print(f"  MBAR       : {'enabled' if mbar else 'disabled'}")
    if dry_run:
        print("  DRY RUN — no files will be written.")
    print(f"{'═' * 62}\n")

    # ── Step 1: build network ────────────────────────────────────────────
    print("[1/3] Building perturbation network...")
    network = build_perturbation_network(pdbs)

    print(f"\n  {'Ligand':<8}  {'Heavy atoms':>11}  PDB")
    print(f"  {'─'*8}  {'─'*11}  {'─'*30}")
    for lig in sorted(network["ligands"], key=lambda x: x["n_heavy"]):
        print(f"  {lig['name']:<8}  {lig['n_heavy']:>11d}  {Path(lig['pdb']).name}")

    print(f"\n  MST edges — run order (smallest average size first):")
    print(f"  {'#':<4}  {'Old':<8}  {'New':<8}  {'Tanimoto':>8}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}")
    for k, edge in enumerate(network["edges"], 1):
        print(f"  {k:<4}  {edge['old']:<8}  {edge['new']:<8}  "
              f"{edge['similarity']:>8.3f}")

    _print_ascii_tree(network)

    if not dry_run:
        net_path = Path("network.json")
        net_path.write_text(json.dumps(network, indent=2))
        print(f"\n  Network written → {net_path}")

    # ── Step 2: parameterise each unique ligand once ─────────────────────
    print(f"\n[2/3] Parameterising {n} ligands...")

    def _param_dir(name: str) -> Path:
        return Path("parameters") / name

    param_cache: dict[str, tuple[Path, Path]] = {}

    for lig in sorted(network["ligands"], key=lambda x: x["n_heavy"]):
        name = lig["name"]
        pdb  = Path(lig["pdb"])
        if skip_param:
            lib    = _param_dir(name) / f"{name}.lib"
            frcmod = _param_dir(name) / f"{name}.frcmod"
            for f in (lib, frcmod):
                if not f.exists():
                    sys.exit(f"  --skip-param: file not found: {f}")
            param_cache[name] = (lib, frcmod)
            print(f"  {name}: using existing parameters")
        elif not dry_run:
            print(f"  Parameterising {name}...")
            lib, frcmod = parameterize_ligand(pdb, name, cfg)
            param_cache[name] = (lib, frcmod)
        else:
            param_cache[name] = (
                _param_dir(name) / f"{name}.lib",
                _param_dir(name) / f"{name}.frcmod",
            )

    # ── Step 3: prepare each edge ────────────────────────────────────────
    n_edges = len(network["edges"])
    print(f"\n[3/3] Preparing {n_edges} transformations...")

    for k, edge in enumerate(network["edges"], 1):
        old_name = edge["old"]
        new_name = edge["new"]
        mutation = f"{old_name}_to_{new_name}"
        print(f"\n{'─' * 62}")
        print(f"  Edge {k}/{n_edges}: {mutation}  "
              f"(Tanimoto = {edge['similarity']:.3f})")
        print(f"{'─' * 62}")

        old_lib, old_frcmod = param_cache[old_name]
        new_lib, new_frcmod = param_cache[new_name]
        edge_cfg = {
            **cfg,
            "ligands": {
                "old": {
                    "pdb":    edge["old_pdb"],
                    "name":   old_name,
                    "lib":    str(old_lib),
                    "frcmod": str(old_frcmod),
                },
                "new": {
                    "pdb":    edge["new_pdb"],
                    "name":   new_name,
                    "lib":    str(new_lib),
                    "frcmod": str(new_frcmod),
                },
            },
        }
        # skip_param=True only when files actually exist (not in dry-run)
        prepare(edge_cfg, mode=mode, dry_run=dry_run, skip_param=not dry_run)

    print(f"\n{'═' * 62}")
    if dry_run:
        print("  Network dry run complete — no files written.")
    else:
        print(f"  Network preparation complete: {n_edges} transformations ready.")
        print(f"  Submit:  python rbfe_runner.py network-submit --mode {mode}")
        print(f"  Analyse: python rbfe_runner.py network-analyse")
    print(f"{'═' * 62}\n")


def submit_network(cfg: dict, mode: str = "serial") -> None:
    """Submit all transformations recorded in network.json."""
    net_path = Path("network.json")
    if not net_path.exists():
        sys.exit("network.json not found — run 'network' first.")
    network = json.loads(net_path.read_text())

    for edge in network["edges"]:
        mutation = f"{edge['old']}_to_{edge['new']}"
        print(f"\n  ── {mutation} ──")
        edge_cfg = {
            **cfg,
            "ligands": {
                "old": {"pdb": edge["old_pdb"], "name": edge["old"]},
                "new": {"pdb": edge["new_pdb"], "name": edge["new"]},
            },
        }
        submit(edge_cfg, mode=mode)


def analyse_network(cfg: dict, tail: int = 4000) -> None:
    """Analyse all transformations in network.json; print a ΔΔG summary table."""
    net_path = Path("network.json")
    if not net_path.exists():
        sys.exit("network.json not found — run 'network' first.")
    network = json.loads(net_path.read_text())

    summary: list[dict] = []

    for edge in network["edges"]:
        mutation = f"{edge['old']}_to_{edge['new']}"
        print(f"\n{'─' * 62}")
        print(f"  {mutation}")
        print(f"{'─' * 62}")
        edge_cfg = {
            **cfg,
            "ligands": {
                "old": {"pdb": edge["old_pdb"], "name": edge["old"]},
                "new": {"pdb": edge["new_pdb"], "name": edge["new"]},
            },
        }
        try:
            result = analyse(edge_cfg, tail=tail)
            summary.append({"mutation": mutation, **(result or {})})
        except SystemExit as exc:
            print(f"  WARNING: {mutation} failed — {exc}")
            summary.append({"mutation": mutation, "error": str(exc)})

    print(f"\n{'═' * 62}")
    print("  Network ΔΔG Summary")
    print(f"{'═' * 62}")
    print(f"  {'Transformation':<26}  {'ΔΔG TI (kcal/mol)':>18}  "
          f"{'ΔΔG MBAR (kcal/mol)':>20}")
    print(f"  {'─'*26}  {'─'*18}  {'─'*20}")
    for row in summary:
        if "error" in row:
            print(f"  {row['mutation']:<26}  {'ERROR':>18}  {'ERROR':>20}")
            continue
        ti_str = mbar_str = "—"
        ti = row.get("ti", {})
        mb = row.get("mbar", {})
        if "bound" in ti and "unbound" in ti:
            b, ub = ti["bound"], ti["unbound"]
            ddg = b[0] - ub[0]
            err = (b[1] ** 2 + ub[1] ** 2) ** 0.5
            ti_str = f"{ddg:+.2f} ± {err:.2f}"
        if "bound" in mb and "unbound" in mb:
            b, ub = mb["bound"], mb["unbound"]
            ddg = b[0] - ub[0]
            err = (b[1] ** 2 + ub[1] ** 2) ** 0.5
            mbar_str = f"{ddg:+.2f} ± {err:.2f}"
        print(f"  {row['mutation']:<26}  {ti_str:>18}  {mbar_str:>20}")
    print(f"{'═' * 62}\n")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AmberTools-native RBFE workflow — automated atom mapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Config file (default: config.yaml)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Parameterise, map, build, write inputs")
    p_prep.add_argument("--mode", default="serial",
                        choices=["serial", "parallel", "local"])
    p_prep.add_argument("--dry-run", action="store_true",
                        help="Validate config and show what would be done; "
                             "do not run any external tools or write files.")
    p_prep.add_argument("--skip-param", action="store_true",
                        help="Skip antechamber/parmchk2/tleap and use existing "
                             "lib and frcmod files. Paths are read from the "
                             "ligands.{old,new}.lib/frcmod config keys, falling "
                             "back to parameters/{name}/{name}.lib/.frcmod.")
    p_prep.add_argument("--override-mapping",
                        help="Path to a manually edited mapping.json to use "
                             "instead of running MCS (skip steps 1-2).")

    p_sub = sub.add_parser("submit", help="Submit SLURM / local jobs")
    p_sub.add_argument("--mode", default="serial",
                       choices=["serial", "parallel", "local"])

    p_ana = sub.add_parser("analyse", help="TI + MBAR ΔΔG analysis")
    p_ana.add_argument("--tail", type=int, default=4000, metavar="N",
                       help="Last N dV/dλ records per window (default: 4000)")

    p_net = sub.add_parser(
        "network",
        help="High-throughput: build MST network and prepare all transformations",
    )
    p_net.add_argument(
        "--structures-dir", default="structures",
        help="Directory containing ligand PDB files (default: structures/)",
    )
    p_net.add_argument("--mode", default="serial",
                       choices=["serial", "parallel", "local"])
    p_net.add_argument("--dry-run", action="store_true",
                       help="Show network and mapping without writing files.")
    p_net.add_argument("--skip-param", action="store_true",
                       help="Reuse existing parameters/*/  lib and frcmod files.")

    p_netsub = sub.add_parser(
        "network-submit",
        help="Submit all jobs listed in network.json",
    )
    p_netsub.add_argument("--mode", default="serial",
                          choices=["serial", "parallel", "local"])

    p_netana = sub.add_parser(
        "network-analyse",
        help="Analyse all transformations in network.json; print ΔΔG table",
    )
    p_netana.add_argument("--tail", type=int, default=4000, metavar="N",
                          help="Last N dV/dλ records per window (default: 4000)")

    p_netvis = sub.add_parser(
        "network-visualize",
        help="Generate self-contained interactive HTML tree visualization",
    )
    p_netvis.add_argument("--network", default="network.json", metavar="PATH",
                          help="Path to network.json (default: network.json)")
    p_netvis.add_argument("--output", default="network.html", metavar="PATH",
                          help="Output HTML file (default: network.html)")
    p_netvis.add_argument("--structures-dir", default="structures", metavar="DIR",
                          help="Ligand PDB directory — used when network.json is absent")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.command == "prepare":
        if getattr(args, "override_mapping", None):
            with open(args.override_mapping) as fh:
                cfg["_override_mapping"] = json.load(fh)
        prepare(cfg, mode=args.mode, dry_run=args.dry_run,
                skip_param=args.skip_param)
    elif args.command == "submit":
        submit(cfg, mode=args.mode)
    elif args.command == "analyse":
        analyse(cfg, tail=args.tail)
    elif args.command == "network":
        prepare_network(
            cfg,
            structures_dir=Path(args.structures_dir),
            mode=args.mode,
            dry_run=args.dry_run,
            skip_param=args.skip_param,
        )
    elif args.command == "network-submit":
        submit_network(cfg, mode=args.mode)
    elif args.command == "network-analyse":
        analyse_network(cfg, tail=args.tail)
    elif args.command == "network-visualize":
        visualize_network(
            network_path=Path(args.network),
            output_path=Path(args.output),
            structures_dir=Path(args.structures_dir),
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
