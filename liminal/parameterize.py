"""Ligand parameterisation: charge detection, antechamber, parmchk2, tleap lib."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .utils import _run, _check_frcmod, _write_renamed_pdb

_GAFF_SOURCE = {"gaff": "leaprc.gaff", "gaff2": "leaprc.gaff2"}


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


def detect_formal_charges(pdb_old: Path, pdb_new: Path) -> tuple[int, int]:
    """Return (q_old, q_new) formal charges from PDB files via RDKit."""
    return _infer_net_charge(pdb_old), _infer_net_charge(pdb_new)


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
    patched = re.sub(
        r"Atom\s+Element\s+Mulliken Charge\s*\n",
        "Mulliken charges:\n          1\n",
        text,
    )
    if patched == text:
        return False
    sqm_out.write_text(patched)
    return True


def _run_antechamber(cmd: list, cwd: Path, desc: str) -> str:
    """Run antechamber, retrying once after patching sqm.out if charge parsing fails."""
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


def parameterize_ligand(
    pdb: Path, resname: str, cfg: dict,
) -> tuple[Path, Path]:
    """
    Run antechamber → parmchk2 → tleap for one ligand.

    Outputs are written to parameters/{resname}/:
        {resname}.mol2    GAFF atom types + AM1-BCC charges
        {resname}.frcmod  missing GAFF parameters
        {resname}.lib     AMBER off-library file
        {resname}.pdb     copy of input PDB with residue name corrected

    Returns (lib, frcmod).  Aborts on any parmchk2 ATTN warning.
    """
    work_dir = Path("parameters") / resname
    lib_out  = work_dir / f"{resname}.lib"
    frc_out  = work_dir / f"{resname}.frcmod"
    if lib_out.exists() and frc_out.exists():
        print(f"    {resname}: parameters already exist — skipping antechamber")
        renamed_pdb = work_dir / f"{resname}.pdb"
        if not renamed_pdb.exists():
            _write_renamed_pdb(pdb.resolve(), renamed_pdb.resolve(), resname)
        return lib_out.resolve(), frc_out.resolve()

    work_dir.mkdir(parents=True, exist_ok=True)
    gaff   = cfg["forcefield"]["ligand"]
    method = cfg["system"]["charge_method"]
    mult   = cfg["system"].get("multiplicity", 1)
    charge = _resolve_charge(pdb, cfg)

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
        f"source {_GAFF_SOURCE.get(gaff, 'leaprc.' + gaff)}\n"
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

    renamed_pdb = work_dir / f"{resname}.pdb"
    _write_renamed_pdb(pdb.resolve(), renamed_pdb.resolve(), resname)

    return lib, frcmod
