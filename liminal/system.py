"""tleap system building, PDB pre-flight check, and TI mask derivation."""

import re
import sys
from pathlib import Path

from .utils import _run

_FF_SOURCES: dict[str, str] = {
    "ff14SB":  "leaprc.protein.ff14SB",
    "ff19SB":  "leaprc.protein.ff19SB",
    "gaff":    "leaprc.gaff",
    "gaff2":   "leaprc.gaff2",
    "tip3p":   "leaprc.water.tip3p",
    "tip4pew": "leaprc.water.tip4pew",
    "opc":     "leaprc.water.opc",
}

_BOX_NAME: dict[str, str] = {
    "tip3p":   "TIP3PBOX",
    "tip4pew": "TIP4PEWBOX",
    "opc":     "OPCBOX",
}


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

    non_std = (
        set(re.findall(r"^HETATM.{8}(\S+)", text, re.MULTILINE))
        - {"HOH", "WAT", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE"}
    )
    if non_std:
        print(
            f"  NOTE [{pdb.name}]: non-standard HETATM residues found: "
            f"{', '.join(sorted(non_std))}\n"
            "  These are fine if they are your TI ligands; otherwise they need "
            "separate parameterisation."
        )


def _estimate_salt_pairs(conc_M: float, padding: float) -> int:
    """Return the number of NaCl pairs to add for the target ionic strength.

    Neutralisation is handled separately by tleap's addIons command.
    Box volume is estimated assuming edge ≈ 2*padding + 50 Å.
    """
    edge_A = 2 * padding + 50.0
    vol_L  = (edge_A * 1e-10) ** 3 * 1e3
    return max(0, round(conc_M * 6.022e23 * vol_L))


def _ff_source(key: str, val: str) -> str:
    if val in _FF_SOURCES:
        return _FF_SOURCES[val]
    if key == "protein":
        return f"leaprc.protein.{val}"
    if key == "water":
        return f"leaprc.water.{val}"
    return f"leaprc.{val}"


def _tleap_load(resname: str, path: Path) -> str:
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

    Returns (parm7_path, rst7_path).
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    ff      = cfg["forcefield"]
    sys_cfg = cfg["system"]
    padding = float(sys_cfg["box_padding"])
    conc    = float(sys_cfg["ion_concentration"])
    water   = ff["water"].lower()
    box     = _BOX_NAME.get(water, "TIP3PBOX")

    n_salt = _estimate_salt_pairs(conc, padding)

    old_coord_pdb = (lig_old_struct.parent / f"{old_resname}.pdb").resolve()
    new_coord_pdb = (lig_new_struct.parent / f"{new_resname}.pdb").resolve()

    sources = [
        f"source {_ff_source(key, val)}"
        for key in ("protein", "ligand", "water")
        for val in [ff.get(key)]
        if val
    ]
    lines: list[str] = sources + ["",
        f"loadAmberParams {lig_old_frcmod.resolve()}",
        _tleap_load(old_resname, lig_old_struct),
        f"{old_resname} = loadPdb {old_coord_pdb}",
        "",
        f"loadAmberParams {lig_new_frcmod.resolve()}",
        _tleap_load(new_resname, lig_new_struct),
        f"{new_resname} = loadPdb {new_coord_pdb}",
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
        f"solvateOct solute {box} {padding}",
        "addIons solute Na+ 0",
        "addIons solute Cl- 0",
    ]
    if n_salt:
        lines.append(f"addIonsRand solute Na+ {n_salt} Cl- {n_salt}")

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


def derive_masks(
    parm7: Path,
    old_resname: str, new_resname: str,
    unique_old_names: list[str], unique_new_names: list[str],
) -> tuple[str, str, str, str]:
    """
    Derive AMBER TI mask strings by parsing the parm7 RESIDUE_LABEL section.

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
            return '""'
        return f'":{resnum}@{",".join(names)}"'

    return timask1, timask2, _scmask(old_resnum, unique_old_names), _scmask(new_resnum, unique_new_names)
