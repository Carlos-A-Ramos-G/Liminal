"""Low-level subprocess, PDB I/O, and filesystem helpers."""

import os
import subprocess
import sys
from pathlib import Path


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


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def _write_exe(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _write_renamed_pdb(src: Path, dst: Path, new_resname: str) -> None:
    """Copy src PDB to dst with the residue-name field (cols 18-21) set to new_resname."""
    new_rn = f"{new_resname:<4s}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line[:6] in ("ATOM  ", "HETATM"):
                line = line[:17] + new_rn + line[21:]
            fout.write(line)


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
