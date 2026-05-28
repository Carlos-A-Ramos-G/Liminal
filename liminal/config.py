"""Config validation and environment checks."""

import shutil
import sys
from pathlib import Path


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
            _req(lig, "pdb", label=f"ligands.{side}")
            pdb = Path(lig["pdb"])
            if not pdb.exists():
                sys.exit(f"Config error: ligands.{side}.pdb not found: {pdb}")
            if "name" in lig and len(lig["name"]) > 4:
                sys.exit(
                    f"Config error: ligands.{side}.name must be ≤ 4 characters "
                    f"(AMBER limit); got '{lig['name']}'"
                )

    if "protein" in cfg:
        pdb = Path(cfg["protein"]["pdb"])
        if not pdb.exists():
            sys.exit(f"Config error: protein.pdb not found: {pdb}")

    _req(cfg["forcefield"], "protein", "ligand", "water", label="forcefield")
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
