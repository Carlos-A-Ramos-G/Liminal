"""Single-pair workflow: prepare, submit, analyse."""

import json
import subprocess
import sys
from pathlib import Path

from .analysis import _ti_system_dg, _mbar_system_dg
from .config import check_environment, validate_config
from .mapping import compute_fe_mapping
from .parameterize import parameterize_ligand, detect_formal_charges
from .simulation import compute_gl_quadrature, _middle, _mbar_beta, generate_leg_inputs
from .system import build_system, check_pdb, derive_masks
from .utils import _stem_to_resname, _write_renamed_pdb


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
    pdb_old  = Path(old_cfg["pdb"])
    pdb_new  = Path(new_cfg["pdb"])
    old_name = old_cfg.get("name") or _stem_to_resname(pdb_old)
    new_name = new_cfg.get("name") or _stem_to_resname(pdb_new)
    old_cfg["name"] = old_name
    new_cfg["name"] = new_name

    mutation = f"{old_name}_to_{new_name}"
    prep_dir = Path(mutation) / "prep"

    lambdas, weights = compute_gl_quadrature(cfg["n_lambdas"])
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
        old_struct = _param_dir(old_name) / f"{old_name}.lib"
        old_frcmod = _param_dir(old_name) / f"{old_name}.frcmod"
        new_struct = _param_dir(new_name) / f"{new_name}.lib"
        new_frcmod = _param_dir(new_name) / f"{new_name}.frcmod"
        for f in (old_struct, old_frcmod, new_struct, new_frcmod):
            if not f.exists():
                sys.exit(
                    f"  --skip-param: file not found: {f}\n"
                    f"  Place GAFF2 parameter files under parameters/{{name}}/ "
                    f"or omit --skip-param to run antechamber."
                )
        for name, pdb in ((old_name, pdb_old), (new_name, pdb_new)):
            renamed = _param_dir(name) / f"{name}.pdb"
            if not renamed.exists():
                _write_renamed_pdb(pdb.resolve(), renamed.resolve(), name)
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

    legs = ["unbound", "bound"] if has_protein else ["unbound"]
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
                          '"soft-core-old"', '"soft-core-new"')
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
        print(f"  Submit with:  python -m rbfe submit --mode {mode}")
    print(f"{'═' * 62}\n")


def submit(cfg: dict, mode: str = "serial") -> None:
    """Submit the equilibration job for each leg."""
    ligs = cfg["ligands"]
    for side in ("old", "new"):
        if "name" not in ligs[side]:
            ligs[side]["name"] = _stem_to_resname(Path(ligs[side]["pdb"]))
    mutation = f"{ligs['old']['name']}_to_{ligs['new']['name']}"
    legs = ["unbound", "bound"] if "protein" in cfg else ["unbound"]

    if mode == "local":
        for leg in legs:
            script = Path(mutation) / leg / "run_local.sh"
            if not script.exists():
                sys.exit(f"Script not found: {script}\nRun 'prepare --mode local' first.")
            log = Path(mutation) / leg / "run.log"
            proc = subprocess.Popen(
                f"bash {script.resolve()} > {log.resolve()} 2>&1",
                shell=True, start_new_session=True,
            )
            print(f"  [{leg}] launched in background (PID {proc.pid}) → {log}")
        return

    for leg in legs:
        leg_dir  = Path(mutation) / leg
        cmd_file = leg_dir / "EQUILIBRATION.cmd"
        if not cmd_file.exists():
            sys.exit(f"Script not found: {cmd_file}\nRun 'prepare' first.")
        res = subprocess.run(
            ["sbatch", "EQUILIBRATION.cmd"],
            capture_output=True, text=True, cwd=str(leg_dir),
        )
        if res.returncode == 0:
            print(f"  [{leg}] {res.stdout.strip()}")
        else:
            print(f"  [{leg}] sbatch failed: {res.stderr.strip()}", file=sys.stderr)


def analyse(cfg: dict, tail: int = 4000) -> dict:
    """Compute ΔΔG = ΔG(bound) − ΔG(unbound) via TI and, if enabled, MBAR."""
    validate_config(cfg)
    ligs = cfg["ligands"]
    for side in ("old", "new"):
        if "name" not in ligs[side]:
            ligs[side]["name"] = _stem_to_resname(Path(ligs[side]["pdb"]))

    mutation = f"{cfg['ligands']['old']['name']}_to_{cfg['ligands']['new']['name']}"
    base     = Path(mutation)
    n_lam    = cfg["n_lambdas"]
    n_rep    = cfg["replicates"]
    temp     = float(cfg["temperature"])
    mbar     = bool(cfg.get("mbar", True))
    _, weights = compute_gl_quadrature(n_lam)
    legs     = ["unbound", "bound"] if "protein" in cfg else ["unbound"]

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
            ti_results[leg] = _ti_system_dg(leg, base, n_rep, n_lam, weights, tail)
        except (FileNotFoundError, RuntimeError) as exc:
            sys.exit(f"  ERROR: {exc}")

        if mbar:
            print(f"\n  [{leg}]  MBAR")
            try:
                mbar_results[leg] = _mbar_system_dg(leg, base, n_rep, n_lam, tail, temp)
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
        for leg, (dg, std) in ti_results.items():
            print(f"  [{leg}]  ΔG(TI) = {dg:+.3f} ± {std:.3f} kcal/mol")

    print(f"{'═' * 62}\n")
    return {"ti": ti_results, "mbar": mbar_results}
