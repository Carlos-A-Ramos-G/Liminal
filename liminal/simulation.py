"""AMBER input templates, Gauss-Legendre quadrature, and job-script generation."""

import sys
from pathlib import Path

import numpy as np

from .utils import _symlink, _write_exe

# ── Physics constants ─────────────────────────────────────────────────────────

_KB_KCAL: float = 0.001987204258   # kcal mol⁻¹ K⁻¹


def compute_gl_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """GL nodes on [0, 1] and weights summing to 1."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (nodes + 1) / 2, weights / 2


def _middle(n: int) -> int:
    return (n + 1) // 2


def _mbar_beta(temp: float) -> float:
    """β = 1/(k_B T) in mol kcal⁻¹."""
    return 1.0 / (_KB_KCAL * temp)


def _format_mbar_block(lambdas: np.ndarray) -> str:
    """AMBER &cntrl fragment for ifmbar=1, including λ=0 and λ=1 endpoints."""
    all_lam = np.concatenate([[0.0], lambdas, [1.0]])
    lam_str = ", ".join(f"{l:.5f}" for l in all_lam)
    return (
        f"\n   ifmbar = 1, mbar_states = {len(all_lam)},"
        f"\n   mbar_lambda = {lam_str},"
    )


# ── AMBER input templates ─────────────────────────────────────────────────────

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


# ── SLURM / local script helpers ──────────────────────────────────────────────

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


def _equil_cpptraj_params(cfg: dict, n_replicas: int) -> tuple[int, int, int]:
    equil = cfg["simulation"]["equil"]
    total = equil["nstlim"] // equil["ntwx"]
    start = total // 2
    if n_replicas == 1:
        return total, total, 1
    step = (total - start) // (n_replicas - 1)
    return start, total, step


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


def _gen_equilibration_cmd(
    leg: str, n_replicas: int, mid: int, cfg: dict, mode: str
) -> str:
    res = cfg["slurm"]["gpu"]
    gpu = cfg["execution_command"]["gpu"]
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


def _gen_prod_cmd(
    window: int, replica: int, n_windows: int, n_replicas: int,
    leg: str, cfg: dict, mode: str
) -> str:
    mid  = _middle(n_windows)
    res  = cfg["slurm"]["gpu"]
    gpu  = cfg["execution_command"]["gpu"]

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


def _gen_local_script(
    leg: str, n_windows: int, n_replicas: int,
    lambdas: np.ndarray, cfg: dict,
    timask1: str, timask2: str, scmask1: str, scmask2: str,
) -> str:
    mid   = _middle(n_windows)
    sim   = cfg["simulation"]
    temp  = cfg["temperature"]
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


def generate_leg_inputs(
    leg: str,
    prep_parm7: Path, prep_rst7: Path,
    timask1: str, timask2: str, scmask1: str, scmask2: str,
    lambdas: np.ndarray, weights: np.ndarray,
    cfg: dict, mode: str,
) -> None:
    """Write all AMBER input files and job scripts for one leg (unbound / bound)."""
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

    mbar_blk = _format_mbar_block(lambdas) if mbar else ""
    mask_kw  = dict(timask1=timask1, timask2=timask2,
                    scmask1=scmask1, scmask2=scmask2)

    _symlink(prep_parm7.resolve(), leg_dir / "ti.parm7")
    _symlink(prep_rst7.resolve(),  leg_dir / "ti.rst7")

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

    if mode == "local":
        _write_exe(leg_dir / "run_local.sh",
                   _gen_local_script(leg, n_lam, n_rep, lambdas, cfg,
                                     timask1, timask2, scmask1, scmask2))
    else:
        _write_exe(leg_dir / "EQUILIBRATION.cmd",
                   _gen_equilibration_cmd(leg, n_rep, mid, cfg, mode))

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
                    _gen_prod_cmd(w_idx, replica, n_lam, n_rep, leg, cfg, mode),
                )
