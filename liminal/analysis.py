"""TI (Gauss-Legendre) and MBAR free energy analysis."""

import sys
from pathlib import Path

import numpy as np

from .simulation import compute_gl_quadrature, _mbar_beta


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
    """
    Parse MBAR cross-energies from the AMBER mdout (.out) file.

    AMBER writes MBAR data to the mdout file, not the .en file, in blocks:
        MBAR Energy analysis:
        Energy at {lam} =    {value}
        ...                              (n_states lines)
    """
    out_file = en_file.with_suffix(".out")
    if not out_file.exists():
        raise RuntimeError(
            f"MBAR output file not found: {out_file}\n"
            "  AMBER writes MBAR energies to the mdout (.out) file."
        )

    frames: list[list[float]] = []
    block:  list[float]       = []
    in_block = False

    with open(out_file) as fh:
        for line in fh:
            s = line.strip()
            if s == "MBAR Energy analysis:":
                in_block = True
                block    = []
            elif in_block:
                if s.startswith("Energy at"):
                    try:
                        block.append(float(s.split("=")[-1]))
                    except ValueError:
                        in_block = False
                        block    = []
                else:
                    if len(block) == n_states:
                        frames.append(block)
                    in_block = False
                    block    = []

    if in_block and len(block) == n_states:
        frames.append(block)

    if not frames:
        raise RuntimeError(
            f"No MBAR records in {out_file}.\n"
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
    n_states = n_lambdas + 2

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
