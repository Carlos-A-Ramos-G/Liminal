"""Perturbation network: MST construction, high-throughput prepare/submit/analyse."""

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

from .commands import prepare, submit, analyse
from .parameterize import _infer_net_charge, parameterize_ligand
from .config import check_environment, validate_config
from .utils import _stem_to_resname, _write_renamed_pdb

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


def _kruskal_mst(
    n: int, edges: list[tuple[float, int, int]]
) -> list[tuple[int, int, float]]:
    """Kruskal's MST via union-find with path compression."""
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
    """Return all .pdb files in structures_dir sorted alphabetically,
    excluding the file whose stem matches protein_stem (case-insensitive)."""
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
                "old": {"pdb": edge["old_pdb"], "name": old_name,
                        "lib": str(old_lib), "frcmod": str(old_frcmod)},
                "new": {"pdb": edge["new_pdb"], "name": new_name,
                        "lib": str(new_lib), "frcmod": str(new_frcmod)},
            },
        }
        prepare(edge_cfg, mode=mode, dry_run=dry_run, skip_param=not dry_run)

    print(f"\n{'═' * 62}")
    if dry_run:
        print("  Network dry run complete — no files written.")
    else:
        print(f"  Network preparation complete: {n_edges} transformations ready.")
        print(f"  Submit:  python -m rbfe network-submit")
        print(f"  Analyse: python -m rbfe network-analyse")
    print(f"{'═' * 62}\n")


def submit_network(cfg: dict, mode: str = "branch") -> None:
    """
    Submit all transformations in network.json.

    mode = "branch"   — per-branch afterok chains (default).
    mode = "parallel" — submit every equil job immediately; no dependencies.
    mode = "local"    — run sequentially in the foreground (testing).
    """
    net_path = Path("network.json")
    if not net_path.exists():
        sys.exit("network.json not found — run 'network' first.")
    network = json.loads(net_path.read_text())

    legs = ["unbound", "bound"] if "protein" in cfg else ["unbound"]

    adj: dict[str, list[str]] = {}
    for edge in network["edges"]:
        adj.setdefault(edge["old"], []).append(edge["new"])
        adj.setdefault(edge["new"], []).append(edge["old"])

    root = min(network["ligands"], key=lambda x: x["n_heavy"])["name"]

    parent: dict[str, str | None] = {root: None}
    depth:  dict[str, int]        = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, []):
            if nb not in parent:
                parent[nb] = node
                depth[nb]  = depth[node] + 1
                queue.append(nb)

    def _child(edge: dict) -> str:
        return edge["new"] if parent.get(edge["new"]) == edge["old"] else edge["old"]

    sorted_edges = sorted(network["edges"], key=lambda e: depth[_child(e)])

    if mode == "local":
        for edge in sorted_edges:
            mutation = f"{edge['old']}_to_{edge['new']}"
            print(f"\n  ── {mutation} ──")
            submit(
                {**cfg, "ligands": {
                    "old": {"pdb": edge["old_pdb"], "name": edge["old"]},
                    "new": {"pdb": edge["new_pdb"], "name": edge["new"]},
                }},
                mode="local",
            )
        return

    node_jids: dict[str, list[int]] = {root: []}

    waves: dict[int, list[dict]] = {}
    for edge in sorted_edges:
        waves.setdefault(depth[_child(edge)], []).append(edge)

    n_waves = max(waves) + 1
    print(f"\n{'═' * 62}")
    print(f"  Network submission — {len(network['edges'])} transformations, "
          f"{n_waves} waves")
    print(f"  Strategy : {'branch-aware afterok' if mode == 'branch' else 'all parallel'}")
    print(f"{'═' * 62}")

    for w in sorted(waves):
        print(f"\n  Wave {w}:")
        for edge in waves[w]:
            old_n, new_n = edge["old"], edge["new"]
            child    = _child(edge)
            src      = parent[child]
            mutation = f"{old_n}_to_{new_n}"

            dep_jids = node_jids.get(src, []) if mode == "branch" else []
            dep_flag = (
                ["--dependency", f"afterok:{':'.join(str(j) for j in dep_jids)}"]
                if dep_jids else []
            )

            submitted: list[int] = []
            for leg in legs:
                leg_dir  = Path(mutation) / leg
                cmd_file = leg_dir / "EQUILIBRATION.cmd"
                if not cmd_file.exists():
                    print(f"    [{leg}] EQUILIBRATION.cmd not found — skipping",
                          file=sys.stderr)
                    continue

                res = subprocess.run(
                    ["sbatch"] + dep_flag + ["EQUILIBRATION.cmd"],
                    capture_output=True, text=True, cwd=str(leg_dir),
                )
                if res.returncode == 0:
                    jid = int(res.stdout.strip().split()[-1])
                    submitted.append(jid)
                    dep_str = (f" (afterok: {', '.join(str(j) for j in dep_jids)})"
                               if dep_jids else "")
                    print(f"    {mutation}/{leg} → job {jid}{dep_str}")
                else:
                    print(f"    {mutation}/{leg} — sbatch failed: "
                          f"{res.stderr.strip()}", file=sys.stderr)

            node_jids[child] = submitted

    print(f"\n{'═' * 62}")
    print(f"  All jobs submitted. Monitor with: squeue -u $USER")
    print(f"{'═' * 62}\n")


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
