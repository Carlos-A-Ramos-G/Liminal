# Liminal

Between states. Between ligands. Between 0 and 1.

AmberTools-native Relative Binding Free Energy (RBFE) workflow. Drop your ligand and protein PDBs — Liminal handles parameterization, atom mapping, system building, and TI/MBAR analysis end-to-end.

## Requirements

```bash
conda create -n AmberTools25 -c conda-forge ambertools rdkit pyyaml pymbar
conda activate AmberTools25
```

`pymbar` is only required when `mbar: true` in the config.

---

## Package layout

```
rbfe/
├── __init__.py       package marker
├── __main__.py       CLI entry point (argparse dispatch)
├── utils.py          subprocess helpers, PDB I/O, filesystem utilities
├── config.py         config validation, environment checks
├── parameterize.py   charge detection, antechamber → parmchk2 → tleap lib
├── mapping.py        FE-aware MCS atom mapping
├── system.py         tleap system building, PDB screening, TI mask derivation
├── simulation.py     AMBER input templates, GL quadrature, job-script generation
├── analysis.py       TI (Gauss-Legendre) and MBAR free energy analysis
├── commands.py       single-pair workflow (prepare, submit, analyse)
├── network.py        perturbation network MST, high-throughput workflow
└── visualize.py      interactive HTML tree visualization
rbfe_runner.py        thin shim — delegates to the package (backward-compatible)
```

Both invocation styles are equivalent:

```bash
python rbfe_runner.py --config my.yaml prepare   # original style
python -m rbfe       --config my.yaml prepare   # package style
```

---

## Workflow overview

```
prepare → submit → analyse
```

| Step | What happens |
|---|---|
| **prepare** | Parameterise ligands (antechamber → parmchk2 → tleap lib), compute FE-aware MCS atom mapping, build solvated dual-topology systems (tleap), derive TI masks, write all AMBER input files and job scripts |
| **submit** | Submit the equilibration SLURM job (or run locally) |
| **analyse** | Compute ΔΔG via TI (Gauss-Legendre quadrature) and optionally MBAR |

For a ligand network (multiple ligands, MST of transformations):

```
network → network-submit → network-analyse
```

| Step | What happens |
|---|---|
| **network** | Discover all ligand PDBs, build MST via Morgan-Tanimoto similarity, parameterise each ligand once, run **prepare** for every edge |
| **network-submit** | Submit all transformations with branch-aware SLURM `afterok` dependencies |
| **network-analyse** | Run **analyse** for every edge and print a ΔΔG summary table |

---

## Single-pair usage

### 1. Prepare

```bash
# Full preparation (antechamber + mapping + tleap + input files)
python -m rbfe --config rbfe_config.yaml prepare

# Skip parameterisation if lib/frcmod already exist in parameters/{name}/
python -m rbfe --config rbfe_config.yaml prepare --skip-param

# Dry run: validate config and show atom mapping without writing any files
python -m rbfe --config rbfe_config.yaml prepare --dry-run

# Use a manually edited atom mapping instead of running MCS
python -m rbfe --config rbfe_config.yaml prepare --override-mapping ING_to_INI/prep/mapping.json

# Choose submission mode (affects the generated job scripts)
python -m rbfe --config rbfe_config.yaml prepare --mode serial    # default
python -m rbfe --config rbfe_config.yaml prepare --mode parallel  # windows run in parallel
python -m rbfe --config rbfe_config.yaml prepare --mode local     # no SLURM, sequential script
```

Preparation creates the following structure:

```
{OLD}_to_{NEW}/
├── prep/
│   ├── mapping.json          atom mapping — review before production
│   ├── unbound/
│   │   ├── ti.parm7
│   │   └── ti.rst7
│   └── bound/
│       ├── ti.parm7
│       └── ti.rst7
├── unbound/
│   ├── ti.parm7 → (symlink)
│   ├── ti.rst7  → (symlink)
│   ├── min.in
│   ├── heating.in
│   ├── equil.in
│   ├── EQUILIBRATION.cmd     (or run_local.sh for --mode local)
│   └── replica_{1..N}/
│       └── {1..n_lambdas}/
│           ├── ti.parm7 → (symlink)
│           ├── ti_{w}.in
│           └── FEP_PROD_{w}.cmd
└── bound/
    └── (same structure as unbound/)
```

Parameters are written under `parameters/{name}/`:

```
parameters/
└── ING/
    ├── ING.mol2     GAFF atom types + AM1-BCC charges
    ├── ING.frcmod   missing GAFF parameters
    ├── ING.lib      AMBER off-library file
    └── ING.pdb      copy of input PDB with residue name corrected
```

### 2. Submit

```bash
# Submit via SLURM (serial or parallel window submission)
python -m rbfe --config rbfe_config.yaml submit
python -m rbfe --config rbfe_config.yaml submit --mode parallel

# Run locally in the background (no SLURM required)
python -m rbfe --config rbfe_config.yaml submit --mode local
```

### 3. Analyse

```bash
# TI (Gauss-Legendre) + MBAR analysis using last 4000 dV/dλ records per window
python -m rbfe --config rbfe_config.yaml analyse

# Use a different number of records (e.g. last 2000 for a shorter run)
python -m rbfe --config rbfe_config.yaml analyse --tail 2000
```

Output:

```
══════════════════════════════════════════════════════════════
  Mutation : ING_to_INI
  Windows  : 9    Replicas : 3
  ...
══════════════════════════════════════════════════════════════

  [unbound]  TI (Gauss-Legendre)
    replica 1: ΔG(TI) =    -3.142 kcal/mol
    ...
    mean      : ΔG(TI) =    -3.210 ± 0.091 kcal/mol

  [bound]  TI (Gauss-Legendre)
    ...

  ΔΔG = ΔG(bound) − ΔG(unbound)

  TI / Gauss-Legendre:
    ΔΔG = -1.234 ± 0.120 kcal/mol

  MBAR (pymbar):
    ΔΔG = -1.198 ± 0.105 kcal/mol
```

---

## Network (high-throughput) usage

### 1. Prepare the network

Place all ligand PDB files (with explicit H) in `structures/` alongside the protein PDB.

```bash
python -m rbfe --config rbfe_config.yaml network
python -m rbfe --config rbfe_config.yaml network --dry-run       # preview only
python -m rbfe --config rbfe_config.yaml network --skip-param    # reuse existing parameters
python -m rbfe --config rbfe_config.yaml network --structures-dir path/to/pdbs
```

This discovers all `.pdb` files (excluding `protein.pdb`), computes a minimum spanning tree using Morgan-Tanimoto similarity, parameterises each ligand exactly once, and runs the full prepare pipeline for every edge. A `network.json` file is written with the full graph definition.

The ASCII tree printed during preparation shows the MST structure and wave ordering:

```
  Transformation tree  (heavy atoms in parentheses):

  L1 (22)
  ├── L4A (28)  [sim=0.712]
  │   ├── L4B (30)  [sim=0.823]
  │   └── L4H (31)  [sim=0.801]
  └── L5  (29)  [sim=0.688]
      └── L7I (33)  [sim=0.754]
```

### 2. Submit the network

```bash
# Branch-aware submission (default): each edge waits for its parent edge to finish.
# A failed branch stops only its downstream transformations; sibling branches continue.
python -m rbfe --config rbfe_config.yaml network-submit

# Submit all equilibration jobs immediately with no inter-transformation dependencies
python -m rbfe --config rbfe_config.yaml network-submit --mode parallel

# Run sequentially in the foreground (for local testing)
python -m rbfe --config rbfe_config.yaml network-submit --mode local
```

The branch-aware mode groups submissions by wave depth and applies SLURM `--dependency=afterok` so that:

- All transformations in wave 1 start immediately.
- A transformation in wave N waits for its wave N−1 parent to complete.
- If a parent job fails, only its downstream branch is blocked; unrelated branches are unaffected.

### 3. Analyse the network

```bash
python -m rbfe --config rbfe_config.yaml network-analyse
python -m rbfe --config rbfe_config.yaml network-analyse --tail 2000
```

Prints a ΔΔG summary table across all MST edges:

```
══════════════════════════════════════════════════════════════
  Network ΔΔG Summary
══════════════════════════════════════════════════════════════
  Transformation              ΔΔG TI (kcal/mol)   ΔΔG MBAR (kcal/mol)
  ──────────────────────────  ──────────────────  ────────────────────
  L4A_to_L4B                       -1.23 ± 0.09         -1.19 ± 0.08
  L4A_to_L4H                       +0.44 ± 0.12         +0.41 ± 0.11
  ...
```

### 4. Visualize the network

```bash
# Generate interactive HTML from an existing network.json
python -m rbfe --config rbfe_config.yaml network-visualize

# Custom input/output paths
python -m rbfe --config rbfe_config.yaml network-visualize \
    --network network.json \
    --output  network.html

# Build network on-the-fly (no network.json needed)
python -m rbfe --config rbfe_config.yaml network-visualize \
    --structures-dir structures/
```

Opens a self-contained HTML file with:
- Hierarchical tree layout with the smallest ligand at the root
- 2D molecular depictions aligned to a common scaffold
- Color-coded edges (red → yellow → green by Tanimoto similarity)
- Scroll to zoom, drag to pan, hover edges to see similarity values

---

## Configuration

Copy `rbfe_config.yaml` and edit for each mutation. Key sections:

```yaml
# ── Alchemical settings ───────────────────────────────────────
temperature: 300.0       # K
n_lambdas:   9           # GL windows: 5 (fast), 9 (balanced), 12 (accurate)
replicates:  3           # independent runs for error estimation
mbar:        true        # also run MBAR in addition to TI

# ── Ligands (single-pair mode) ────────────────────────────────
# name is optional: derived from the PDB filename stem if omitted.
# e.g. structures/7g.pdb → L7G,  structures/4a.pdb → L4A
ligands:
  old:
    pdb: structures/LIG1.pdb
    # name: LIG1               # optional override (≤ 4 chars)
  new:
    pdb: structures/LIG2.pdb

# ── Protein (omit for ligand hydration FEP) ───────────────────
protein:
  pdb: structures/protein.pdb

# ── Force fields ──────────────────────────────────────────────
forcefield:
  protein: ff14SB          # ff14SB | ff19SB
  ligand:  gaff2           # gaff   | gaff2
  water:   tip3p           # tip3p  | tip4pew | opc

# ── System building ───────────────────────────────────────────
system:
  box_padding:       12.0   # Å from solute to box edge
  charge_method:     bcc    # bcc (AM1-BCC) | mul (Mulliken)
  ion_concentration: 0.15   # mol/L NaCl on top of neutralisation ions
  # net_charge: auto        # override auto-detected charge

# ── SLURM ─────────────────────────────────────────────────────
slurm:
  gpu:
    partition: gpu
    account:   YOUR_ACCOUNT
    time:      "24:00:00"
    ntasks:    1
    gres:      gpu:1
```

### Using `--skip-param`

When `--skip-param` is passed, antechamber/parmchk2 are skipped and existing files under `parameters/{name}/` are used instead. The expected layout (created automatically on a previous run, or supplied manually) is:

```
parameters/
└── {NAME}/
    ├── {NAME}.lib      AMBER off-library file
    ├── {NAME}.frcmod   GAFF parameter corrections
    └── {NAME}.pdb      ligand PDB with residue name set to {NAME}
```

If `{NAME}.pdb` is absent but the `.lib` and `.frcmod` are present, it is created automatically from the source PDB in the config.

### Manually overriding the atom mapping

```bash
# 1. Run a dry-run to see the auto mapping
python -m rbfe --config rbfe_config.yaml prepare --dry-run

# 2. Run prepare once to write the mapping file (no dry-run)
python -m rbfe --config rbfe_config.yaml prepare

# 3. Edit prep/mapping.json to correct any mismatched atoms

# 4. Re-run prepare with the override
python -m rbfe --config rbfe_config.yaml prepare \
    --override-mapping ING_to_INI/prep/mapping.json
```

`mapping.json` fields:

| Field | Description |
|---|---|
| `matched_old` | Atom names in old ligand that map to the common core |
| `matched_new` | Corresponding atom names in new ligand |
| `unique_old` | Atoms unique to old ligand → `scmask1` (soft-core) |
| `unique_new` | Atoms unique to new ligand → `scmask2` (soft-core) |
| `score` | FE penalty score (lower is better; for reference only) |
| `n_common_heavy` | Number of matched heavy atoms |

---

## Notes

- Ligand PDB files must contain **all hydrogens** explicitly. Net charge is auto-detected via RDKit `DetermineBonds`, or override with `net_charge` in the config.
- The protein PDB is screened for alternate locations (ALTLOC). Pre-process with `pdb4amber -i raw.pdb -o protein.pdb --nohyd --dry` if needed.
- `parmchk2 ATTN` warnings (missing GAFF parameters) abort the run before any topology is written.
- For ΔQ ≠ 0 transformations a warning is printed; rigorous treatment requires separate endpoint topologies (AMBER TI tutorial 3).
- If your sqm and antechamber versions mismatch (sqm writes `Atom Element Mulliken Charge` instead of `Mulliken charges:`), the script detects this automatically, patches `sqm.out`, and retries antechamber.
- Neutralisation ions (`addIons Na+ 0` / `addIons Cl- 0`) and salt (`addIonsRand Na+ N Cl- N`) are added in separate tleap steps so the final system always has zero net charge.
