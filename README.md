# rbfe_runner

AmberTools-native Relative Binding Free Energy (RBFE) workflow. Automates the full pipeline from ligand PDB files and a protein structure to production-ready AMBER TI input files, with automatic atom mapping, soft-core mask derivation, and optional MBAR analysis.

## Requirements

```
conda create -n AmberTools25 -c conda-forge ambertools rdkit pyyaml pymbar
conda activate AmberTools25
```

`pymbar` is only required when `mbar: true` in the config.

## Workflow overview

```
prepare → submit → analyse
```

| Step | What happens |
|---|---|
| **prepare** | Parameterise ligands (antechamber → parmchk2 → tleap lib), compute FE-aware MCS atom mapping, build solvated dual-topology systems (tleap), derive TI masks, write all AMBER input files and job scripts |
| **submit** | Submit the equilibration SLURM job (or run locally) |
| **analyse** | Compute ΔΔG via TI (Gauss-Legendre quadrature) and optionally MBAR |

## Usage

```bash
# Full preparation
python rbfe_runner.py --config rbfe_config.yaml prepare

# Skip parameterisation if lib/frcmod files already exist
python rbfe_runner.py --config rbfe_config.yaml prepare --skip-param

# Validate config and show mapping without writing any files
python rbfe_runner.py --config rbfe_config.yaml prepare --dry-run

# Submit jobs (serial, parallel, or local)
python rbfe_runner.py --config rbfe_config.yaml submit --mode local

# Analyse results
python rbfe_runner.py --config rbfe_config.yaml analyse
```

## Directory layout

```
project/
├── rbfe_config.yaml          # one config per mutation
├── structures/
│   ├── ING.pdb               # ligand PDB with explicit H
│   ├── INI.pdb
│   └── protein.pdb
├── parameters/               # written by prepare (or supplied via --skip-param)
│   ├── ING/
│   │   ├── ING.mol2
│   │   ├── ING.frcmod
│   │   └── ING.lib
│   └── INI/
│       ├── INI.mol2
│       ├── INI.frcmod
│       └── INI.lib
└── ING_to_INI/               # written by prepare
    ├── prep/
    │   ├── mapping.json      # atom mapping — review before production
    │   ├── unbound/          # dual-topology solvated systems
    │   └── bound/
    └── unbound/              # AMBER input files and job scripts
        ├── min.in
        ├── heating.in
        ├── equil.in
        ├── replica_1/
        │   ├── 1/ti_1.in
        │   └── ...
        └── ...
```

## Configuration

Copy `rbfe_config.yaml` and edit for each mutation. Key sections:

```yaml
ligands:
  old:
    pdb: structures/ING.pdb
    name: ING                   # ≤ 4 characters, must match residue name
    lib:    parameters/ING/ING.lib      # optional — for --skip-param
    frcmod: parameters/ING/ING.frcmod   # optional — for --skip-param
  new:
    pdb: structures/INI.pdb
    name: INI

forcefield:
  protein: ff14SB
  ligand:  gaff2
  water:   tip3p
  ions:    ionsjc_tip3p         # optional — comment out to omit

system:
  box_padding:       12.0       # Å
  charge_method:     bcc        # bcc | mul
  ion_concentration: 0.15       # mol/L
  # net_charge: auto            # override automatic charge detection
```

## Atom mapping

The MCS-based atom mapper scores all possible mappings and picks the lowest-penalty one, penalising element changes, ring/non-ring switches, and formal charge changes. The result is written to `prep/mapping.json` before any topology is built.

To inspect and manually override the mapping:

```bash
# Run dry-run to see the mapping without building systems
python rbfe_runner.py prepare --dry-run

# Edit prep/mapping.json, then re-run with the override
python rbfe_runner.py prepare --override-mapping ING_to_INI/prep/mapping.json
```

## Notes

- Ligand PDB files must contain **all hydrogens** explicitly — net charge is auto-detected via RDKit `DetermineBonds`.
- The protein PDB is screened for alternate locations (ALTLOC); a warning is printed if found but the run continues.
- `parmchk2 ATTN` warnings (missing GAFF parameters) abort the run before any topology is written.
- For ΔQ ≠ 0 transformations a warning is printed; rigorous handling requires separate endpoint topologies (AMBER TI tutorial 3).
