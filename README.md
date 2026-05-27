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
├── rbfe_config.yaml          # template — copy and edit for each mutation
├── structures/
│   ├── ING.pdb               # ligand PDB with explicit H
│   ├── INI.pdb
│   └── protein.pdb
├── parameters/               # written by prepare; or supply your own for --skip-param
│   ├── ING/
│   │   ├── ING.mol2          # GAFF atom types + AM1-BCC charges
│   │   ├── ING.frcmod        # missing GAFF parameters
│   │   └── ING.lib           # AMBER off-library file
│   └── INI/
│       ├── INI.mol2
│       ├── INI.frcmod
│       └── INI.lib
└── ING_to_INI/               # written by prepare
    ├── prep/
    │   ├── mapping.json      # atom mapping — review before production
    │   ├── unbound/          # dual-topology solvated systems (.parm7 / .rst7)
    │   └── bound/
    ├── unbound/              # AMBER input files and job scripts
    │   ├── min.in
    │   ├── heating.in
    │   ├── equil.in
    │   ├── replica_1/
    │   │   ├── 1/ti_1.in
    │   │   └── ...
    │   └── ...
    └── bound/
```

## Configuration

Copy `rbfe_config.yaml` and edit for each mutation. Key sections:

```yaml
ligands:
  old:
    pdb:    structures/ING.pdb
    name:   ING                          # ≤ 4 characters, must match residue name
    # lib:    parameters/ING/ING.lib     # optional — explicit path for --skip-param
    # frcmod: parameters/ING/ING.frcmod
  new:
    pdb:    structures/INI.pdb
    name:   INI

forcefield:
  protein: ff14SB    # ff14SB | ff19SB
  ligand:  gaff2     # gaff   | gaff2
  water:   tip3p     # tip3p  | tip4pew | opc
  # ions key is not needed — ion parameters are loaded automatically
  # by the water leaprc (e.g. leaprc.water.tip3p includes ionsjc_tip3p)

system:
  box_padding:       12.0    # Å — distance from solute to box edge
  charge_method:     bcc     # bcc (AM1-BCC) | mul (Mulliken)
  ion_concentration: 0.15    # mol/L NaCl on top of neutralisation ions
  # net_charge: auto         # override automatic charge detection

slurm:
  gpu:
    partition: gpu
    account:   YOUR_ACCOUNT
```

## Atom mapping

The MCS-based atom mapper scores all possible mappings and picks the lowest-penalty one, penalising element changes, ring/non-ring switches, and formal charge changes. The result is written to `prep/mapping.json` before any topology is built.

To inspect and manually override the mapping:

```bash
# See the mapping without building any files
python rbfe_runner.py --config rbfe_config.yaml prepare --dry-run

# Edit prep/mapping.json, then re-run with the override
python rbfe_runner.py --config rbfe_config.yaml prepare --override-mapping ING_to_INI/prep/mapping.json
```

## Notes

- Ligand PDB files must contain **all hydrogens** explicitly — net charge is auto-detected via RDKit `DetermineBonds`, or set `net_charge` to an explicit integer in the config.
- The protein PDB is screened for alternate locations (ALTLOC); a warning is printed if found but the run continues. Clean with `pdb4amber -i raw.pdb -o protein.pdb --nohyd --dry` beforehand if needed.
- `parmchk2 ATTN` warnings (missing GAFF parameters) abort the run before any topology is written.
- For ΔQ ≠ 0 transformations a warning is printed; rigorous treatment requires separate endpoint topologies (AMBER TI tutorial 3).
- If your sqm and antechamber versions mismatch (sqm writes `Atom    Element    Mulliken Charge` instead of `Mulliken charges:`), the script detects this automatically, patches `sqm.out`, and retries antechamber without re-running the QM calculation.
