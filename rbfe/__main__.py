"""CLI entry point — python -m rbfe <command> [options]"""

import json
import sys
from pathlib import Path

import argparse

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required:  conda install -c conda-forge pyyaml")

from .commands import prepare, submit, analyse
from .network import prepare_network, submit_network, analyse_network
from .visualize import visualize_network


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AmberTools-native RBFE workflow — automated atom mapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Config file (default: config.yaml)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Parameterise, map, build, write inputs")
    p_prep.add_argument("--mode", default="serial",
                        choices=["serial", "parallel", "local"])
    p_prep.add_argument("--dry-run", action="store_true")
    p_prep.add_argument("--skip-param", action="store_true")
    p_prep.add_argument("--override-mapping",
                        help="Path to a manually edited mapping.json")

    p_sub = sub.add_parser("submit", help="Submit SLURM / local jobs")
    p_sub.add_argument("--mode", default="serial",
                       choices=["serial", "parallel", "local"])

    p_ana = sub.add_parser("analyse", help="TI + MBAR ΔΔG analysis")
    p_ana.add_argument("--tail", type=int, default=4000, metavar="N")

    p_net = sub.add_parser("network",
                            help="Build MST network and prepare all transformations")
    p_net.add_argument("--structures-dir", default="structures")
    p_net.add_argument("--mode", default="serial",
                       choices=["serial", "parallel", "local"])
    p_net.add_argument("--dry-run", action="store_true")
    p_net.add_argument("--skip-param", action="store_true")

    p_netsub = sub.add_parser("network-submit",
                               help="Submit all jobs listed in network.json")
    p_netsub.add_argument("--mode", default="branch",
                          choices=["branch", "parallel", "local"])

    p_netana = sub.add_parser("network-analyse",
                               help="Analyse all transformations in network.json")
    p_netana.add_argument("--tail", type=int, default=4000, metavar="N")

    p_netvis = sub.add_parser("network-visualize",
                               help="Generate interactive HTML tree visualization")
    p_netvis.add_argument("--network", default="network.json", metavar="PATH")
    p_netvis.add_argument("--output", default="network.html", metavar="PATH")
    p_netvis.add_argument("--structures-dir", default="structures", metavar="DIR")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.command == "prepare":
        if getattr(args, "override_mapping", None):
            with open(args.override_mapping) as fh:
                cfg["_override_mapping"] = json.load(fh)
        prepare(cfg, mode=args.mode, dry_run=args.dry_run,
                skip_param=args.skip_param)

    elif args.command == "submit":
        submit(cfg, mode=args.mode)

    elif args.command == "analyse":
        analyse(cfg, tail=args.tail)

    elif args.command == "network":
        prepare_network(
            cfg,
            structures_dir=Path(args.structures_dir),
            mode=args.mode,
            dry_run=args.dry_run,
            skip_param=args.skip_param,
        )

    elif args.command == "network-submit":
        submit_network(cfg, mode=args.mode)

    elif args.command == "network-analyse":
        analyse_network(cfg, tail=args.tail)

    elif args.command == "network-visualize":
        visualize_network(
            network_path=Path(args.network),
            output_path=Path(args.output),
            structures_dir=Path(args.structures_dir),
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
