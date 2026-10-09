import argparse
import logging

from .config import Config
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Run the full NIFTY 50 strategy research pipeline.")
    parser.add_argument("--refresh", action="store_true", help="re-download prices from Yahoo Finance")
    parser.add_argument("--skip-ml", action="store_true", help="skip the walk-forward ML section")
    parser.add_argument("--cost-bps", type=float, default=Config.cost_bps, help="one-way cost in bps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(Config(cost_bps=args.cost_bps), refresh=args.refresh, skip_ml=args.skip_ml)


if __name__ == "__main__":
    main()
