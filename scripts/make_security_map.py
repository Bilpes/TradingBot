"""Build a symbol -> Dhan security_id map from Dhan's official scrip master.

    python scripts/make_security_map.py --scrip-master dhan_scrip_master.csv \
        --out config/security_map.json

Refuses to write a partial map: a missing symbol is a loud failure now, not a
wrong stock bought later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingbot.data.dhan import SecurityResolutionError, resolve_security_ids
from tradingbot.markets.nse import NSE_UNIVERSE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrip-master", required=True,
                        help="path to Dhan's dhan_scrip_master.csv")
    parser.add_argument("--out", default="config/security_map.json")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="default: the full NSE universe in markets/nse.py")
    args = parser.parse_args()

    symbols = args.symbols or list(NSE_UNIVERSE)
    try:
        mapping = resolve_security_ids(symbols, args.scrip_master)
    except SecurityResolutionError as exc:
        print(f"REFUSED to write a partial map: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(sorted(mapping.items())), indent=2) + "\n")
    print(f"Wrote {len(mapping)} security ids to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
