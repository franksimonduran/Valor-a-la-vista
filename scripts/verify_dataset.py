"""Audita el piloto local TACO sin entrenar, modificar archivos ni usar la red."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from valor_a_la_vista.dataset import audit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = audit_dataset(args.dataset)
    print(json.dumps({"dataset": str(args.dataset.resolve()), **result}, ensure_ascii=False, indent=2))
    print("Auditoría técnica correcta; no sustituye la revisión visual, de derechos ni la validación del modelo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
