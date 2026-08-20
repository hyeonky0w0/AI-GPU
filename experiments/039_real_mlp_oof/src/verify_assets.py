"""039 full 실행 전에 Network Volume 자산의 존재와 SHA-256을 검증한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import sha256


EXP = Path(__file__).resolve().parents[1]


def verify(asset_root: Path, contract_path: Path = EXP / "configs" / "assets.json") -> dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = contract["sha256"]
    missing = [relative for relative in expected if not (asset_root / relative).is_file()]
    if missing:
        details = "\n".join(f"  - {asset_root / relative}" for relative in missing)
        raise FileNotFoundError(f"039 필수 Network Volume 자산이 없습니다:\n{details}")

    checked: dict[str, dict[str, object]] = {}
    mismatches: list[str] = []
    for relative, wanted in expected.items():
        if len(wanted) != 64:
            raise ValueError(f"assets.json SHA-256 계약이 64자리가 아닙니다: {relative}={wanted!r}")
        path = asset_root / relative
        actual = sha256(path)
        checked[relative] = {"bytes": path.stat().st_size, "sha256": actual}
        if actual != wanted:
            mismatches.append(f"  - {relative}: expected={wanted}, actual={actual}")
    if mismatches:
        raise ValueError("039 Network Volume 자산 SHA-256 불일치:\n" + "\n".join(mismatches))
    return {"status": "PASS", "asset_root": str(asset_root.resolve()), "assets": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True, help="assets/, router_sources/를 포함한 루트")
    parser.add_argument("--contract", default=str(EXP / "configs" / "assets.json"))
    args = parser.parse_args()
    print(json.dumps(verify(Path(args.asset_root), Path(args.contract)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
