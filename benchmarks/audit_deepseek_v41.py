"""离线检查 DeepSeek V4.1 checkpoint；只读文件头，不初始化 CUDA。"""

import argparse
import hashlib
import json
from pathlib import Path

from mysglang.modeling.deepseek_v41.checkpoint import audit_checkpoint
from mysglang.modeling.deepseek_v41.config import DeepSeekV41Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--config", type=Path, help="HF config 或 inference/config.json 的显式路径")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config or args.model / "config.json"
    config = DeepSeekV41Config.from_json(config_path)
    report = audit_checkpoint(
        args.model,
        config,
        tp_size=args.tp,
        text_only=args.text_only,
        include_draft=not args.no_draft,
    )
    report["config_path"] = str(config_path.resolve())
    report["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
