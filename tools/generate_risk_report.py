"""Generate human-readable home-risk reports from fusion JSON with DeepSeek.

The script deliberately sends only report-relevant evidence to the language
model.  It never sends segmentation contours or image data, and it preserves
the original score and level rather than asking the model to rescore them.

Examples
--------
Copy .env.example to .env, set DEEPSEEK_API_KEY, then run:

    python tools/generate_risk_report.py \
        --input res_indoor_risk_v2/analysis/ADE_val_00001971_features.json

For a directory of feature JSON files:

    python tools/generate_risk_report.py \
        --input res_indoor_risk_v2/analysis \
        --output-dir res_indoor_risk_v2/reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
REPORT_VERSION = "1.0"

REPORT_FEATURE_KEYS = (
    "corridor_valid",
    "corridor_fallback_used",
    "corridor_obstacle_occupancy",
    "yolo_corridor_occupancy",
    "narrowest_passage_width_ratio",
    "corridor_slippery_ratio",
    "step_threshold_count",
    "low_light_flag",
)

SYSTEM_PROMPT = """你是居家环境跌倒风险报告助手。你只能依据用户提供的风险分析JSON生成报告。

严格规则：
1. 不得修改、重算或质疑 source_risk 中的 score、level 和 component_scores。
2. 不得编造JSON中不存在的危险、位置、检测结果、医疗结论或紧急情况。
3. 对模型检测结果使用“检测到”或“疑似”；不得将其写成已被人工确认的事实。
4. 若 quality 或 evidence 存在可靠性限制，必须在 reliability_note 中说明。
5. 建议应具体、可执行，按立即处理、尽快改善、日常维护排序；没有依据的优先级不要输出。
6. 面向老人和家属，使用简洁自然的中文，不输出技术术语堆砌。

只输出一个合法JSON对象，且字段必须恰好为：
{
  "title": "",
  "risk_summary": "",
  "key_evidence": [""],
  "actions": [
    {"priority": "立即处理|尽快改善|日常维护", "action": "", "basis": ""}
  ],
  "reliability_note": ""
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="A feature JSON file or a directory containing *_features.json files.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for *_report.json files. Defaults beside a single input or to <input>/reports.",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--dotenv", default=str(PROJECT_ROOT / ".env"),
        help="Local .env file used only when the API-key environment variable is absent.",
    )
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def load_dotenv_value(path: Path, key: str) -> str | None:
    """Read one simple KEY=value entry without adding a dependency."""
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            return value.strip().strip("\"'")
    return None


def get_api_key(args: argparse.Namespace) -> str:
    key = os.environ.get(args.api_key_env)
    if not key:
        key = load_dotenv_value(Path(args.dotenv), args.api_key_env)
    if not key or key == "replace_with_your_deepseek_api_key":
        raise RuntimeError(
            "Missing DeepSeek API key. Copy .env.example to .env, fill in "
            "DEEPSEEK_API_KEY, or set the environment variable before running."
        )
    return key


def select_feature_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError("Input path does not exist: {}".format(input_path))
    yield from sorted(input_path.glob("*_features.json"))


def build_report_payload(source: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    """Keep the evidence audit-friendly and compact before the API request."""
    features = source.get("features", {})
    quality = source.get("quality", {})
    risk = source.get("risk", {})
    detections = []
    for item in source.get("yolo_detections", []):
        detections.append({
            "class_name": item.get("class_name"),
            "confidence": item.get("confidence"),
            "in_corridor": item.get("in_corridor"),
        })
    return {
        "image_id": image_id,
        "source_risk": {
            "score": risk.get("score"),
            "level": risk.get("level"),
            "scoring_version": risk.get("scoring_version"),
            "reasons": risk.get("reasons", []),
            "suggestions": risk.get("suggestions", []),
            "component_scores": risk.get("component_scores", {}),
        },
        "quality": {
            "mask_valid": quality.get("mask_valid"),
            "corridor_valid": quality.get("corridor_valid"),
            "corridor_fallback_used": quality.get("corridor_fallback_used"),
            "slippery_surface_reliability": quality.get("slippery_surface_reliability"),
            "step_threshold_reliability": quality.get("step_threshold_reliability"),
            "absolute_metric_scale_available": quality.get("absolute_metric_scale_available"),
        },
        "evidence": {
            "features": {key: features.get(key) for key in REPORT_FEATURE_KEYS},
            "yolo_detections": detections,
        },
    }


def call_deepseek(payload: Dict[str, Any], api_key: str, model: str) -> Dict[str, Any]:
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请依据以下JSON生成报告，且只输出JSON：\n" + json.dumps(
                    payload, ensure_ascii=False
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": 1200,
    }
    encoded = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    request = Request(
        DEEPSEEK_URL,
        data=encoded,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content or not content.strip():
        raise ValueError("DeepSeek returned empty content")
    result = json.loads(content)
    required = {"title", "risk_summary", "key_evidence", "actions", "reliability_note"}
    if set(result) != required:
        raise ValueError("DeepSeek response does not match the report schema")
    return result


def generate_report(
    source_file: Path, output_dir: Path, api_key: str, args: argparse.Namespace
) -> Path:
    source = json.loads(source_file.read_text(encoding="utf-8"))
    image_id = source_file.name.removesuffix("_features.json")
    payload = build_report_payload(source, image_id)
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            report = call_deepseek(payload, api_key, args.model)
            break
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == args.retries:
                raise RuntimeError(
                    "DeepSeek failed for {} after {} attempts: {}".format(
                        source_file.name, attempt + 1, error
                    )
                ) from error
            time.sleep(2 ** attempt)
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError("Unexpected retry state") from last_error

    output = {
        "report_version": REPORT_VERSION,
        "source_file": source_file.name,
        "model": args.model,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_risk": payload["source_risk"],
        "report": report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "{}_report.json".format(image_id)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    files = list(select_feature_files(input_path))
    if not files:
        raise FileNotFoundError("No *_features.json files found in: {}".format(input_path))
    output_dir = Path(args.output_dir) if args.output_dir else (
        input_path.parent if input_path.is_file() else input_path / "reports"
    )
    api_key = get_api_key(args)
    for source_file in files:
        output_path = generate_report(source_file, output_dir, api_key, args)
        print("Created {}".format(output_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
