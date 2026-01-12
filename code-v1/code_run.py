from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# --- Integrated Utility Functions (from 2.py) ---

def escape_newlines_in_strings(text: str) -> str:
    out = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue
        if ch == '\\':
            out.append(ch)
            escape = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            i += 1
            continue
        if in_string and ch in ('\r', '\n'):
            # collapse CRLF or LF in strings
            if ch == '\r' and i + 1 < len(text) and text[i+1] == '\n':
                i += 2
            else:
                i += 1
            out.append('\\n')
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def normalize_fragment(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove spacing between digits if any specific format requires it, 
    # but strictly generally valid JSON doesn't need this regex. 
    # Kept for compatibility with original utils.
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    text = escape_newlines_in_strings(text)
    # Also fix common trailing comma issues
    text = re.sub(r",\s*(\]|\})", r"\1", text)
    text = re.sub(r"'", '"', text)
    return text


def extract_valid_objects(raw_text: str) -> List[Dict[str, Any]]:
    s = raw_text
    depth = 0
    in_string = False
    escape = False
    start = None
    objs: List[Dict[str, Any]] = []
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                frag = s[start:i+1]
                fixed = normalize_fragment(frag)
                try:
                    obj = json.loads(fixed)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except Exception:
                    pass
                start = None
    return objs


def load_dataset_robust(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass
    objs = extract_valid_objects(raw)
    rows: List[Dict[str, Any]] = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        md = o.get("Metadata") or {}
        if isinstance(md, dict) and (md.get("Trajectory ID") or md.get("Trace ID")):
            rows.append(o)
    return rows

# --- End Utilities ---


# Constants
MAX_RETRIES = 6
BACKOFF_BASE = 2.0
BACKOFF_MAX = 30.0

SYSTEM_PROMPT = (
    "You are a pragmatic, helpful assistant. Below is the user's objective and an Available Tools List (each tool includes concrete return data, already provided).\n"
    "Please output only the final answer in English:\n"
    "- Output natural language only; do not output JSON.\n"
    "- Do not say you will call tools, and do not repeat the tool list verbatim.\n"
    "- Use the provided Return Data where relevant to reason and answer.\n"
    "- If information is insufficient, state reasonable assumptions and give safe alternatives."
)


def first(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def find_tools(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("Available Tools List", "Available Tools", "Tool List", "tools"):
        v = entry.get(key)
        if isinstance(v, list):
            return v
    return []


def format_tool_items(tools: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for t in tools or []:
        name = first(t, "Tool Name", "name") or ""
        tid = first(t, "Tool ID", "ID", "id") or ""
        desc = first(t, "Tool Description", "description") or ""
        params = first(t, "Parameters", "parameters") or []
        if not isinstance(params, list):
            params = [params] if params else []
        norm_params: List[str] = []
        for p in params:
            if isinstance(p, str):
                norm_params.append(p)
            else:
                try:
                    norm_params.append(json.dumps(p, ensure_ascii=False))
                except Exception:
                    norm_params.append(str(p))
        raw_ret = first(t, "Return Data", "return", "data")
        ret_text = None
        if raw_ret is not None:
            if isinstance(raw_ret, str):
                try:
                    obj = json.loads(raw_ret)
                    ret_text = json.dumps(obj, ensure_ascii=False, indent=2)
                except Exception:
                    try:
                        obj = json.loads(normalize_fragment(raw_ret))
                        ret_text = json.dumps(obj, ensure_ascii=False, indent=2)
                    except Exception:
                        ret_text = raw_ret
            else:
                try:
                    ret_text = json.dumps(raw_ret, ensure_ascii=False, indent=2)
                except Exception:
                    ret_text = str(raw_ret)
        block = [
            f"- Tool Name: {name}",
            f"  Tool ID: {tid}",
            f"  Description: {desc}",
            f"  Parameters: {', '.join(norm_params)}",
        ]
        if ret_text:
            block.append("  Return Data:")
            for line in ret_text.splitlines():
                block.append("    " + line)
        blocks.append("\n".join(block))
    return "\n".join(blocks)


def call_bigmodel(
    url: str,
    authorization: str,
    model: str,
    system_prompt: str,
    user_content: str,
    timeout: float = 30.0,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": authorization if authorization.lower().startswith("bearer") else f"Bearer {authorization}",
    }
    base_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "top_p": 0.9,
    }

    def _stream_once() -> Optional[str]:
        payload = dict(base_payload)
        payload["stream"] = True
        with requests.post(url, headers=headers, json=payload, timeout=(timeout, timeout), stream=True) as r:
            if r.status_code == 429:
                return None
            r.raise_for_status()
            content_parts: List[str] = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    try:
                        line = line.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                line = line.strip()
                if line == "data: [DONE]":
                    break
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        continue
                    choices = obj.get("choices") or []
                    for ch in choices:
                        delta = ch.get("delta") or {}
                        if isinstance(delta, dict):
                            ct = delta.get("content")
                            if isinstance(ct, str):
                                content_parts.append(ct)
            txt = "".join(content_parts).strip()
            return txt

    def _nonstream_once() -> Optional[str]:
        payload = dict(base_payload)
        payload["stream"] = False
        r = requests.post(url, headers=headers, json=payload, timeout=(timeout, timeout))
        if r.status_code == 429:
            return None
        r.raise_for_status()
        data = r.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            txt = _stream_once()
            if txt:
                return txt
        except Exception as e:
            last_exc = e
        
        try:
            txt2 = _nonstream_once()
            if txt2:
                return txt2
        except Exception as e:
            last_exc = e
            
        if attempt < MAX_RETRIES - 1:
            wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
            time.sleep(wait)
            continue
    if last_exc:
        raise last_exc
    return ""


def main() -> int:
    ap = argparse.ArgumentParser("Run DeepSeek-style prompt")
    ap.add_argument("--dataset", type=Path, default=Path("dataset.json"))
    ap.add_argument("--output", type=Path, default=Path("output_results.json"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--call-sleep", type=float, default=1.0)
    args = ap.parse_args()

    # Use environment variables for authentication
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    authorization = os.getenv("BIGMODEL_AUTH", "")
    model = "glm-4.6"

    if not authorization:
        print("Error: environment variable BIGMODEL_AUTH is not set.")
        return 1

    if not args.dataset.exists():
        print(f"Error: dataset {args.dataset} not found.")
        return 1

    data = load_dataset_robust(args.dataset)
    if args.limit is not None and args.limit > 0:
        data = data[: args.limit]

    os.makedirs(args.output.parent, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for idx, entry in enumerate(data, 1):
        if args.call_sleep and args.call_sleep > 0:
            time.sleep(float(args.call_sleep) + random.uniform(0, 0.1))

        meta = first(entry, "Metadata") or {}
        tid = first(meta, "Trajectory ID", "Trace ID") or ""
        user_goal = first(meta, "User Objective") or ""
        tools = find_tools(entry)

        tools_block = format_tool_items(tools)
        user_content = (
            f"User Objective:\n{user_goal}\n\n"
            f"Available Tools List (with real Return Data; no further calls will be made):\n{tools_block}\n\n"
            f"Please provide the final answer now."
        )

        output = ""
        err: Optional[str] = None
        try:
            output = call_bigmodel(url, authorization, model, SYSTEM_PROMPT, user_content)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        rec = {
            "Trajectory ID": tid,
            "Input": user_content,
            "Output": output,
        }
        if err:
            rec["Error"] = err
        results.append(rec)

        try:
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        status = "OK" if output else ("ERR:" + (err.split(":", 1)[0] if err else ""))
        print(f"[{idx}/{len(data)}] {tid} -> {status}")

    print(f"\nWrote {len(results)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())