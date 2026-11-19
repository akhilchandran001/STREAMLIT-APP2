import json
import re


def _synthesize_sentence_from_news_dict(item: dict) -> str:
    """Ensures raw Newspaper JSON objects without a 'sentence' key still work with TTS & pipeline."""
    parts = []
    hl = item.get("headline")
    if isinstance(hl, dict):
        hl_txt = " ".join(
            str(hl.get(k, "")).strip() for k in ("prefix", "highlight", "suffix") if hl.get(k)
        ).strip()
    else:
        hl_txt = str(hl or "").strip()

    if hl_txt:
        parts.append(hl_txt.replace("\n", " "))
    if item.get("subheadline"):
        parts.append(str(item["subheadline"]).strip())

    bp = item.get("body_paragraphs") or item.get("body")
    if not parts and bp:
        if isinstance(bp, list):
            parts.append(" ".join(str(p).strip() for p in bp[:2] if str(p).strip()))
        else:
            parts.append(str(bp).strip())

    return ". ".join(p.rstrip(".") for p in parts if p).strip()


def parse_input_script(raw_content: str) -> list[dict]:
    """
    Parses both .json and .txt inputs into a standardized scene list:
    - If valid JSON: parses directly and preserves all Newspaper/Whiteboard/Default fields.
    - If plain .txt: splits into lines and structures each line.
      Format: "Sentence | Prompt/Equation/Highlights | Headline | Subheadline"
    """
    cleaned = raw_content.strip()
    if not cleaned:
        raise ValueError("Script content is empty.")

    # 1. Try parsing as standard JSON
    if (cleaned.startswith("[") and cleaned.endswith("]")) or (cleaned.startswith("{") and cleaned.endswith("}")):
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list) and len(data) > 0:
                normalized = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    obj = dict(item)
                    if not str(obj.get("sentence", "")).strip():
                        obj["sentence"] = _synthesize_sentence_from_news_dict(obj)
                    if not str(obj.get("query", "")).strip():
                        obj["query"] = obj.get("sentence", "")
                    normalized.append(obj)
                if normalized:
                    return normalized
        except json.JSONDecodeError:
            pass

    # 2. Parse plain text line-by-line
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    scenes = []

    for idx, line in enumerate(lines, start=1):
        # Format support: "Sentence | Prompt or Equation or Highlights | Headline | Subheadline"
        parts = [p.strip() for p in line.split("|")]
        sentence = parts[0]
        extra = parts[1] if len(parts) > 1 else ""
        headline_raw = parts[2] if len(parts) > 2 else ""
        subheadline_raw = parts[3] if len(parts) > 3 else ""

        scene_obj = {
            "sentence": sentence,
            "query": extra or sentence,
            "prompt": f"black and white clean vector line art sketch of {extra or sentence}, white background, coloring book style"
        }

        # Support comma/semicolon-separated yellow highlights in Newspaper mode
        if extra and "latex:" not in extra.lower() and "\\" not in extra:
            highlights = [h.strip() for h in re.split(r"[,;]+", extra) if h.strip()]
            if highlights:
                scene_obj["yellow_highlights"] = highlights

        # Detect LaTeX equation
        if "$" in sentence or "\\" in sentence or "latex:" in extra.lower():
            eq_text = extra.replace("latex:", "").strip() if "latex:" in extra.lower() else sentence
            scene_obj["equation"] = eq_text
            scene_obj["formula_name"] = headline_raw or f"Equation {idx}"

        if headline_raw:
            scene_obj["headline"] = {"prefix": "", "highlight": headline_raw, "suffix": ""}

        if subheadline_raw:
            scene_obj["subheadline"] = subheadline_raw

        scenes.append(scene_obj)

    if not scenes:
        raise ValueError("Could not extract any scenes from the provided text.")

    return scenes
