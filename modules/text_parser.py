import json
import re

def parse_input_script(raw_content: str) -> list[dict]:
    """
    Parses both .json and .txt inputs into a standardized scene list:
    - If valid JSON: parses directly.
    - If plain .txt: splits into lines/paragraphs and structures each line.
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
                return data
        except json.JSONDecodeError:
            pass

    # 2. Parse plain text line-by-line
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    scenes = []

    for idx, line in enumerate(lines, start=1):
        # Format support: "Sentence | Prompt or Equation | Headline"
        parts = [p.strip() for p in line.split("|")]
        sentence = parts[0]
        extra = parts[1] if len(parts) > 1 else ""
        headline_raw = parts[2] if len(parts) > 2 else ""

        scene_obj = {
            "sentence": sentence,
            "query": sentence,
            "prompt": f"black and white clean vector line art sketch of {extra or sentence}, white background, coloring book style"
        }

        # Detect LaTeX equation
        if "$" in sentence or "\\" in sentence or "latex:" in extra.lower():
            eq_text = extra.replace("latex:", "").strip() if "latex:" in extra.lower() else sentence
            scene_obj["equation"] = eq_text
            scene_obj["formula_name"] = headline_raw or f"Equation {idx}"

        if headline_raw:
            scene_obj["headline"] = {"prefix": "", "highlight": headline_raw, "suffix": ""}

        scenes.append(scene_obj)

    if not scenes:
        raise ValueError("Could not extract any scenes from the provided text.")

    return scenes
