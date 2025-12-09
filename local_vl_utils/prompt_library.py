"""Centralized prompt templates shared across VLM OCR clients."""

from __future__ import annotations


STRUCTURED_JSON_EXAMPLE = (
    "[\n"
    '  {"type": "text", "bbox": [0.075, 0.15, 0.184, 0.228], "angle": null, "content": "防伪标识"}\n'
    "]"
)


def _structured_prompt_suffix(block_type: str | None, content_desc: str) -> str:
    """Helper used by Qwen prompts to enforce structured JSON responses."""
    type_instruction = (
        f'Use the same schema but set "type" to "{block_type}". '
        if block_type
        else 'Use the same schema and set "type" to the block type you are transcribing '
        '(e.g., "text", "code"). '
    )
    return (
        "\nOutput strictly as a JSON array of objects with the keys \"type\", \"bbox\", \"angle\", \"content\". Example:\n"
        f"{STRUCTURED_JSON_EXAMPLE}\n"
        f"{type_instruction}"
        "Provide normalized bbox coordinates (0.0-1.0) relative to the supplied crop, set \"angle\" to null or 0/90/180/270, "
        f"and place {content_desc} inside \"content\". Return [] if nothing is recognizable. Do not include explanations "
        "outside the JSON array."
    )


QWEN_DEFAULT_PROMPTS: dict[str, str] = {
    "table": (
        "\nDocument Table Recognition.\n"
        "Convert the cropped table into HTML format. Use proper <table>, <tr>, <th>, <td> tags. "
        "Preserve header rows (use <th>), merged cells (use colspan/rowspan attributes), "
        "numeric precision, units, and the original row/column ordering. "
        "Return clean HTML without <!DOCTYPE> or surrounding tags."
        + _structured_prompt_suffix("table", "the HTML table string described above")
    ),
    "equation": (
        "\nFormula Recognition.\n"
        "Return the mathematical expression verbatim in LaTeX, keeping fractions, superscripts, subscripts, "
        "Greek letters, and spacing. Do not paraphrase or simplify."
        + _structured_prompt_suffix("equation", "the LaTeX string")
    ),
    "title": (
        "\nTitle Recognition.\n"
        "Transcribe the heading exactly, keeping numbering, punctuation, emoji, and capitalization so the "
        "reading order is preserved."
        + _structured_prompt_suffix("title", "the heading text")
    ),
    "list": (
        "\nList Recognition.\n"
        "Output the list in plain text while keeping bullet/number markers and indentation (use two spaces "
        "per nested level). Preserve checkbox or dash prefixes."
        + _structured_prompt_suffix("list", "the full list text with markers")
    ),
    "image": (
        "\nImage Description.\n"
        "Describe the image content clearly and concisely. Include key visual elements, objects, people, "
        "text visible in the image, charts, diagrams, or any important details. Keep the description factual."
        + _structured_prompt_suffix("image", "the image description")
    ),
    "handwritten": (
        "\nHandwritten Content Recognition.\n"
        "Transcribe the handwritten text as-is. Maintain line breaks and mark illegible characters with [?] "
        "to reflect uncertainty."
        + _structured_prompt_suffix("handwritten", "the handwritten transcription")
    ),
    "seal": (
        "\nSeal Recognition.\n"
        "Read the seal/stamp or signature chop exactly. Output only the deciphered characters (typically "
        "short phrases, institution names, or dates) with no extra commentary."
        + _structured_prompt_suffix("seal", "the deciphered seal text")
    ),
    "[default]": (
        "\nDocument Text Recognition.\n"
        "Transcribe the selected region verbatim, keeping punctuation, inline math, emoji, and formatting "
        "markers. Preserve natural line breaks to retain the document reading order."
        + _structured_prompt_suffix(None, "the verbatim transcription")
    ),
    "[layout]": (
        "\nDocument Layout Detection.\n"
        "Analyze the full page and detect all layout elements. Identify text blocks, titles, lists, tables, "
        "images, handwritten content, and seals. Maintain the natural reading order of the document.\n"
        "Return ONLY a valid JSON array. Each element MUST be exactly "
        "{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"TYPE\"}. Use integers 0-1000 for bbox coordinates with x1<x2 "
        "and y1<y2. Do not emit extra keys, Markdown, prose, or trailing commas.\n"
        "Allowed TYPE values (case-insensitive): text, title, list, table, image, handwritten, seal. Map any "
        "stamp-related prediction to \"seal\". If unsure, downgrade to text. Sort entries by true reading order "
        "(top-to-bottom, then left-to-right). Output [] when no elements exist.\n"
        "If a region is rotated, append the rotation token (<|rotate_right|>, <|rotate_left|>, <|rotate_down|>) "
        "inside the label string so downstream parsing can recover the orientation."
    ),
}


DEEPSEEK_DEFAULT_PROMPTS: dict[str, str] = {
    "table": (
        "<image>\n"
        "Parse the table and convert to HTML. Use <table>, <tr>, <th>, <td> tags. "
        "Preserve header rows and merged cells. Only output the HTML table."
    ),
    "equation": (
        "<image>\n"
        "Extract the mathematical formula in LaTeX format. "
        "Keep fractions, superscripts, subscripts, Greek letters. Only output the LaTeX code."
    ),
    "title": (
        "<image>\n"
        "Free OCR. Transcribe the heading text exactly with numbering and punctuation. "
        "Only output the raw text."
    ),
    "list": (
        "<image>\n"
        "Free OCR. Extract the list with bullet/number markers and indentation. "
        "Only output the raw text."
    ),
    "handwritten": (
        "<image>\n"
        "Free OCR. Transcribe the handwritten text. Use [?] for illegible characters. "
        "Only output the raw text."
    ),
    "seal": (
        "<image>\n"
        "Free OCR. Read the seal/stamp text. "
        "Only output the deciphered characters."
    ),
    "image": (
        "<image>\n"
        "Describe this image. Focus on visible key elements, objects, text, and diagrams."
    ),
    "[default]": (
        "<image>\n"
        "Free OCR. Transcribe the text verbatim with punctuation and line breaks. "
        "Only output the raw text."
    ),
    "[layout]": (
        "<image>\n"
        "<|grounding|>Detect all text blocks, titles, lists, tables, images, handwritten content, and seals in this document."
    ),
    "[find]": (
        "<image>\n"
        "<|grounding|>Locate and extract all visible elements in the image."
    ),
}


HUNYUAN_STABILITY_SUFFIX = " 直接输出结果，不要包含任何开场白、结束语或Markdown代码块标记（如```html）。"

HUNYUAN_DEFAULT_PROMPTS: dict[str, str] = {
    "table": "\n提取表格内容，请返回完整的 HTML 表格结构。" + HUNYUAN_STABILITY_SUFFIX,
    "equation": "\n识别该区域的公式，输出准确的 LaTeX。" + HUNYUAN_STABILITY_SUFFIX,
    "title": "\n识别标题内容，保持原有的编号、标点、格式。" + HUNYUAN_STABILITY_SUFFIX,
    "list": "\n识别列表内容，保持原有的项目符号或编号，嵌套用两个空格缩进。" + HUNYUAN_STABILITY_SUFFIX,
    "image": "\n描述图片内容，清晰、简洁地说明关键视觉元素、物体、人物或图表细节。" + HUNYUAN_STABILITY_SUFFIX,
    "handwritten": "\n识别手写内容，保持原始换行，难以辨认的字符用[?]占位。" + HUNYUAN_STABILITY_SUFFIX,
    "seal": "\n识别印章或签章上的文字，只输出精确的文字内容。" + HUNYUAN_STABILITY_SUFFIX,
    "[default]": "\n识别文本内容并保持原有的行文顺序。" + HUNYUAN_STABILITY_SUFFIX,
    "[layout]": (
        "\n检测文档布局，按照阅读顺序标注每个区域的类型和位置。"
        "类型必须从{text, title, list, table, image, handwritten, seal}中选择，"
        "手写内容必须标注为handwritten，印章/签章为seal。"
    ),
}


MINERU_DEFAULT_PROMPTS: dict[str, str] = {
    "table": (
        "\nTable Recognition: Convert the table to HTML format with proper <table>, <tr>, <th>, <td> tags. "
        "Preserve headers and merged cells."
    ),
    "equation": "\nFormula Recognition:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}


__all__ = [
    "STRUCTURED_JSON_EXAMPLE",
    "QWEN_DEFAULT_PROMPTS",
    "DEEPSEEK_DEFAULT_PROMPTS",
    "HUNYUAN_STABILITY_SUFFIX",
    "HUNYUAN_DEFAULT_PROMPTS",
    "MINERU_DEFAULT_PROMPTS",
]
