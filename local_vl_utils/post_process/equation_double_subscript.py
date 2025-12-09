import re

from loguru import logger


def try_fix_equation_double_subscript(latex: str, debug: bool = False) -> str:
    pattern = r"_\s*\{([^{}]|\{[^{}]*\})*\}\s*_\s*\{([^{}]|\{[^{}]*\})*\}"
    if not re.search(pattern, latex):
        return latex
    new_latex = re.sub(pattern, "", latex)
    if debug:
        logger.debug("Fixed equation double-subscript from '{}' to '{}'", latex, new_latex)
    return new_latex
