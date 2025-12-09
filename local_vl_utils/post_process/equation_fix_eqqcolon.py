import re

from loguru import logger


def try_fix_equation_eqqcolon(latex: str, debug: bool = False) -> str:
    new_latex = re.sub(r"\\eqqcolon", "=:", latex)
    new_latex = re.sub(r"\\coloneqq", ":=", new_latex)
    if debug and new_latex != latex:
        logger.debug("Fixed equation eq-colon from '{}' to '{}'", latex, new_latex)
    return new_latex
