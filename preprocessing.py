"""Data preprocessing functions."""

import re

from config import EOS_TOKEN


def load_non_breaking_prefixes(filepath: str) -> list[str]:
    """Loads and processes non-breaking prefixes from a file.

    Args:
        filepath: The path to the file containing non-breaking prefixes.

    Returns:
        A list of non-breaking prefixes, each with a period added.
    """
    with open(filepath, "r", encoding="utf8") as file:
        lines = [line.strip() for line in file.readlines()]
    prefixes = [line + "." for line in lines if line != "" and "#" not in line]
    return prefixes


def sentence_boundary_disambiguation(corpus: str, non_breaking_prefixes: list[str]) -> str:
    """Removes all periods that do not indicate the end of a sentence.

    Args:
        corpus: The input text to be preprocessed.
        non_breaking_prefixes: A list of non-breaking prefixes, i.e. tokens where a period normally 
        does not indicate the end of a sentence.

    Returns:
        The preprocessed text with non-breaking periods removed.
    """
    corpus_cleaned = corpus
    # Mark non-breaking prefixes that are not at the end of a line
    for prefix in non_breaking_prefixes:
        corpus_cleaned = corpus_cleaned[:-2].replace(prefix, prefix + '$$$') + corpus_cleaned[-2:]
    
    # Mark periods that are immediately followed by an alphanumerical character
    corpus_cleaned = re.sub(r"\.(?=[a-zA-Z0-9])", ".$$$", corpus_cleaned)
    
    # Remove all marked non-breaking periods
    corpus_cleaned = re.sub(r"\.\$\$\$+", "", corpus_cleaned)
    
    # Remove multiple consecutive white spaces
    corpus_cleaned = re.sub(r"  +", " ", corpus_cleaned)

    # Add <EOS> token to all statements
    corpus_cleaned += " " + EOS_TOKEN
    return corpus_cleaned