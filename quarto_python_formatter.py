#!/usr/bin/env python3
# numpydoc ignore=GL08
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Match


def format_python_code(
    code: str, black_args: List[str]
) -> str:  # numpydoc ignore=RT01
    """Format Python code using Black with custom arguments."""
    try:
        cmd = ["black", "-"] + black_args
        result = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        print(
            "Error: Failed to format Python code with Black.", file=sys.stderr
        )
        return code


def replace_code_block(
    match: Match[str], black_args: List[str]
) -> str:  # numpydoc ignore=RT01
    """Replace code block with formatted version."""
    return f"{match.group(1)}\n{format_python_code(match.group(2), black_args)}{match.group(3)}"


def process_file(
    filepath: Path, black_args: List[str]
) -> None:  # numpydoc ignore=RT01
    """Process the given file, formatting Python code blocks."""
    python_code_block_pattern = r"(```\{python\})(.*?)(```)"
    try:
        content = filepath.read_text()
        formatted_content = re.sub(
            python_code_block_pattern,
            lambda m: replace_code_block(m, black_args),
            content,
            flags=re.DOTALL,
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write(formatted_content)
            temp_filepath = Path(temp_file.name)

        temp_filepath.replace(filepath)
    except IOError as e:
        print(f"Error processing file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            'Usage: python quarto_python_formatter.py "BLACK_ARGS" <filename1.qmd> [filename2.qmd ...]'
        )
        sys.exit(1)

    black_args = sys.argv[1].split()

    missing_files = [file for file in sys.argv[2:] if not Path(file).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Error: The following file(s) do not exist: {', '.join(missing_files)}."
        )
    for filepath in sys.argv[2:]:
        path = Path(filepath)
        process_file(path, black_args)
