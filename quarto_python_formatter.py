# numpydoc ignore=GL08
#!/usr/bin/env python3
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Match


def format_python_code(code: str) -> str:  # numpydoc ignore=RT01
    """Format Python code using Black."""
    try:
        result = subprocess.run(
            ["black", "-", "-q"],
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


def replace_code_block(match: Match[str]) -> str:  # numpydoc ignore=RT01
    """Replace code block with formatted version."""
    return f"{match.group(1)}\n{format_python_code(match.group(2))}{match.group(3)}"


def process_file(filepath: Path) -> None:  # numpydoc ignore=RT01
    """Process the given file, formatting Python code blocks."""
    python_code_block_pattern = r"(```\{python\})(.*?)(```)"
    try:
        content = filepath.read_text()
        formatted_content = re.sub(
            python_code_block_pattern,
            replace_code_block,
            content,
            flags=re.DOTALL,
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write(formatted_content)
            temp_filepath = Path(temp_file.name)

        temp_filepath.replace(filepath)
        print(
            f"Python code cells in {filepath} have been formatted using Black."
        )
    except IOError as e:
        print(f"Error processing file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python format_qmd_python.py <filename1.qmd> [filename2.qmd ...]"
        )
        sys.exit(1)

    for filepath in sys.argv[1:]:
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File {path} does not exist.", file=sys.stderr)
            continue
        process_file(path)
