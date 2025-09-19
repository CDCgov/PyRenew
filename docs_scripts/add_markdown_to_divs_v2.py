import sys
import re
from pathlib import Path

def add_markdown_to_divs(text: str) -> str:
    """Only process actual HTML div tags, preserve everything else."""
    
    # Pattern to match actual HTML div tags (not in code blocks)
    def replace_div(match):
        div_tag = match.group(0)
        # If markdown="1" not already present, add it
        if 'markdown=' not in div_tag:
            # Insert markdown="1" after <div
            div_tag = div_tag.replace('<div', '<div markdown="1"', 1)
        return div_tag
    
    # Only match divs that are actual HTML (not in code blocks)
    # This regex looks for <div at the start of a line (not indented)
    pattern = r'^<div(?:\s+[^>]*)?>'
    
    lines = text.split('\n')
    in_code_block = False
    result = []
    
    for line in lines:
        # Track if we're in a code block
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            result.append(line)
        elif in_code_block:
            # Don't process lines inside code blocks
            result.append(line)
        elif line.strip().startswith('<div'):
            # Process div tags outside of code blocks
            result.append(re.sub(pattern, replace_div, line))
        else:
            result.append(line)
    
    return '\n'.join(result)

if __name__ == "__main__":
    target = Path(sys.argv[1])
    if target.is_file():
        text = target.read_text(encoding="utf-8")
        updated = add_markdown_to_divs(text)
        target.write_text(updated, encoding="utf-8")
        print(f"Processed {target}")
    elif target.is_dir():
        for f in target.rglob("*.md"):
            text = f.read_text(encoding="utf-8")
            updated = add_markdown_to_divs(text)
            f.write_text(updated, encoding="utf-8")
            print(f"Processed {f}")
