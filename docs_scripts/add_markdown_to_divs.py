# numpydoc ignore=GL08

import sys
from pathlib import Path

from bs4 import BeautifulSoup


def add_markdown_to_divs(html: str) -> str:  # numpydoc ignore=GL08
    soup = BeautifulSoup(html, "html.parser")
    for div in soup.find_all("div"):
        if "markdown" not in div.attrs:
            div["markdown"] = "1"
    return soup.decode(formatter=None)


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
