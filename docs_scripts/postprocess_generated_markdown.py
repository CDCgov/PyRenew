# numpydoc ignore=GL08

import sys
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag


def _next_tag(element: Tag) -> Tag | None:  # numpydoc ignore=GL08
    for sibling in element.next_siblings:
        if isinstance(sibling, Tag):
            return sibling
        if str(sibling).strip():
            return None
    return None


def _add_alt_text_paragraphs(soup: BeautifulSoup) -> None:  # numpydoc ignore=GL08
    for img in soup.find_all("img"):
        alt_text = img.get("alt", "").strip()
        if not alt_text:
            continue

        next_tag = _next_tag(img)
        if (
            next_tag
            and next_tag.name == "p"
            and next_tag.get_text(strip=True) == alt_text
        ):
            continue

        caption = soup.new_tag("p")
        caption.string = alt_text
        img.insert_after(caption)


def postprocess_generated_markdown(html: str) -> str:  # numpydoc ignore=GL08
    soup = BeautifulSoup(html, "html.parser")
    for div in soup.find_all("div"):
        if "markdown" not in div.attrs:
            div["markdown"] = "1"
    for img in soup.find_all("img"):
        img.attrs.pop("width", None)
        img.attrs.pop("height", None)
    _add_alt_text_paragraphs(soup)
    return soup.decode(formatter=None)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    if target.is_file():
        text = target.read_text(encoding="utf-8")
        updated = postprocess_generated_markdown(text)
        target.write_text(updated, encoding="utf-8")
        print(f"Processed {target}")
    elif target.is_dir():
        for f in target.rglob("*.md"):
            text = f.read_text(encoding="utf-8")
            updated = postprocess_generated_markdown(text)
            f.write_text(updated, encoding="utf-8")
            print(f"Processed {f}")
