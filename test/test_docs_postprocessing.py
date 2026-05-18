# numpydoc ignore=GL08

import importlib.util
from pathlib import Path


def _load_postprocessor():  # numpydoc ignore=GL08
    script_path = Path(__file__).parents[1] / "docs_scripts" / "add_markdown_to_divs.py"
    spec = importlib.util.spec_from_file_location("add_markdown_to_divs", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_markdown_to_divs_adds_alt_text_paragraph():  # numpydoc ignore=GL08
    postprocessor = _load_postprocessor()

    result = postprocessor.add_markdown_to_divs(
        '<div><img alt="Figure caption." height="400" src="plot.png" width="600"/></div>'
    )

    assert result == (
        '<div markdown="1"><img alt="Figure caption." src="plot.png"/>'
        "<p>Figure caption.</p></div>"
    )


def test_add_markdown_to_divs_does_not_duplicate_alt_text_paragraph():  # numpydoc ignore=GL08
    postprocessor = _load_postprocessor()

    result = postprocessor.add_markdown_to_divs(
        '<img alt="Figure caption." src="plot.png"/><p>Figure caption.</p>'
    )

    assert result == '<img alt="Figure caption." src="plot.png"/><p>Figure caption.</p>'
