# numpydoc ignore=GL08

import importlib.util
from pathlib import Path


def _load_postprocessor():  # numpydoc ignore=GL08
    script_path = (
        Path(__file__).parents[1] / "docs_scripts" / "postprocess_generated_markdown.py"
    )
    spec = importlib.util.spec_from_file_location(
        "postprocess_generated_markdown", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_postprocess_generated_markdown_adds_alt_text_paragraph():  # numpydoc ignore=GL08
    postprocessor = _load_postprocessor()

    result = postprocessor.postprocess_generated_markdown(
        '<div><img alt="Figure caption." height="400" src="plot.png" width="600"/></div>'
    )

    assert result == (
        '<div markdown="1"><img alt="Figure caption." src="plot.png"/>'
        "<p>Figure caption.</p></div>"
    )


def test_postprocess_generated_markdown_does_not_duplicate_alt_text_paragraph():  # numpydoc ignore=GL08
    postprocessor = _load_postprocessor()

    result = postprocessor.postprocess_generated_markdown(
        '<img alt="Figure caption." src="plot.png"/><p>Figure caption.</p>'
    )

    assert result == '<img alt="Figure caption." src="plot.png"/><p>Figure caption.</p>'
