# numpydoc ignore=GL08

from pipeline.placeholder import add


def test_add():  # numpydoc ignore=GL08
    want = 2
    got = add(1, 1)
    assert want == got
