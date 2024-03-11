from pipeline.placeholder import add


def test_add():
    want = 2
    got = add(1, 1)
    assert want == got
