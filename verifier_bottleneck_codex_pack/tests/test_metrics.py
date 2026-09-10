from vbexp.metrics import empirical_pass_at_k, signature_entropy, unbiased_pass_at_k


def test_pass_at_k():
    assert empirical_pass_at_k([False, True, False], 2) == 1.0
    assert unbiased_pass_at_k(10, 0, 1) == 0.0
    assert unbiased_pass_at_k(10, 10, 5) == 1.0


def test_entropy():
    assert signature_entropy([]) == 0.0
    assert signature_entropy([("A",), ("A",)]) == 0.0
