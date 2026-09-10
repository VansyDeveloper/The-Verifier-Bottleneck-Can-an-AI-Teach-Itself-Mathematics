from vbexp.polynomial import ac1, apply_program, ax1, is_order_sensitive, rev, sc2, sh1


def test_sh1_manual_quadratic():
    # P(x)=2+3x+4x^2; P(x+1)=9+11x+4x^2 = 9+0x+4x^2 mod 11
    assert sh1((2, 3, 4), 11) == (9, 0, 4)


def test_sc2_manual():
    assert sc2((2, 3, 4), 11) == (2, 6, 5)


def test_simple_operations():
    state = (1, 2, 3, 4)
    assert rev(state, 11) == (4, 3, 2, 1)
    assert ac1(state, 11) == (2, 2, 3, 4)
    assert ax1(state, 11) == (1, 3, 3, 4)


def test_noncommuting_shift_scale():
    state = (1, 2, 3)
    assert apply_program(state, ("SH1", "SC2"), 11) != apply_program(state, ("SC2", "SH1"), 11)
    assert is_order_sensitive(state, ("SH1", "SC2"), 11)
