from vbexp.polynomial import apply_program
from vbexp.search import all_solutions, shortest_solutions


def test_shortest_solutions_find_witness():
    start = (1, 2, 3)
    program = ("SH1", "SC2")
    target = apply_program(start, program, 11)
    shortest = shortest_solutions(start, target, 11, ("SH1", "SC2", "REV", "AC1", "AX1"), 2)
    assert shortest
    assert len(shortest[0]) == 2
    assert program in all_solutions(start, target, 11, ("SH1", "SC2", "REV", "AC1", "AX1"), 2)
