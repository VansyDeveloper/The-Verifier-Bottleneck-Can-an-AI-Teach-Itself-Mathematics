from vbexp.polynomial import apply_program
from vbexp.task import Task
from vbexp.verifier import verify


def test_apply_verifier():
    target = apply_program((1, 2, 3), ("SH1",), 11)
    task = Task("a", "apply", "test", 11, 2, (1,2,3), ("SH1","SC2","REV","AC1","AX1"), target=target, program=("SH1",))
    result = verify(task, f"RESULT: [{', '.join(map(str, target))}]")
    assert result.parse_ok and result.is_correct


def test_plan_verifier():
    start = (1, 2, 3)
    target = apply_program(start, ("SH1", "SC2"), 11)
    task = Task("p", "plan", "test", 11, 2, start, ("SH1","SC2","REV","AC1","AX1"), target=target, max_steps=2)
    result = verify(task, "PROGRAM: SH1 SC2")
    assert result.parse_ok and result.is_correct


def test_bad_format_is_not_exception():
    task = Task("p", "plan", "test", 11, 2, (1,2,3), ("SH1",), target=(1,2,3), max_steps=1)
    result = verify(task, "nonsense")
    assert not result.parse_ok
    assert not result.is_correct
