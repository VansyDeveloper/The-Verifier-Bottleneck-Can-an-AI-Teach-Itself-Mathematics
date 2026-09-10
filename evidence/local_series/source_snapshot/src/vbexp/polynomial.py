from __future__ import annotations

from math import comb
from typing import Callable, Iterable, Sequence

State = tuple[int, ...]
Operation = Callable[[State, int], State]


def normalize(values: Iterable[int], p: int) -> State:
    if p <= 1:
        raise ValueError("p must be greater than 1")
    return tuple(int(v) % p for v in values)


def validate_state(state: Sequence[int], p: int) -> State:
    if len(state) < 2:
        raise ValueError("state must contain at least c0 and c1")
    return normalize(state, p)


def sh1(state: State, p: int) -> State:
    """P(x) -> P(x+1) over F_p."""
    state = validate_state(state, p)
    degree = len(state) - 1
    out = [0] * (degree + 1)
    for i, ci in enumerate(state):
        for j in range(i + 1):
            out[j] = (out[j] + ci * comb(i, j)) % p
    return tuple(out)


def sc2(state: State, p: int) -> State:
    """P(x) -> P(2x) over F_p."""
    state = validate_state(state, p)
    return tuple((ci * pow(2, i, p)) % p for i, ci in enumerate(state))


def rev(state: State, p: int) -> State:
    """Reverse coefficients with a fixed degree cap."""
    state = validate_state(state, p)
    return tuple(reversed(state))


def ac1(state: State, p: int) -> State:
    """Add one to the constant coefficient."""
    values = list(validate_state(state, p))
    values[0] = (values[0] + 1) % p
    return tuple(values)


def ax1(state: State, p: int) -> State:
    """Add one to the x coefficient."""
    values = list(validate_state(state, p))
    values[1] = (values[1] + 1) % p
    return tuple(values)


OPERATIONS: dict[str, Operation] = {
    "SH1": sh1,
    "SC2": sc2,
    "REV": rev,
    "AC1": ac1,
    "AX1": ax1,
}


def apply_operation(state: State, operation: str, p: int) -> State:
    try:
        function = OPERATIONS[operation]
    except KeyError as exc:
        raise ValueError(f"unknown operation: {operation}") from exc
    return function(state, p)


def apply_program(state: Sequence[int], program: Sequence[str], p: int) -> State:
    current = validate_state(state, p)
    for operation in program:
        current = apply_operation(current, operation, p)
    return current


def swapped_program(program: Sequence[str], index: int) -> tuple[str, ...]:
    if index < 0 or index + 1 >= len(program):
        raise IndexError("index must point to the first item of an adjacent pair")
    result = list(program)
    result[index], result[index + 1] = result[index + 1], result[index]
    return tuple(result)


def is_order_sensitive(state: Sequence[int], program: Sequence[str], p: int) -> bool:
    baseline = apply_program(state, program, p)
    for index in range(len(program) - 1):
        if apply_program(state, swapped_program(program, index), p) != baseline:
            return True
    return False
