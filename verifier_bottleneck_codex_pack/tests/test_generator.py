from vbexp.generator import GenerationSpec, generate_many
from vbexp.polynomial import is_order_sensitive
from vbexp.search import shortest_solutions


def test_generate_plan_tasks_are_exact_depth_and_sensitive():
    spec = GenerationSpec(
        split="test",
        mode="plan",
        primes=(5,7),
        degree_caps=(2,3),
        depths=(2,),
        require_order_sensitive=True,
    )
    tasks = generate_many(spec, count=10, seed=0)
    assert len(tasks) == 10
    for task in tasks:
        witness = tuple(task.metadata["witness_program"])
        assert is_order_sensitive(task.start, witness, task.p)
        shortest = shortest_solutions(task.start, task.target, task.p, task.operations, task.max_steps)
        assert shortest and len(shortest[0]) == 2
