from pathlib import Path

from vbexp.experiment import sha256_file
from vbexp.io import read_tasks, write_tasks


def main():
    source = Path("artifacts/data/pilot/sft_validation_apply.jsonl")
    output = Path("artifacts/data/pilot/sft_validation_apply_non_sh1.jsonl")
    tasks = [task for task in read_tasks(source) if task.program != ("SH1",)]
    if len(tasks) != 80:
        raise ValueError(f"expected 80 non-SH1 tasks, found {len(tasks)}")
    write_tasks(output, tasks)
    print(f"{output} count={len(tasks)} sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
