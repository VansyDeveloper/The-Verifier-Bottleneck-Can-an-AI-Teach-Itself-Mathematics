# Зависимости

Core sandbox устанавливается через:

```bash
python -m pip install -e ".[dev,analysis]"
```

ML-зависимости намеренно не закреплены жёстко до выполнения preflight, потому что API Transformers/PEFT/TRL и установленная CUDA должны быть согласованы с реальной средой. Codex обязан после успешного дымового запуска создать один из файлов:

- `requirements/lock.txt` через `pip freeze`;
- или `requirements/conda-lock.yml`;
- или эквивалентный lock-файл используемого менеджера.

Минимальный набор: torch, transformers, datasets, accelerate, peft, trl.
