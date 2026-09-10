# Расчёты Артёма к ICLR

Нужны Linux x86_64, Git, uv и Python 3.11. [Кто что делает](plans/README.md).

```bash
git clone --branch artem_iclr --single-branch https://github.com/VansyDeveloper/The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics.git
cd The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics
uv sync --locked
uv run python -m pytest -q
uv run python -m iclr.smoke --out outputs/smoke
uv run python -m iclr.smoke --device cuda --out outputs/smoke_gpu
```

Две последние команды проверяют весь путь на маленькой случайной модели, сначала на CPU, затем на GPU. Веса не скачиваются. Штатная серия работает в FP32; карты на 4 ГБ для неё недостаточно. После проверки подготовьте данные и атомарную модель:

```bash
uv run python -m iclr.data --out outputs/data
uv run python -m iclr.train --init --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data outputs/data --out outputs/atomic_06 --num-examples 10000 --device cuda
```

Перед полной очередью проверьте PLAN/APPLY каждой операции в `outputs/atomic_06/eval/summary.json`. Это новая атомарная подготовка; старых весов в Git нет.

```bash
uv run python -m iclr.run --recipe replay trace \
  --model Qwen/Qwen3-0.6B --base outputs/atomic_06/checkpoint \
  --data outputs/data --output outputs/q06 --seeds 0 1 2 --device cuda
```

Команда показывает очередь; `--execute` запускает её. `replay trace` дают 21 расчёт, отдельный `--recipe set` добавляет 9. Для первого замера выберите `--recipe size --seeds 0`. Готовые одинаковые расчёты пропускаются; менять параметры следует в новом `--output`.

Для Qwen3-1.7B повторите подготовку с её моделью и revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, отдельными `--out` и `--output`; затем запустите `--recipe size` с её checkpoint. Параметры `--micro-batch`, `--effective-batch`, `--prefix-batch` должны совпадать внутри пары.

Верните весь `outputs/`: данные, веса, логи и результаты. Сводка лежит в `results_*.csv`; очередь также печатает команды парного анализа каждого сравнения. [Команды для H(K)](evidence/README.md), [готовые локальные кривые](evidence/local_series/README.md), [проверка ветки](evidence/audit_20260911.json).
