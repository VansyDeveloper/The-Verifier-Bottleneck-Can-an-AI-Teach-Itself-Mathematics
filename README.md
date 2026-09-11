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

Две последние команды проверяют весь путь на маленькой случайной модели, сначала на CPU, затем на GPU. Веса не скачиваются. По умолчанию используется FP32; для BF16 добавьте `--dtype bfloat16` к GPU-проверке и выберите новый `--out`. Для полной серии нужна карта больше 4 ГБ. Затем подготовьте данные и атомарную модель:

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

В исходной серии используется Qwen3-0.6B. Для Qwen3-1.7B повторите подготовку с revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` и отдельными каталогами.

Для крупной модели той же семьи есть [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). Конфигурация и токенизатор проверены; полный запуск 8B ещё не выполнен:

```bash
uv run python -m iclr.train --init --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --data outputs/data --out outputs/atomic_8b --num-examples 10000 \
  --device cuda --dtype bfloat16 --prefix-batch 1
uv run python -m iclr.run --recipe size --model Qwen/Qwen3-8B \
  --base outputs/atomic_8b/checkpoint --data outputs/data --output outputs/q8b \
  --seeds 0 --device cuda --dtype bfloat16 --prefix-batch 1
```

Здесь тоже нужен `--execute` для запуска очереди. В сравниваемых вариантах сохраняйте одинаковые `--dtype`, `--micro-batch`, `--effective-batch`, `--prefix-batch`; для сравнения размеров повторите пару 0.6B с теми же настройками. BF16 требует подходящей CUDA-карты. Время и память полной серии на A100 пока не измерены.

Для предложенной 9B нужно точное название модели. [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) имеет другую архитектуру и с закреплёнными зависимостями не загружается; её поддержка пока не подготовлена.

Верните весь `outputs/`: данные, веса, логи и результаты. Сводка лежит в `results_*.csv`; очередь также печатает команды парного анализа каждого сравнения. [Команды для H(K)](evidence/README.md), [готовые локальные кривые](evidence/local_series/README.md), [последняя проверка ветки](evidence/audit_followup_20260911.json).
