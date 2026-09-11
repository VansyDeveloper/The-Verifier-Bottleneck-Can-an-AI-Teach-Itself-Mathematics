# Расчёты Артёма к ICLR

Нужны Linux x86_64, Git, uv и Python 3.11. [Кто что делает](plans/README.md), [что готово и что осталось](plans/STATUS.md).

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

Для сравнения размеров выбраны Qwen3-0.6B и [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (8,2 млрд параметров). Обе пары запускаем в BF16. Для 0.6B нужна отдельная подготовка с той же точностью:

```bash
uv run python -m iclr.train --init --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data outputs/data --out outputs/atomic_06_bf16 --num-examples 10000 \
  --device cuda --dtype bfloat16 --prefix-batch 1
uv run python -m iclr.run --recipe size --model Qwen/Qwen3-0.6B \
  --base outputs/atomic_06_bf16/checkpoint --data outputs/data --output outputs/q06_bf16 \
  --seeds 0 --device cuda --dtype bfloat16 --prefix-batch 1
```

Для 8B конфигурация и токенизатор проверены; полный запуск ещё не выполнен:

```bash
uv run python -m iclr.train --init --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --data outputs/data --out outputs/atomic_8b --num-examples 10000 \
  --device cuda --dtype bfloat16 --prefix-batch 1
uv run python -m iclr.run --recipe size --model Qwen/Qwen3-8B \
  --base outputs/atomic_8b/checkpoint --data outputs/data --output outputs/q8b \
  --seeds 0 --device cuda --dtype bfloat16 --prefix-batch 1
```

Перед очередью проверьте PLAN/APPLY в `eval/summary.json` каждой атомарной подготовки. Здесь тоже нужен `--execute` для запуска очереди. После замера с seed 0 повторите команды очереди с `--seeds 0 1 2` в тех же каталогах: готовые расчёты сохранятся. Для нового анализа выберите другой `--output`.

В сравниваемых вариантах сохраняйте одинаковые `--dtype`, `--micro-batch`, `--effective-batch`, `--prefix-batch`. BF16 требует подходящей CUDA-карты. Время и память полной серии на A100 пока не измерены.

Верните весь `outputs/`: данные, веса, логи и результаты. Сводка лежит в `results_*.csv`; очередь также печатает команды парного анализа каждого сравнения. [Что сохраняется и как проверить результаты](evidence/README.md), [готовые локальные кривые](evidence/local_series/README.md), [проверка по исходным планам](evidence/plan_audit_20260911.json).
