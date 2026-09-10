# Расчёты Артёма к ICLR

Эта ветка готовит supervised-эксперименты из переписки команды.
Дима ведёт GRPO, выбор разнообразных траекторий и исключение пар.
[Распределение и протокол](plans/README.md). Исходная ветка: `artem_branch`, commit `42c70c8`.
Уже построенные H(K) по локальным результатам: [данные и команда](evidence/local_series/README.md).

## Андрею: установка и проверка

Нужны Linux x86_64, Git, uv и Python 3.11. Команды выполняются из корня репозитория.

```bash
git clone --branch artem_iclr --single-branch https://github.com/VansyDeveloper/The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics.git
cd The-Verifier-Bottleneck-Can-an-AI-Teach-Itself-Mathematics
uv sync --locked
uv run pytest -q
uv run python -m iclr.smoke --out outputs/smoke
```

Smoke работает на CPU без скачивания весов: маленькая случайная Qwen3 проходит
обучение, сохранение, загрузку и оценку. Это проверка кода. Время и память
полноразмерной серии надо измерить на своей GPU.

## Данные и исходная атомарная модель

Перед полной подготовкой проверьте обучение на своей GPU:

```bash
uv run python -m iclr.data --smoke --out outputs/gpu_data
uv run python -m iclr.train --init --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data outputs/gpu_data --out outputs/gpu_probe --num-examples 10 --epochs 1 --device cuda
```

Время и память будут в `outputs/gpu_probe/budget.json`. Затем основная подготовка:

```bash
uv run python -m iclr.data --out outputs/data
uv run python -m iclr.train --init --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data outputs/data --out outputs/atomic_06 --num-examples 10000 --device cuda
```

Это новая атомарная подготовка на PLAN/APPLY. Её качество по каждой операции
лежит в `outputs/atomic_06/eval/summary.json`; проверьте его до серии.
Исторических весов в Git нет. Если они найдутся, используйте проверенный
самодостаточный atomic export вместо новой подготовки. Не подставляйте Base-модель
или адаптер другой размерности. Для фиксации удалённых весов у `--init` есть `--revision`.

## Запуски

```bash
uv run python -m iclr.run --recipe replay trace \
  --model Qwen/Qwen3-0.6B --base outputs/atomic_06/checkpoint \
  --data outputs/data --output outputs/q06 --seeds 0 1 2
```

Команда сохранит конфиги и покажет очередь. Добавьте `--execute` для запуска.
Для первого замера можно выбрать `--recipe size --seeds 0`: только основная пара.
Затем повторите команду с нужными seeds; готовые идентичные расчёты пропускаются.
`replay trace` вместе дают 21 ветвь; `set` добавляет 9 отдельных depth-3 ветвей.
Общая TRACE/r=20% ветвь во всех рецептах используется повторно.

Для Qwen3-1.7B повторите атомарную подготовку с новым `--model` и `--out`,
затем запустите только `--recipe size` с её `--base` и отдельным `--output`.
Проверенная revision 1.7B: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
Размер batch регулируется `--micro-batch`, `--effective-batch`, `--prefix-batch`.
Сохраните одинаковые параметры внутри сравниваемой пары.

## Что передать обратно

Нужен каталог `--output` целиком: конфиги, логи, бюджеты, адаптеры, полные ранги,
атомарные ответы, `DONE`/`FAILED`. Вместе с ним сохраните данные и atomic checkpoint.
Результаты записываются в `runs/*/eval/`; исходная оценка — в `baseline_*`.
Сводная таблица — `results_*.csv`, готовый список пар для H(K) — `comparison_*.json`.
Повтор команды проверяет входы и готовые файлы; другой конфиг требует нового каталога.

[Анализ H(K) и состояние исходных артефактов](evidence/README.md).
Старый интерпретатор, scorer и статистика скопированы без изменений в `legacy/`;
их происхождение и хэши записаны в `legacy/SOURCES.json`.
