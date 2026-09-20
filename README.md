# Эксперименты Артёма к ICLR

**Прежние запуски из ветки `artem_iclr` уже выполнены. Повторно запускать
старую очередь и заново обучать прежние atomic checkpoint не нужно.**
Эта версия добавляет новые проверки и обучения из
[goal_to_iclr.md](plans/source/goal_to_iclr.md).

## Что уже получено

| Прежняя серия | Зачем она нужна | Подтверждённые результаты |
|---|---|---|
| Qwen3-0.6B | Основная парная проверка atomic control / composition | Две ветки × seed 0/1/2; raw, midpoint/final, конфиги и квитанции |
| Qwen3-8B | Проверка того же контраста при большем размере Qwen | Две ветки × seed 0/1/2; согласованные raw/summary/CSV внутри ZIP |

По сохранённому dev воспроизведена A-only калибровка. Hit@32 на B у SFT:
0.6B — 7,83% → 56,00%; 8B — 8,33% → 58,33%. Atomic control после такой же
поправки: 44,00% и 33,83%. Это пересчёт ранжирования, без обучения и inference.
Внешний отдельный CSV 8B отличается от raw в 18 значениях; отчёт сохраняет
расхождения. [Проверенные файлы и границы свидетельств](evidence/upgrade/README.md).

Новая оценка прежних checkpoint нужна для **новых закрытых задач и реальных
вероятностей на префиксах** — этого нет в прежних результатах. Новые полные
GPU-запуски пока не выполнены. Остальные старые команды сохранены только
в [архивной инструкции](plans/LEGACY_COMMANDS.md); новая очередь их не запускает.

## Подготовить компьютер

Нужны Linux, Git, uv, Python 3.11 и CUDA GPU. Ожидаются A100; фактические карты
проверяет оператор. После публикации этой версии ветки:

```bash
git switch artem_iclr
git pull --ff-only
uv sync --locked
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
uv run python -m pytest -q
```

Пути задаются в [plans/upgrade_models.json](plans/upgrade_models.json),
относительно корня checkout. При необходимости используйте копию профиля
с другими путями через `--models path/to/models.json`.

| Модель | Уже обученный atomic checkpoint | Прежние continuation adapter |
|---|---|---|
| 0.6B | `outputs/atomic_06_bf16/checkpoint` | Шесть `*/adapter` в `outputs/q06_bf16_mb16_pb32` |
| 8B | `outputs/atomic_8b_bf16/checkpoint` | Шесть `*/adapter` в `outputs/q8b_bf16_mb16_pb32` |

Это исторические пути на `/workspace/verifier-bottleneck`, а не подтверждение
текущего размещения. ZIP `no_models` весов не содержит. Код проверяет хеши
только тех weights, которые нужны выбранному блоку. Для блоков 02–04 нужен
только atomic 0.6B; прежние adapters и 8B для них не нужны.

## Один общий набор данных для всех компьютеров

Один раз на машине, где сохранился прежний `outputs/data`:

```bash
mkdir -p outputs/shared_inputs_20260921
cp -a outputs/data outputs/shared_inputs_20260921/reference_data
uv run python -m iclr.upgrade_data \
  --reference outputs/shared_inputs_20260921/reference_data \
  --out outputs/shared_inputs_20260921/new_tasks
tar -czf outputs/shared_inputs_20260921.tar.gz outputs/shared_inputs_20260921
```

Скопируйте этот TAR на остальные компьютеры и распакуйте из корня checkout:

```bash
tar -xzf shared_inputs_20260921.tar.gz
```

Все компьютеры должны использовать **одни и те же файлы**. Один seed
генератора не гарантирует побайтное совпадение MILP на разных машинах.
Каждый запуск копирует общий набор в свою рабочую папку, проверяет хеши
и выполняет полный аудит до inference. В общий TAR веса не входят.

Если прежнего `outputs/data` нет, его можно восстановить командой
`uv run python -m iclr.data --out outputs/data`. Ожидаемый SHA-256 manifest:
`8a1a9cbc63fa4d9fc5bbf4b82db8d87190fff44f3a88ba50aa550111e161a866`.
Это подготовка исходных задач, не повтор прежнего обучения.

## Выбрать и запустить нужный блок

| ID для `--experiments` | Что будет сделано | Новых обучений | Необходимые веса |
|---|---|---:|---|
| `01_recheck_existing_models` | Новая closed A-only оценка старых моделей, TARGET-контроль, full/local/format, глубина 4, исполнение | 0 | Atomic + шесть adapters каждого размера |
| `02_new_pair_masks` | Четыре допустимые маски, две ветки, seed 0/1 | 16 | Atomic 0.6B |
| `03_new_triples_and_positions` | Новая тройка и новая позиция пары, две ветки, seed 0/1 | 8 | Atomic 0.6B |
| `04_correct_program_choice` | Fixed/uniform/balanced witness, normalized-single, MML; seed 0/1/2 | 15 | Atomic 0.6B |
| `05_reward_gradient` | Exact/sampled reward: четыре SGD pilot, четыре AdamW runs, новая оценка исходного SFT | 8 | Atomic 0.6B + прежний composition adapter seed 0 |
| `06_other_model_family` | SmolLM2-1.7B atomic init, шесть continuation; шесть сопоставимых Qwen0.6 continuation | 13 | Atomic 0.6B; Smol скачивается по закреплённой revision |

Основные блоки 01–04 дают 94 задания / 39 обучений. Все шесть вместе:
137 заданий / 60 обучений. Остальные задания — подготовка, аудит и оценки.
Сначала разумно выполнить 01–04. 05 — условный reward-контроль; 06 —
долгосрочная внешняя репликация из исходного плана.

Пример: запустить блок 02 на двух свободных GPU:

```bash
uv run python -m iclr.upgrade start \
  --experiments 02_new_pair_masks \
  --inputs outputs/shared_inputs_20260921 \
  --out outputs/02_new_pair_masks --gpus 0 1
```

Для другого блока замените ID и имя `--out` по таблице. Можно перечислить
несколько ID после `--experiments`. Один процесс занимает одну GPU;
подставляйте реально свободные индексы. Чтобы только проверить состав,
замените `start` на `plan` и уберите `--gpus`. Очередь и конфиги сохранятся.

Все основные блоки на одном компьютере:

```bash
uv run python -m iclr.upgrade start \
  --inputs outputs/shared_inputs_20260921 \
  --out outputs/01_to_04_main_experiments --gpus 0 1 2 3
```

Для добавления 05/06 используйте `--include-grpo --include-external`.
Новый состав, код или batch требуют нового `--out`. По умолчанию BF16,
`prefix-batch=32`; для меньшей памяти задайте `--prefix-batch 8` при старте.
Рекомендуется запускать диспетчер в tmux.

## Разнести работу между компьютерами

Самый простой вариант: разные блоки на разных компьютерах с тем же общим
набором данных и нужными весами. Например, компьютер A — блок 01; B — 02;
C — 03 и 04. Внутри машины `--gpus` включает параллельность по картам.

Блоки 01–04 также можно разделить по seed, сохраняя обе сравниваемые ветки
на каждой машине. Для масок:

```bash
# Компьютер A: все четыре маски, обе ветки, seed 0.
uv run python -m iclr.upgrade start \
  --experiments 02_new_pair_masks --seeds 0 \
  --inputs outputs/shared_inputs_20260921 \
  --out outputs/02_new_pair_masks_seed0 --gpus 0 1

# Компьютер B: те же четыре маски, обе ветки, seed 1.
uv run python -m iclr.upgrade start \
  --experiments 02_new_pair_masks --seeds 1 \
  --inputs outputs/shared_inputs_20260921 \
  --out outputs/02_new_pair_masks_seed1 --gpus 0 1
```

Не назначайте одну и ту же пару `(блок, seed)` двум компьютерам.
Блоки 05 и 06 распределяются **целиком**: у 05 общий допуск после всех pilot,
у 06 общий новый atomic checkpoint. Внутри них несколько GPU работают
параллельно после зависимостей. Код отклоняет неполный набор seed для 05/06.
Разные компьютеры не должны писать в одну папку очереди.

`status.json` показывает состояние каждого задания, GPU/PID и причину сбоя;
`logs/<job>.log` содержит вывод. Для продолжения существующей очереди:

```bash
uv run python -m iclr.upgrade run \
  --queue outputs/02_new_pair_masks/queue.json --gpus 0 1
```

Готовые результаты проверяются по хешам; живые процессы не дублируются.
После устранения причины сбоя добавьте `--retry-failed`. Без TRAINED
незавершённое обучение повторяется с исходных весов; после TRAINED
восстанавливается оценка. Частичная генерация данных не перезаписывается:
сохраните её и выберите новый `--out`.

## Что отправить Артёму

После успешного `start` или `run` автоматически появляются:

```text
outputs/send_to_artem_exp_02_new_pair_masks_ДАТА_ВРЕМЯ/
outputs/send_to_artem_exp_02_new_pair_masks_ДАТА_ВРЕМЯ.zip
```

**Отправьте ZIP целиком. В нём нет весов моделей.** Папка содержит ту же
выборку файлов в распакованном виде. В ней есть `START_HERE_RU.md`, общая
сводка, результаты каждой задачи, сырые score/префиксы/ответы, калибровка,
все данные, бюджеты, configs, логи, квитанции и снимок кода. Каталоги
`results/experiments/01_…`–`06_…` называются так же, как блоки в таблице.

Чтобы собрать посылку повторно или отдельно от запуска:

```bash
uv run python -m iclr.upgrade send --out outputs/02_new_pair_masks
```

Для явно неполных результатов после остановки очереди добавьте
`--allow-incomplete`; посылка будет помечена `partial`. Во время живых
запусков экспорт запрещён. ZIP проверяется по SHA-256 каждого файла;
проверить после скачивания можно без CUDA и весов:

```bash
python -m iclr.upgrade verify-bundle --archive send_to_artem_exp_....zip
```

[Полный протокол, все пункты G0–G12 и состав результатов](plans/GOAL_TO_ICLR.md).
[Локальные проверки на tiny и настоящей Qwen0.6](evidence/upgrade/README.md).

Локальная рабочая копия ветки: `/tmp/artem_iclr`. Открыть в VS Code:
`code -n /tmp/artem_iclr`. Она уже закреплена за этой worktree; Git не позволяет
checkout той же ветки во второй рабочей копии одновременно.
