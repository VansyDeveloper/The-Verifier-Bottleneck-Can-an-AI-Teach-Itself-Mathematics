# artem_iclr: результаты Q1/Q3 и следующая проверка

Состояние на 23 сентября 2026: Q1/Q3 на Qwen3-0.6B уже выполнены по v4. Подтверждённого преимущества CF нет. Следующий научный запуск — ограниченная проверка устойчивости CE **R0–R3**, затем общий режим CE/CF при пройденном допуске. Код этой последовательности подготовлен; большие новые обучения здесь не выполнялись.

| Уже выполнено | Что установлено |
|---|---|
| Q1: baseline + atomic-control/composition, seeds 0/1/2, original mask, dev | Crossed joint 9,38% → 42,71%; ordinary Hit@32 25,58% → 9,58%. Это разные задачи выбора программы. История шести фактических адаптеров не восстановлена. |
| Q3: initial + CE/CE+H/CE+CF/CE+CF+H, mask1/seed0, 128 steps | CE+CF − CE joint +3,125 п.п., CI [−6,25; 12,50]. Продолжения существенно потеряли atomic PLAN. |
| Повторная проверка v4 | 34 хеша и 35 парных endpoint comparisons из неизменённого feedback-пакета воспроизводятся. Полный предыдущий разбор находится внутри него. |
| Новая CPU-диагностика Q1 | Прибавление средней поправки scores из 64 A-reference задач даёт joint 11,46%, исходный composition — 42,71%. Одной этой поправки недостаточно. |
| Atomic APPLY audit | Метки и parser outcomes воспроизведены без расхождений; сохранены таблица по операциям/полям/степеням и 20 raw ошибок SH1/SC2. |

Проверки кода и точный статус запусков: [RESEARCH_STATUS.md](plans/RESEARCH_STATUS.md). Начало повторного ревью: [REVIEW_REQUEST_RU.md](REVIEW_REQUEST_RU.md). Интерпретация новых чисел: [POSTHOC_RESULTS_RU.md](evidence/research_v5/POSTHOC_RESULTS_RU.md).

## Подготовить машину

Python 3.11, зависимости из `uv.lock`, GPU с CUDA для обучения. После распаковки архива кода:

```bash
uv sync --locked
mkdir -p outputs
tar -xzf evidence/research_v4/Q1_Q3_20260923/shared_inputs_v4.tar.gz -C outputs
cp plans/research_v5_models.example.json models.local.json
```

В `models.local.json` заменить `base` на абсолютный путь к тому же `atomic_06_bf16/checkpoint`. Ожидаемый base hash уже указан. Веса в архив не включены. Данные v4 включены побайтно; повторно генерировать их не нужно. Новые replay/monitor manifests лежат в `evidence/research_v5/amendments/` и ссылаются на исходные data/plan hashes.

Сначала проверить очередь без обучения:

```bash
uv run python -m iclr.research plan --queue-name stability   --inputs outputs/shared_inputs_v4 --amendments evidence/research_v5/amendments   --models models.local.json --model-names q06 --out outputs/v5_stability
```

После проверки этого пакета запустить подготовленную очередь; на одной GPU заменить `0 1` на `0`:

```bash
uv run python -m iclr.research run --queue outputs/v5_stability/queue.json --gpus 0 1
```

Это четыре CE-прогона: R0 `(lr=1e-4, replay=0)`, R1 `(3e-5, 0)`, R2 `(1e-4, 0.25)`, R3 `(3e-5, 0.25)`. У всех mask1, seed0, 64 панели и максимум 128 composition updates; checkpoints 0/8/16/32/64/128. Replay добавляет по одному PLAN и APPLY примеру на step. Вес loss, доля примеров и количество токенов записываются отдельно.

Каждый checkpoint имеет свой receipt и состояние Adam/RNG/cursor. Monitor использует фиксированные train/dev входы; полная dev-оценка выполняется для initial и конечных адаптеров. Если очередь прервана, повторить `research run ... --retry-failed`: обучение продолжится с последнего целого checkpoint. Код и входы работающей очереди менять нельзя.

## Что делать после R0–R3

Выбрать один общий режим по сохранению навыков и обучению композициям. Для промежуточного шага сначала выполнить полную dev-оценку; затем получить admission artifact. Команды и точные допуски: [RUN_V5_RU.md](plans/RUN_V5_RU.md).

После PASS доступна очередь `matched`: CE против CE+CF на одинаковых LR/replay/шагах. Варианты с энтропией включаются явно. Только затем возможны выбор одной пары, mask1/mask2 × seeds0/1/2 и immutable lock перед final. Final не используется для выбора checkpoint.

Сейчас не запускать автоматически старую четырёхметодную confirmation, 256 steps/четыре эпохи, 8B или SIGReg. Q2 требует подтверждённой истории initialization и собственного неизменённого численного допуска. Неизвестная история допускает описательные сравнения, но блокирует `never_seen_pairs`, `length_extrapolation` и утверждения об эффекте исторического рецепта.

## Локальная проверка

```bash
uv run pytest -q
ICLR_SMOKE_DEVICE=cuda uv run pytest tests/test_research_v5.py -q
uv run python analysis/q1_prior_residual.py   --results evidence/research_v5/q1_posthoc --out outputs/q1_posthoc_check
```

Архив `evidence/research_v4/Q1_Q3_20260923/feedback_Q1_Q3_20260923.zip` содержит неизменённые v4 analyzer, `check_primary.py`, endpoint tables и исходный разбор. Распаковать его и запустить `check_primary.py` внутри полученной папки. Старые receipts не переписываются под новый code hash.
