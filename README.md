# ICLR: запуск экспериментов v4 — 22 сентября

**При двух GPU начните Q1 на q06 и Q3 параллельно.** Q2 и Q4 можно запускать
независимо после них. Один процесс занимает одну GPU; каждой очереди нужен
отдельный `--out`, всем машинам — одинаковые данные.

| Очередь | Что запускается сразу | Следующий этап по dev |
|---|---|---|
| **Q1** | Старые checkpoints: START×TARGET, обычные задачи, энтропия; без обучения | Оценка 8B и закрытая final |
| **Q3** | 4 CE/counterfactual/entropy условия, mask1, seed0 | Бюджет выбранной пары, затем 2 маски × 3 seed |
| **Q2** | Exact/MC и реальные Adam-шаги на **original** | Reward pilot после численного допуска |
| **Q4** | Доступ к текущему состоянию + контроль токенов | State/SIGReg pilot после положительного dev-сигнала |

## Подготовка

```bash
git switch artem_iclr
git pull --ff-only
uv sync --locked
nvidia-smi
```

Пути к прежним atomic checkpoints и adapters Qwen3-0.6B/8B — в
[upgrade_models.json](plans/upgrade_models.json). При других путях скопируйте
профиль и передавайте `--models путь.json`. Веса в Git не входят.

Один раз на CPU подготовьте v4 из общего набора v2:

```bash
uv run python -m iclr.research_data \
  --source outputs/shared_inputs_feedback_v2/new_tasks \
  --reference outputs/shared_inputs_feedback_v2/reference_data \
  --out outputs/shared_inputs_v4
tar -czf shared_inputs_v4.tar.gz outputs/shared_inputs_v4
```

На каждой машине из корня репозитория: `tar -xzf shared_inputs_v4.tar.gz`.
Если v2 отсутствует, [команды восстановления](plans/RUN_V4_RU.md#данные).
Данные, планы и locks v2/v3 сохраняются отдельно.

## Запуск на GPU

Первые две команды запустите **в двух терминалах**. На одной GPU замените
`--gpus 1` на `--gpus 0` и выполняйте последовательно. На разных машинах
используйте `--gpus 0` на каждой.

```bash
uv run python -m iclr.research start --queue-name Q1 --model-names q06 --inputs outputs/shared_inputs_v4 --out outputs/v4_Q1_dev --gpus 0
uv run python -m iclr.research start --queue-name Q3 --inputs outputs/shared_inputs_v4 --out outputs/v4_Q3_dev --gpus 1
```

После них, на свободной GPU:

```bash
uv run python -m iclr.research start --queue-name Q2 --inputs outputs/shared_inputs_v4 --out outputs/v4_Q2_diagnostic --gpus 0
uv run python -m iclr.research start --queue-name Q4 --inputs outputs/shared_inputs_v4 --out outputs/v4_Q4_diagnostic --gpus 0
uv run python -m iclr.research start --queue-name Q1 --model-names q8b --inputs outputs/shared_inputs_v4 --out outputs/v4_Q1_8b_dev --gpus 0
```

По умолчанию CUDA/BF16; без доступной GPU запуск остановится. Просмотр заданий:
`start` → `plan`, убрать `--gpus`. При нехватке памяти уменьшите
`--prefix-batch 16` до `1` в новой папке очереди.
Возобновление: `uv run python -m iclr.research run --queue outputs/v4_Q3_dev/queue.json --gpus 0`.
Ошибки — в `status.json` и `logs/`; повтор после исправления — `--retry-failed`.

Каждая очередь сама создаёт **папку и ZIP** `outputs/send_to_artem_exp_*`.
Отправьте ZIP завершённых очередей: внутри данные, сырые оценки, сводки,
конфиги, логи, история обучения, dev-решения, допуски и locks; **весов нет**.
Начните с `START_HERE_RU.md`; полнота и хеши — `EXPORT_MANIFEST.json`.

[Следующие этапы, метрики и final](plans/RUN_V4_RU.md).
[Исправления по второму отзыву и проверенные smoke](evidence/research_v4/README.md).
