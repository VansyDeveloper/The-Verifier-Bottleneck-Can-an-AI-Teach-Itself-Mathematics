# ICLR: запуск экспериментов — 22 сентября

**Q1–Q4 можно запускать параллельно на разных GPU или машинах.**
Один процесс занимает одну GPU. При двух GPU: сначала Q1+Q2, затем Q3+Q4.
Каждой очереди — отдельный `--out`; всем машинам — одинаковые данные.

| Очередь | Что запускается сразу | После разбора dev |
|---|---|---|
| **Q1** | Старые Qwen3-0.6B/8B: START×TARGET, обычные задачи, энтропия; без обучения | Закрытая final-оценка |
| **Q2** | Exact/MC gradients, реальные шаги Adam, FP32/BF16 | 4 reward/entropy условия после численной проверки |
| **Q3** | 4 CE/counterfactual/entropy условия, одна маска, seed0 | Выбранная пара × 2 маски × 3 paired seed |
| **Q4** | Текущее состояние после выбранного префикса + контроль токенов | 4 state/SIGReg условия при положительном dev-сигнале |

## Подготовка

```bash
git switch artem_iclr
git pull --ff-only
uv sync --locked
nvidia-smi
```

Существующие atomic checkpoint и adapters 0.6B/8B переиспользуются.
Пути — в [upgrade_models.json](plans/upgrade_models.json). При других путях
сделайте копию профиля и передавайте `--models путь.json`. Веса в Git не входят.

Один раз на CPU подготовьте v3 из общего набора v2:

```bash
uv run python -m iclr.research_data \
  --source outputs/shared_inputs_feedback_v2/new_tasks \
  --reference outputs/shared_inputs_feedback_v2/reference_data \
  --out outputs/shared_inputs_v3
tar -czf shared_inputs_v3.tar.gz outputs/shared_inputs_v3
```

Скопируйте этот TAR на все машины и распакуйте из корня репозитория.
Если v2 отсутствует, [команды восстановления данных](plans/RUN_V3_RU.md#данные).
Старые результаты и analysis lock v2 сохраняются.

## Запуск на GPU

Ниже четыре независимые команды для отдельных машин с GPU `0`.
На одной машине назначьте разные свободные индексы через `--gpus`.

```bash
uv run python -m iclr.research start --queue-name Q1 --model-names q06 q8b --inputs outputs/shared_inputs_v3 --out outputs/Q1_dev --gpus 0
uv run python -m iclr.research start --queue-name Q2 --inputs outputs/shared_inputs_v3 --out outputs/Q2_diagnostic --gpus 0
uv run python -m iclr.research start --queue-name Q3 --inputs outputs/shared_inputs_v3 --out outputs/Q3_dev --gpus 0
uv run python -m iclr.research start --queue-name Q4 --inputs outputs/shared_inputs_v3 --out outputs/Q4_diagnostic --gpus 0
```

По умолчанию BF16. Для просмотра заданий: `start` → `plan`, убрать `--gpus`.
При нехватке памяти уменьшите `--prefix-batch 16` до `1` в новой папке очереди.
Возобновление: `uv run python -m iclr.research run --queue outputs/Q3_dev/queue.json --gpus 0`.
Причины ошибок — в `status.json` и `logs/`; повтор после исправления — `--retry-failed`.

По завершении отправьте `outputs/send_to_artem_exp_*.zip`: данные, оценки,
конфиги и логи включены, веса исключены.

[Следующие этапы, final, второй домен и сравнение результатов](plans/RUN_V3_RU.md).
[Что проверено и границы smoke](evidence/research_v3/README.md).
