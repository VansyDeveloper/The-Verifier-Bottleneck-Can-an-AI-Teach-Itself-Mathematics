# artem_iclr: результаты Q1/Q3 и следующие запуски

Q1/Q3 на Qwen3-0.6B уже выполнены по v4; подтверждённого преимущества CF нет. Следующий запуск — восемь заранее заданных пачек v6, без промежуточного выбора по результатам. Они ещё не запускались.

| Уже выполнено | Что установлено |
|---|---|
| Q1: baseline + atomic-control/composition, seeds 0/1/2, original mask, dev | Crossed joint 9,38% → 42,71%; ordinary Hit@32 25,58% → 9,58%. Это разные задачи выбора программы. История шести фактических адаптеров не восстановлена. |
| Q3: initial + CE/CE+H/CE+CF/CE+CF+H, mask1/seed0, 128 steps | CE+CF − CE joint +3,125 п.п., CI [−6,25; 12,50]. Продолжения существенно потеряли atomic PLAN. |
| Повторная проверка v4 | 34 хеша и 35 парных endpoint comparisons из неизменённого feedback-пакета воспроизводятся. Полный предыдущий разбор находится внутри него. |
| Новая CPU-диагностика Q1 | Прибавление средней поправки scores из 64 A-reference задач даёт joint 11,46%, исходный composition — 42,71%. Одной этой поправки недостаточно. |
| Atomic APPLY audit | Метки и parser outcomes воспроизведены без расхождений; сохранены таблица по операциям/полям/степеням и 20 raw ошибок SH1/SC2. |

Два присланных `send_to_artem_exp_v4_Q1_*` / `Q3_*` уже разобраны. Новые пачки начинают с той же `atomic_06_bf16/checkpoint`; adapters из результатов Q1/Q3 для них не нужны. Старые очереди и отдельный пилот R0–R3 повторять не надо: эти условия входят в пачки.

Проверки кода и пачек: [VALIDATION.json](evidence/research_v6/VALIDATION.json); статус запусков: [RESEARCH_STATUS.md](plans/RESEARCH_STATUS.md). Запрос reviewer: [REVIEW_REQUEST_RU.md](REVIEW_REQUEST_RU.md). Интерпретация чисел: [POSTHOC_RESULTS_RU.md](evidence/research_v5/POSTHOC_RESULTS_RU.md).

## Подготовить машину

Python 3.11, зависимости из `uv.lock`, GPU с CUDA для обучения. После распаковки архива кода:

```bash
uv sync --locked
mkdir -p outputs
tar -xzf evidence/research_v4/Q1_Q3_20260923/shared_inputs_v4.tar.gz -C outputs
cp plans/research_v5_models.example.json models.local.json
```

В `models.local.json` заменить `base` на абсолютный путь к тому же `atomic_06_bf16/checkpoint`. Ожидаемый base hash уже указан. Веса в архив не включены. Данные v4 включены побайтно; повторно генерировать их не нужно. Новые replay/monitor manifests лежат в `evidence/research_v6/amendments/` и ссылаются на исходные data/plan hashes.

## Что запускать

Сначала **1–4**, затем **5–8**. Каждая пачка: две маски × три seed × два метода = 12 обучений; всего 96. У каждого 128 updates, checkpoints 0/8/16/32/64/128, monitor и полная dev-оценка. Всё выполняется автоматически.

| № | Сравнение | LR | Replay |
|---|---|---:|---:|
| 1 | CE / CE+CF | 1e-4 | 0 |
| 2 | CE / CE+CF | 3e-5 | 0 |
| 3 | CE / CE+CF | 1e-4 | 0.25 |
| 4 | CE / CE+CF | 3e-5 | 0.25 |
| 5 | CE+H / CE+CF+H | 1e-4 | 0 |
| 6 | CE+H / CE+CF+H | 3e-5 | 0 |
| 7 | CE+H / CE+CF+H | 1e-4 | 0.25 |
| 8 | CE+H / CE+CF+H | 3e-5 | 0.25 |

H — энтропия, CF — counterfactual loss. На одной A100 все номера идут последовательно:

```bash
BATCHES="1 2 3 4 5 6 7 8"
for B in $BATCHES; do
  uv run python -m iclr.research start --queue-name batch --batch "$B" \
    --inputs outputs/shared_inputs_v4 --amendments evidence/research_v6/amendments \
    --models models.local.json --model-names q06 --out "outputs/batch_$B" \
    --gpus 0 --retry-failed || \
  uv run python -m iclr.research send --out "outputs/batch_$B" --allow-incomplete \
    --destination "outputs/send_to_artem_exp_batch_${B}_partial_$(date -u +%Y%m%d_%H%M%S)"
done
```

Разные A100/серверы могут выполнять разные номера параллельно. На **двух** серверах задать списки `1 3 5 7` / `2 4 6 8`; на **четырёх** — `1 5` / `2 6` / `3 7` / `4 8`. Один номер назначается одному исполнителю.

Если несколько GPU в одном сервере, можно оставить весь список и заменить `--gpus 0` на `--gpus 0 1` или `--gpus 0 1 2 3`: задания внутри пачки пойдут параллельно. Каждая пачка сама считает initial для своих масок; общей подготовки между серверами нет. Команду можно повторить после сбоя: готовые задания пропускаются, обучение продолжается с checkpoint. Код работающей очереди не менять.

## Что прислать

После каждой пачки готов `outputs/send_to_artem_exp_batch_<номер>_*.zip`. Присылать по мере готовности, в итоге восемь полных ZIP. При ошибке сохраняется архив с `partial` в имени; повтор команды продолжит пачку. В ZIP есть raw scores, метрики, данные, конфиги, логи и хеши, без весов. Каждый ZIP распаковывать в свою папку. Все adapters и `resume_state.pt` оставить у оператора.

Повторная упаковка, например пачки 1: `uv run python -m iclr.research send --out outputs/batch_1`.

План пачек: [RUN_V6_RU.md](plans/RUN_V6_RU.md). Это расширенный dev-сбор; выбор лучшего результата делаем после получения всех архивов. Final, Q2, SIGReg, 8B и новые эпохи в команды выше не входят. Прежний поэтапный план сохранён в [RUN_V5_RU.md](plans/RUN_V5_RU.md).

## Локальная проверка

```bash
uv run pytest -q
ICLR_SMOKE_DEVICE=cuda uv run pytest tests/test_research_v5.py tests/test_research_resume_recovery.py -q
uv run python analysis/q1_prior_residual.py   --results evidence/research_v5/q1_posthoc --out outputs/q1_posthoc_check
```

Архив `evidence/research_v4/Q1_Q3_20260923/feedback_Q1_Q3_20260923.zip` содержит неизменённые v4 analyzer, `check_primary.py`, endpoint tables и исходный разбор. Распаковать его и запустить `check_primary.py` внутри полученной папки. Старые receipts не переписываются под новый code hash.
