# Запуски после feedback от 23 сентября

Это exploratory amendment к неизменённому v4. Endpoint hierarchy сохранена. Отдельные plan/data/code hashes проходят через очереди, обучение, evaluation, selection и final lock. Результаты старой версии проверяются её архивным analyzer.

## A. Проверка устойчивости

Подготовка и запуск R0–R3 приведены в корневом README. Все четыре условия стартуют из одного base с hash `a3ca02aa9e41ea7899c5217bf766a596d369b9ff0f92d51320d5ad3d4327d938`. Нужен именно этот payload на машине оператора. Декларация atomic-роли базы не подтверждает её training history.

На двух GPU планировщик выполняет R0/R1, затем R2/R3; между train/eval могут размещаться готовые задачи. На четырёх GPU — все четыре условия. Initial full dev считается один раз. На каждом checkpoint monitor сохраняет полный 125-candidate score space выбранных depth3 задач, все dev B/D crossed, 20 atomic задач на операцию и четыре целые train-панели. Входы заданы manifest, train и dev помечены отдельно. На полном dev считаются все прежние ordinary/panel/depth4/atomic задачи.

В `training_metrics.jsonl`: full-vocabulary CE, local-action CE, legal-token gate CE, raw/weighted CF и entropy, replay CE, pre-clip norm, clipping factor и фактическая норма изменения параметров. В `budget.json`: composition/replay exposures, PLAN/APPLY loss tokens, replay weight и доля примеров. Replay/no-replay не является сравнением равных вычислительных бюджетов.

В output каждой тренировки:

- `checkpoints/step_000032/{adapter/,DONE,budget.json,exposure.json,resume_state.pt}` — пример состояния шага 32;
- `monitors/step_000032/` — фиксированный train/dev монитор;
- `training_stream.jsonl`, `replay_stream.jsonl` — фактические потоки;
- корневой `DONE` появляется только после завершения всего объявленного run и проверки save/reload.

Для ручного resume использовать исходный config, исходный output и последний целый checkpoint. Восстанавливаются Adam, Python/NumPy/Torch/CUDA RNG, счетчики и курсор. Изменение LR/числа шагов/данных — новый run с другим output. `resume_state.pt` и веса нужны для resume; обычный weight-free export их не содержит.

```bash
uv run python -m iclr.research_train --config outputs/v5_stability/configs/q06_mask1_R2_seed0_train.json --resume-from latest
```

## Выбор одного режима

Числа ниже — пример команды для шага 32 R2, а не рекомендация заранее выбрать именно его. Сначала посмотреть monitor, затем получить **full dev** для кандидата:

```bash
uv run python -m iclr.research_stability evaluate-step --training outputs/v5_stability/training/q06_mask1_R2_seed0 --step 32 --out outputs/v5_R2_step32_full
uv run python -m iclr.research_stability select --queue outputs/v5_stability/queue.json --condition R2 --step 32 --full-evaluation outputs/v5_R2_step32_full --out outputs/v5_stability_selection.json
```

Для конечного шага 128 `--full-evaluation` можно опустить: используется уже готовая full dev evaluation очереди. Сначала дождаться завершения выбранного train run: intermediate checkpoint не превращает его в completed training.

Инженерный допуск записан до следующих запусков в `research_v5.json`:

1. На **каждой** PLAN-операции снижение accuracy не больше 5 п.п.; APPLY — то же для операций с initial accuracy ≥0,8. В обеих оценках одинаковое число задач, не меньше 20 на операцию. SH1/SC2 APPLY с низким initial не выдаются за уже освоенные навыки.
2. Нужен прирост mean correct mass минимум 0,001 на фиксированном TRAIN probe либо на полном ordinary dev A. Это отсеивает отсутствие изменений. B/D не участвуют в выборе режима.
3. Smoke не получает научный admission даже при случайном выполнении численных условий. Gate artifact с `pass=false` сохраняется для разбора, а очередь `matched` его отклоняет.

Если проходят несколько режимов, заранее оговорён порядок: меньше шагов, затем меньшая стоимость replay. Зафиксировать один общий выбор; не подбирать длительность отдельно для будущих arms. Если ни один не проходит, остановить масштабирование CF и разобрать сохранённые diagnostics. Новая LR-точка потребует отдельной явно обозначенной dev-итерации.

## B. CE/CF на выбранном режиме

```bash
uv run python -m iclr.research start --queue-name matched --inputs outputs/shared_inputs_v4 --amendments evidence/research_v5/amendments --models models.local.json --model-names q06 --stability-selection outputs/v5_stability_selection.json --out outputs/v5_matched --gpus 0 1
```

Общий LR, replay, max steps, panel batch, seed, base и manifest наследуются от допуска. CF weight=0,1, tau=1. Для полного факторного сравнения добавить `--with-entropy`; entropy weight=0,01. Наличие этого флага не объявляет комбинацию победителем.

```bash
uv run python -m iclr.research_analysis --plan v5 --runs outputs/v5_matched/evaluations/q06_mask1_ce_seed0 outputs/v5_matched/evaluations/q06_mask1_ce_cf_seed0 --control ce --treatment ce_cf --out outputs/v5_matched_comparison.json
```

Обсудить full dev, retention и ограничения provenance. Затем явно выбрать одну пару, например:

```bash
uv run python -m iclr.research select --queue outputs/v5_matched/queue.json --method ce_cf --reason "Здесь записать фактическое обоснование по dev" --out outputs/v5_method_selection.json
```

Это выбор гипотезы для следующей проверки, а не автоматическое признание преимущества CF. Если CF не улучшает критерии или снова разрушает навыки, такой результат нужно зафиксировать; никакой обязанности продолжать нет.

## C. Одна пара и закрытый final

Только после осмысленного выбора из B:

```bash
uv run python -m iclr.research start --plan v5 --queue-name confirm --selection outputs/v5_method_selection.json --inputs outputs/shared_inputs_v4 --amendments evidence/research_v5/amendments --models models.local.json --model-names q06 --out outputs/v5_confirm --gpus 0 1
uv run python -m iclr.research freeze --queues outputs/v5_confirm/queue.json --out outputs/v5_final_lock.json
uv run python -m iclr.research start --plan v5 --queue-name confirm --phase final --trained-from outputs/v5_confirm --selection outputs/v5_method_selection.json --analysis-lock outputs/v5_final_lock.json --inputs outputs/shared_inputs_v4 --amendments evidence/research_v5/amendments --models models.local.json --model-names q06 --out outputs/v5_final --gpus 0 1
```

Обе masks, paired seeds 0/1/2, одинаковая заранее выбранная длительность. Final оценивает готовые веса без повторного обучения. `freeze` v5 отклоняет очередь stability: промежуточный screening не заменяет confirmation. Nonsignificance не доказывает эквивалентность; для подтверждения null нужен отдельный precision/equivalence plan.

## Независимая CPU-ветвь и происхождение весов

Q1 post-hoc уже выполнен. Его можно полностью пересчитать из компактных raw scores в пакете командой из README. Для воспроизведения исходных receipt/file checks нужны два больших первичных ZIP, их hashes сохранены в `ARCHIVES.json`; в review-пакете лежат достаточные scores для новой арифметики и неизменённый компактный v4 replay.

`evidence/research_v5/initialization_provenance.json` перечисляет фактические шесть Q1 hashes и недостающие training artifacts. `base_training_receipt` в model profile можно добавить только после получения и проверки реального receipt, чьи `payload_hash`, `files` и `exposure.json` описывают именно эту базу. Поле не заполняется вымышленным PASS. Для переупаковки необходима проверка tensors и tokenizer/config оператором.

Если понадобятся все освоенные APPLY-примитивы, нужна отдельная общая atomic initialization на train-only данных и отдельный dev допуск. Текущий replay даёт технический путь обучения PLAN/APPLY, но не является доказательством освоения отсутствующих навыков. Старую и новую initialization нельзя смешивать в одном paired effect.

Q2 не зависит от завершения R0–R3, но требует verified task-training history и собственного numeric gate. Q4/SIGReg — только после положительной state-access диагностики. Четыре эпохи, 8B и увеличение entropy weight сейчас не являются автоматическими следующими шагами.
