# Review 056fd9a — допуск R0–R3 и восстановление после прерывания

## Итог

**Дизайн и содержательная цель R0–R3: OK. Готовность снимка 056fd9a к длинной очереди с заявленным восстановлением: NOT OK до локальных исправлений resume.** Обычный непрерывный запуск не признан неисправным. Замечания относятся к конкретным веткам восстановления, а не к целесообразности экспериментов. После исправления и регрессии — R0–R3; нового расширения экспериментальной сетки не требуется.

Предлагаемый `resume_recovery.patch` не меняет objective, данные, LR, replay weight, метрики или правила выбора. Рабочая ветка пользователя не изменялась. Patch содержит изменения `iclr/research_train.py` и пять дополнительных регрессионных тестов. Проверены применимость к исходному snapshot, синтаксис и изолированные сценарии обработки конфигурации/журналов. Полный Qwen/LoRA forward-backward-resume с этим patch здесь не запускался.

## Что проверено независимо

| Проверка | Результат | Граница |
|---|---|---|
| REVIEW_ARCHIVE_MANIFEST.json | 352 файла: размеры и SHA256 совпали | История Git/push независимо не устанавливалась |
| code_hash | `73a57125b662d7eb70a252d0e674e15203ac04cc27c748f5d1a1e1916b991f82` совпал | Это исходный snapshot, не proposed patch |
| Доступные CPU tests | 28 passed | Не полные 57 tests + 14 subtests из supplied VALIDATION |
| Замороженный v4 feedback | ZIP совпал с ранее приложенным; 34 hashes / 35 comparisons PASS | Без inference |
| Q1 post-hoc | Новый summary.json побайтно совпал с приложенным | Повтор арифметики по компактным scores |
| Независимый verifier audit Q1 | Все correct sets для 992 уникальных задач совпали; 5952 строки scores конечны и имеют длину 125 | Сравнивались множества программ, а не порядок списка correct_programs |
| Atomic errors | Перепроверены labels и parser для всех 20 приложенных raw примеров | Полные raw completions пяти Q3 evaluations в этом пакете не представлены |
| Shared v4 inputs + v5 amendments | Маски mask1/mask2 прошли verify_inputs/verify_amendment; replay labels дополнительно проверены exhaustive depth1 | Относительно приложенного parent, не неизвестного исторического training stream |
| Scientific queue planning | 10 jobs, из них 4 training, без smoke; R0–R3 с 128 updates и заданными checkpoints | Base path в example profile — placeholder, payload модели не проверялся |
| Resume defects | Два основных сценария воспроизведены без загрузки модели | Не crash/inference test реальной Qwen |

Среда reviewer: Python 3.13, torch 2.10 CPU, без transformers/peft и CUDA. Это не locked среда проекта. Попытка установить недостающие библиотеки не удалась из-за недоступности DNS. Архивные CPU/GPU logs просмотрены, но не выдаются за повторённые reviewer GPU-tests.

## Обязательные локальные исправления перед фиксацией новой очереди

### 1. Явный --resume-from конфликтует с тем же полем в config

**Файл:** `iclr/research_train.py:104–108`, CLI `:404–406`; генерация config — `iclr/research.py:260–266`.

В сгенерированном training config уже есть `resume_from="latest"`. Код выполняет:

```python
resume_from = resume_from or config.pop('resume_from', None)
```

Если CLI передал непустой resume_from, short-circuit не выполняет pop. В config остаётся запрещённое поле; config_checked выдаёт:

```text
ValueError: Unknown v4 training settings: {'resume_from'}
```

Именно ручная команда из RUN_V5_RU.md попадает в этот сценарий. Конфигурация без дополнительного CLI аргумента проходит этот участок.

Минимальная поправка:

```python
configured_resume = config.pop('resume_from', None)
resume_from = resume_from if resume_from is not None else configured_resume
```

Регрессия должна передавать resume одновременно в config и отдельным аргументом. Существующий test_research_v5 тестирует отдельный аргумент, но не его конфликт с полем сгенерированного config.

### 2. Неполная последняя строка JSONL ломает восстановление исправного checkpoint

**Файл:** `iclr/research_train.py:174–179`; `iclr/common.py:12–15`.

Текущий код сначала парсит ВЕСЬ живой журнал, затем отбрасывает записи после сохранённого шага. Пример:

```text
{"step": 1, "loss": 0.4}
{"step": 2, "loss":
```

При наличии корректного checkpoint шага 1 recovery всё равно получает JSONDecodeError на оборванном шаге 2. Это не доказательство повреждения весов, Adam или сохранённого RNG; это дефект восстановления журнала.

Предложение: вместе с checkpoint сохранять точные byte offsets и SHA256 зафиксированных префиксов трёх журналов. При resume сначала проверить эти префиксы, затем обрезать только незакоммиченный хвост. Повреждённый committed prefix обязан приводить к ошибке; «пропустить все плохие строки» недопустимо.

В patch добавлены snapshot_logs/restore_logs, offsets/hashes сохраняются в защищённом resume_state.pt. Проверены незавершённый хвост, повреждение committed prefix с отказом без изменения файлов, пустые журналы step0. Это восстановление после process interruption при доступном целостном checkpoint, не гарантия физической durability при любом сбое накопителя.

### Дополнительный узкий случай того же исправления

**Файл:** `iclr/research_train.py:119–130`.

`glob('step_*/DONE')` включает `step_000016.partial/DONE`. Файл DONE создаётся в temporary directory до rename, так что такой кандидат возможен при прерывании внутри checkpoint publication. Patch отбирает только каталоги с числовым suffix после `step_`, игнорируя `.partial`.

Существующее тестирование корректного stop_after + resume полезно, но не покрывает эти сценарии. Расширять архитектуру checkpointing не требуется.

## Ответы на вопросы REVIEW_REQUEST_RU.md

### Checkpoint/resume и step bindings

Для нормального сохранённого checkpoint логика выглядит правильно: сохранены Adam, RNG Python/NumPy/Torch/CUDA, next_step, counters; порядок composition-панелей детерминирован по seed/epoch, replay выбирается по step. Step-specific receipt проверяется evaluator, intermediate checkpoint не объявляется завершённым training run. evaluation_context восстанавливает RNG и train/eval mode модулей.

Архивные tiny tests проверяют именно эти свойства. Они достаточны как plumbing regression на испытанных устройствах и точностях, но не означают, что код прошёл все способы аварийного прерывания. До длинной очереди исправить три случая выше и повторить существующую реальную модельную регрессию.

### Train-only replay и matched arms

OK для приложенных inputs. Оба replay-набора содержат 240 задач, по 48 на операцию, все пять операций и шесть известных полей. Exhaustive depth1 проверка подтвердила unique witness у всех replay-задач. Exclusion audit проходит против task identities и состояний correct trajectories parent dataset, включая final-input registry; final scores не используются для выбора replay.

Первые 128 шагов используют первые 128 replay-строк: это не все 240 задач. Для mask1 получаются AX1=26, SC2=26, REV=25, SH1=26, AC1=25; PLAN и APPLY предъявляются для каждой выбранной строки. Баланс полного пула не следует называть точным балансом каждого короткого префикса.

Loss соответствует объявленному `weight * 0.5 * (mean-token PLAN CE + mean-token APPLY CE)`, оба компонента включают EOS. Composition CE остаётся fixed-length operation-token objective без EOS. Эти разные контракты явно записаны. Budget считает примеры и токены отдельно. Replay/no-replay не является equal-compute comparison; для последующего CE/CF reuse режима LR/replay/steps/order/base сохраняется.

### Bounded R0–R3, retention и progress

OK как engineering screen. Четыре фиксированных условия, mask1/seed0, 64 панели, не больше 128 updates. Retention проверяется на каждой PLAN операции и на APPLY-операциях с initial accuracy >=0.8. Gate использует full dev, а не только 20 monitor-примеров на операцию. Progress >=0.001 — абсолютное увеличение массы правильных программ, то есть 0.1 процентного пункта вероятностной массы; это не прирост Hit@1 на 0.1%.

Train OR dev A progress подходит для отсеивания режима, в котором модель вообще не учится. При проходе только по TRAIN это свидетельство fitting, не переноса. Не менять критерии после результатов и не использовать B/D для выбора stability recipe.

Monitor имеет 100 atomic задач (обе формы PLAN/APPLY), 64 ordinary (16/семью), 128 crossed-клеток B/D (32 панели) и 16 train-задач (4 панели). Это достаточно для выбора кандидатов на full dev, но не для самостоятельного решения о retention <=5 п.п.: одна ошибка на 20 задачах уже 5 п.п. Если все промежуточные monitor-оценки выглядят плохо, это ещё не формальный full-dev fail каждого checkpoint. Дополнительный full dev назначать ограниченно по monitor/train/A, не подгоняя под B/D.

Допуск CE не переносится автоматически на CF: у нового CE/CF comparison снова нужны atomic results. По самому факту admission вывод о качестве метода в статье делать нельзя.

### Q1 additive/nuisance/pool diagnostics

OK как post-hoc анализ именно выбранной аддитивной модели scores. Повтор дал идентичный summary; 64 reference задачи относятся к A и известным полям. Alpha=1, B/D не используются для оценки delta_b. Вероятности нормированы заново по каждой строке, а локальная и full/global-leaf policies анализируются отдельно.

Local B/D:

| Scores | Strict joint, % | Hit@32, % | Correct mass |
|---|---:|---:|---:|
| control | 9.375 | 25.583 | 0.0009477 |
| composition | 42.708 | 9.583 | 0.0027539 |
| control + delta_b | 11.458 | 19.250 | 0.0024926 |
| composition - delta_b | 9.375 | 29.083 | 0.0000501 |

Одной перенесённой средней поправки недостаточно для воспроизведения joint 42.71%. Её удаление существенно меняет joint. Это асимметричный score-level результат, а не оценка доли причинного эффекта prior. Постоянная program-поправка не меняет I и four-target cycles; изменения исходного I нельзя свести к одной такой поправке.

Per-task symmetric pool decomposition корректно выполняется ДО усреднения; between/within percentages — алгебраические слагаемые, не механизм обучения. Exact IID success и ranked Hit@K — разные inference contracts, их расхождение не противоречие. Hit@1=0 во всех пяти вариантах — причина не объявлять панельный выигрыш уже достигнутым улучшением обычного top-1 решения.

### Происхождение весов и scope claims

В текущем описании ограничения достаточны. R0–R3 можно исследовать при одной общей базе с проверенным payload hash, не дожидаясь восстановления всех шести исторических адаптеров. Но неизвестная история не даёт права на `never_seen_pairs`, новый length transfer или причинный вывод об историческом training recipe.

До Q2 нужны реальные материалы именно используемой цепочки initialization: base/model revision и tokenizer/config; payload digest; original training config, manifest, budget и stream либо достаточный воспроизводимый журнал exposure; digest полученного composition adapter; связь этого adapter с тем же base и используемой mask. Для Q1 historical claims аналогичные сведения нужны для всех шести адаптеров. Совпадение названий каталогов недостаточно. При переупаковке сохраняются оба digest и операторская проверка tensors/config/tokenizer. Пропущенная pretraining history остаётся за пределами claims даже после восстановления task-training receipts.

### Новая общая atomic initialization

Не нужна как предварительное условие R0–R3. Этот этап отвечает, можно ли сохранить сильный PLAN и сильные APPLY-операции при composition continuation. Raw audit не выявил ошибку меток/parsing в 20 представленных примерах, но и не установил причину SH1/SC2 провала; оригинальная base-training история отсутствует.

Для последующего утверждения о композиции полностью освоенных примитивов нужна отдельная общая atomic initialization и полный допуск на атомарные навыки. Нельзя считать replay доказательством, что все примитивы освоены, и нельзя смешивать old/new initialization в одном paired effect.

## Что запускать

1. Применить/адаптировать recovery patch до новой очереди. Повторить полный pytest в locked environment и tiny GPU tests. Не использовать мои no-model helper checks вместо model regression.
2. Зафиксировать новый commit/code hash, заново сформировать очередь. Shared v4 inputs, v5 replay manifests и научные пороги не менять. Старые receipts оставить неизменёнными.
3. R0/R1/R2/R3 на 0.6B, mask1/seed0. На 2 GPU: R0/R1, затем R2/R3; на 4 GPU одновременно. Initial full dev один раз; monitor/checkpoints 0/8/16/32/64/128. Максимум 128 updates, не автоматическая дополнительная эпоха.
4. Если конкретный recipe/checkpoint проходит full-dev retention и progress — общий matched CE/CF, с теми же LR/replay/order/steps/base. Если ни один проверенный candidate не проходит — остановить масштабирование CF и разбирать dynamics, не ослаблять gate задним числом.
5. Только после dev выбора — одна пара, 2 masks x 3 paired continuation seeds; immutable lock перед final. Q2 параллельно возможна только после verified task-training history и собственного numerical gate.

Сейчас не запускать автоматически 8B, четыре эпохи/256 steps, confirmation четырёх методов, SIGReg или увеличение entropy weight. Q1 дополнительного обучения для текущего арифметического вывода не требует.

## Команды для patch

```bash
# Из корня рабочей ветки; путь к файлу patch заменить на его фактическое место.
git apply --check /path/to/resume_recovery.patch
git apply /path/to/resume_recovery.patch
uv run pytest -q
ICLR_SMOKE_DEVICE=cuda uv run pytest tests/test_research_v5.py tests/test_research_resume_recovery.py -q
```

Patch нельзя применить в середине научной очереди и назвать последующий run продолжением старой immutable code-binding цепочки. Он добавляет log snapshot в resume_state; исходные старые checkpoint его не содержат. Проверять на свежем tiny run, затем строить новую scientific queue. Отдельный новый цикл научного дизайна или обязательное повторное внешнее ревью не нужен при успешных регрессиях.

## Артефакты reviewer

- `archive_checks.json`, `data_checks.json`, `independent_source_checks.json` — собственные проверки исходного пакета.
- `cpu_tests.log`, `v4_check.log` — журналы реально выполненных тестов.
- `q1_replay/summary.json` — повтор результата из compact scores.
- `resume_checks.json` — воспроизведённые defects и isolated checks предложенных helper bodies.
- `check_resume_and_patch.py` — воспроизведение на неизменённом исходнике без HF/модели.
- `independent_q1_and_atomic.py` — дополнительная проверка labels и raw примеров.
- `resume_recovery.patch`, `test_research_resume_recovery.py` — предлагаемые изменения для рабочей среды.
