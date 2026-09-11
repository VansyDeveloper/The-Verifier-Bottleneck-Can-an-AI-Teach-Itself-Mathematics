# The Verifier Bottleneck — план следующей итерации

Дата: 10 сентября 2026 года.

## Назначение и границы

Это план реализации, а не отчёт о проведённых новых экспериментах. Основание: приложенные `36_The_Verifier_Bottleneck_Can.pdf`, `main10.tex`, три скриншота рецензий, переписка команды и статический просмотр `verifier_bottleneck_code_context.zip`. Обучение, оценка GPU-памяти и восстановление исходных весов в рамках подготовки плана не выполнялись.

Обозначение **«ты»** относится к исполнителю, которому нужен план; **Дима** — vanssy из переписки. Распределение опирается на последний абзац сообщения Димы, где он выбирает GRPO, разнообразие обучающих траекторий и расширение эксперимента с исключёнными парами. Упоминание им других идей не считается принятием их в работу. До запуска вычислений закрепите эту таблицу в репозитории.

## 1. Владельцы и отсутствие дублирования

| ID | Пакет | Единственный владелец | Граница |
|---|---|---|---|
| D1 | GRPO: группы all-wrong/mixed/all-correct, динамика accuracy/diversity/coverage; изменения exploration/novelty/entropy reward | Дима | Ты не реализуешь второй GRPO-тренер и не повторяешь его sweeps |
| D2 | Выбор случайных и разнообразных обучающих подмножеств; покрытие пар и промежуточных состояний | Дима | Ты не пишешь competing maximum-coverage selector |
| D3 | Исключение k=1,2,3,5 пар; несколько наборов на k; контроль случайного удаления | Дима | В твоих основных опытах остаётся один зафиксированный стандартный motif split |
| U0 | Общие форматы результатов, восстановление артефактов, версия протокола, тесты evaluator/budget | Ты | Один общий контракт; Дима экспортирует результаты в него, но не обязан менять исторический scorer |
| U1 | H(K), парные результаты, стратификация ошибок, отчёт по основной шестизапусковой серии | Ты | Дима передаёт свои результаты, не строит второй основной анализ |
| U2 | Replay 0/10/20/40%, сохранение каждой атомарной операции | Ты | Меняется доля replay, не алгоритм выбора разнообразных траекторий |
| U3 | Program-only против program + TRACE; контроль объёма supervision | Ты | Та же выборка задач и тот же witness; без нового поиска |
| U4 | Обучение на всём множестве правильных программ | Ты | Фиксированная offline-выборка и supervised objective, не модификация GRPO |
| U5 | Аудит семантики verifier noise, корректировка описания, при необходимости новая серия качества верификатора | Ты | Старый код Димы остаётся историческим; новые данные и код получают новую версию |
| U6 | Повтор основной пары на другом размере модели | Ты | Не переносишь автоматически всю программу D1–D3 на большую модель |
| U7 | Независимый внешний benchmark и диагностика domain shift | Ты | Новый домен не заменяет motif sweep Димы |
| U8 | Текст статьи, таблица claim→evidence, provenance, reproducibility appendix | Ты; Дима проверяет свои результаты | Не переписывать историю preregistration |

Сергей предлагает идеи и обсуждает выводы; отдельных вычислительных пакетов за ним не закрепляем.

**Общее использование не равно дублированию.** Можно оценивать модели Димы общим evaluator или использовать один baseline-checkpoint в нескольких сравнениях. Нельзя дважды обучить одинаковую ячейку под разными названиями и считать это независимыми экспериментами.

## 2. Что обнаружено в коде и почему это меняет реализацию

Все пути ниже относительны к распакованному `verifier_bottleneck_code_context.zip`.

Для краткости:

```text
ROOT_A = new_stuff/stage4_verifier_bottleneck_audit_v22/source/artifacts
CORE   = ROOT_A/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code
CONF   = ROOT_A/stage4_composition_confirm_v1_0p6b/code
SERIES2 = verifier_bottleneck_codex_pack
```

### 2.1. В архиве несколько разных реализаций

Корневой `modcomp/` относится к более ранней среде аффинных композиций. Для продолжения основной серии статьи точкой входа служат `CONF/confirm.py`, `CORE/composition_core.py`, `CORE/composition_model.py` и `CORE/composition_eval.py`. `SERIES2/src/vbexp/` — отдельная реализация второй серии.

Нельзя взять первый найденный `train.py` и предположить, что он воспроизводит основную таблицу статьи.

### 2.2. Нет исходных весов и построчных ranking-данных

В приложенном context-архиве есть код, конфигурации, сводки и manifests, но нет файлов весов и исходных `.jsonl` с задачами/ранжированиями. В `new_stuff/stage4_verifier_bottleneck_audit_v22/scientific/manifests/archive_manifest.json` есть ссылка на локальный полный архив `stage4_composition_confirmation_full.zip` объёмом 1 104 839 307 байт и его SHA-256. Это ссылка в manifest, а не наличие самого архива здесь. Manifest также отмечает отсутствие некоторых legacy model/dataset payloads.

Первое действие на своей машине — найти полный архив, atomic export, adapters, frozen task files и ranking shards и сверить их с manifests. Не восстанавливать H(K) интерполяцией нескольких агрегатов Hit@K.

### 2.3. Replay сейчас увеличивает общий объём

`CORE/composition_model.py:304–318`, `distill_example_specs`:

```python
replay_n = round(len(specs) * replay / (1.0 - replay))
```

Функция сохраняет все композиционные записи и добавляет атомарные. При изменении replay меняются число примеров и число шагов. Для предложения «заменить часть примеров при постоянном объёме» нужен другой sampler. Значение 1.0 отдельно обрабатывается как pure-atomic control, а не передаётся в эту формулу.

### 2.4. TRACE находится после программы

Та же функция формирует ответ:

```text
<OP...> <OP...> <OP...>
TRACE: [state_1] -> [state_2] -> [state_3]
```

То есть операции предсказываются раньше промежуточных состояний. U3 исследует дополнительное обучающее воздействие TRACE. Он не проверяет использование уже выписанного состояния при выборе следующего действия. Interleaved trace — отдельный эксперимент с другой факторизацией ответа и другим inference, не обязательная часть U3.

### 2.5. Полное множество правильных решений не равно сохранённым witness

`CORE/composition_core.py:261–292`, `exact_shortest_solutions(..., limit=2)`, считает число всех правильных кратчайших программ, но сохраняет максимум два решения. Для U4 нужно вновь перечислить всё пространство фиксированной длины и построить полную correct mask. Значение `total_count` не заменяет сами программы.

### 2.6. Существующий scorer нельзя непосредственно использовать как differentiable loss

`CORE/composition_eval.py:64–101`, `score_task`, помечен `@torch.inference_mode()` и превращает вероятности в Python `float`. Нужна tensor-only версия scorer без `inference_mode`, `.detach()`, `.item()`/`float()` внутри вычисления loss.

При глубине 3 требуется оценить 31 различный префикс: 1+5+25. При глубине 4 — 156. Это число префиксов, а не обещание соответствующего числа model calls: префиксы можно батчировать, а стоимость зависит от длины входа и реализации.

### 2.7. Метрики уже частично реализованы

`CORE/composition_eval.py:50–60`, `ranking_metrics`, уже считает Hit@1/8/16/32/64, `best_rank`, MRR, normalized correct-set mass. Для всех K достаточно построчного `best_rank`; пересчёт модели нужен только при отсутствии этих данных. Для entropy и распределения массы нужны candidate scores.

`CONF/confirm_stats.py` уже содержит per-seed paired counts, run-level t-statistics, exact sign-flip, crossed seed×task bootstrap и Holm. Их адаптировать и покрыть тестами, а не писать другую статистику параллельно.

### 2.8. Две серии используют разные определения score

Основная серия: log-softmax по полному словарю с добавленными специальными operation tokens. Вторая: softmax по пяти допустимым продолжениям. Это не одна и та же нормировка; порядок программ может измениться. В PDF различие раскрыто в приложении E, с. 20–21.

Сохранять `scorer_id`, `tokenizer_hash`, `prompt_version` в каждом результате. Общий формат результатов не означает, что две серии можно объединить в один массив реплик.

### 2.9. Обучающая смесь основной серии уже включает depth 4

`CORE/composition_core.py:26`: `TRAIN_DEPTH_COUNTS = {2:1000, 3:2000, 4:1000}`. Проверить фактический manifest исторического запуска, прежде чем трактовать depth-4 тест. Нельзя автоматически называть его extrapolation по длине. Во второй серии обучение на depth 2/3 и тест на depth 4 — другая постановка.

### 2.10. Старый loader не параметризован для всех новых опытов

`CORE/composition_model.py:250–270`, `load_frozen_atomic`, читает модель и параметры LoRA из глобального legacy `PROTOCOL`, а не полностью из аргументов нового запуска. Он сохраняет FP32 frozen weights и использует BF16 compute, требует определённого CUDA runtime. Это не готовая поддержка квантованного обучения другой модели.

Новый loader обязан использовать переданный resolved config и сохранять фактическую конфигурацию, а не только заявленную.

### 2.11. Нормировка loss и packing требуют отдельных тестов

`train_branch` усредняет microbatch loss и делит на accumulation. При разном количестве supervised tokens это не равно среднему loss по всем supervised tokens accumulation group. Последняя неполная группа также нуждается в корректной нормировке.

Для нового SFT trainer суммировать token losses и нормировать по фактическому числу loss-bearing tokens в update. Для set-objective определить среднее по задачам отдельно. После изменения trainer не считать новые числа буквальным продолжением старого протокола без контрольной bridge-проверки.

`pack_control_to_budget` использует reset position IDs. Нужен поведенческий тест изоляции packed segments: изменение первого примера не должно менять logits второго. Не полагаться только на наличие нужного имени функции в установленной библиотеке.

### 2.12. Скрипт шума реализует label corruption, а не заявленный conditional false-accept rate

`SERIES2/scripts/prepare_noisy_verifier_data.py`, функция `corrupt`, с вероятностью `beta` заменяет правильный supervised target неправильной программой той же длины. Это интервенция в долю неправильных обучающих ответов.

Conditional false-accept rate — другая величина: β=P(accept | incorrect). Перед изменением подписи опубликованного рисунка необходимо установить по manifests, этим ли скриптом и этой ли версией получены его данные. Статический просмотр показывает несоответствие семантики скрипта, но не устанавливает полную provenance каждого опубликованного числа.

## 3. Общая инфраструктура U0

### 3.1. Исторические артефакты неизменяемы

Старые код, protocol, splits и таблицы — read-only. Создать новый пакет, например:

```text
revision/
  protocols/
    ownership.yaml
    polynomial_vnext.yaml
    selected_confirmatory.yaml
  src/vbnext/
    environment.py
    task_schema.py
    targets.py
    mixtures.py
    budgets.py
    model_loader.py
    scoring.py
    losses.py
    evaluation.py
    statistics.py
    provenance.py
  scripts/
    inventory_artifacts.py
    export_legacy_metrics.py
    analyze_hitk.py
    prepare_training_stream.py
    train_sft.py
    train_set_objective.py
    audit_verifier_noise.py
    build_verifier_acceptance_sets.py
    run_external_validation.py
    build_paper_outputs.py
  tests/
  outputs/<experiment_id>/
```

Это предлагаемые новые имена, не существующие команды архива.

### 3.2. Контракт эксперимента

До запуска сохранять:

```yaml
experiment_id: <unique-id>
owner: user
study: U2
status: planned
protocol_hash: <sha256>
code_commit: <commit>
model_id: Qwen/Qwen3-0.6B
model_revision: <immutable-revision>
atomic_export_hash: <sha256>
tokenizer_hash: <sha256>
scorer_id: full_vocab_op_tokens_v1
prompt_version: stage4_plan_v1
train_manifest_hash: <sha256>
eval_manifest_hash: <sha256>
training_seed: <int>
data_seed: <int>
verifier_seed: null
target_format: program_trace
objective: token_ce
replay:
  unit: examples
  fraction: 0.2
budget:
  mode: fixed_exposures
  examples_per_epoch: 5000
  epochs: 2
  effective_batch: 64
selection_split: development
final_test_opened: false
```

Шаблон не является готовым конфигом запуска: placeholders должны быть разрешены, бюджет — рассчитан по фактической tokenization, dataset manifest — создан.

Ключ идентичности вычислительной ячейки строить из модели/инициализации, данных, seed, objective, target format, replay и budget. `owner` и название графика в ключ идентичности не включать: одинаковый эксперимент не должен повторяться из-за смены исполнителя.

### 3.3. Контракт результатов

Один task-level record:

```text
experiment_id, study, owner, protocol_hash, code_commit,
training_seed, checkpoint_step,
model_hash, tokenizer_hash, scorer_id, prompt_version,
dataset_version, split, task_id, task_fingerprint,
field, polynomial_degree_cap, program_depth,
candidate_count, correct_count,
best_rank, correct_set_mass, reciprocal_rank,
program_entropy, correct_conditional_entropy,
correct_effective_count, max_correct_conditional_probability,
format_parse_ok, execution_correct
```

Не все поля применимы к каждому домену или типу оценки: хранить null с причиной, а не ноль. Для entropy при единственном правильном решении отдельно отмечать вырожденный случай. Full ranking сохранять отдельно как `(task_id, candidate_id, program, score, correct, rank)`.

### 3.4. Обязательные проверки до GPU-серии

1. Реальные файлы и SHA совпадают с manifests; ни один отсутствующий shard не заменяется агрегатом.
2. Для каждого d перечислены ровно 5^d программ, без дублей; correct mask повторно проверена interpreter.
3. Для задач в ranking имеется хотя бы одна correct program; все tie-breaks определены.
4. Token IDs операций уникальны, сериализация и загрузка tokenizer не меняют их; train/scoring prefixes сопоставлены токен-в-токен.
5. Пары моделей оцениваются на тех же task IDs и в том же порядке.
6. Фактически показанные модели training inputs/targets не пересекаются с final evaluation по принятому leak-критерию.
7. Для U4 дополнительно проверены все supervised solutions, а не только один witness: ни один запрещённый motif не появился в labels. Если постановка требует отсутствия motif во всех решениях, отбрасывать несовместимую задачу целиком до фиксации данных, а не называть отфильтрованную correct mask «всеми правильными решениями».
8. Для TRACE дополнительно проверены реально предъявляемые промежуточные состояния. Не смешивать с латентными состояниями, которых модель в program-only target не видит.
9. Padding/packing invariance и корректная нормировка последнего optimizer update.
10. Save→reload сохраняет logits/метрики в заранее заданной численной погрешности.

### 3.5. Новые финальные наборы и статистический статус

Исторические A–D используются для воспроизведения опубликованных результатов и явно помеченного reanalysis. Для выбора новых долей replay/λ/learning rate нужен development, а для подтверждения выбранного нового эффекта — новый заранее зафиксированный final test, не участвующий в настройке. Новый final test не превращает старую гипотезу, придуманную после просмотра результатов, в исторически preregistered.

Три training seeds полезны для screening. Они не дают сильного exact sign-permutation подтверждения: минимальное двустороннее p при трёх ненулевых согласованных разностях равно 0.25. Основные новые claims разумно подтверждать отдельной заранее назначенной шестизапусковой серией; число запусков не увеличивать до появления значимости.

## 4. U1 — H(K), парность и ошибки

### Вопрос

Сдвигается ли распределение рангов в целом или вывод зависит от отсечки K=32? На каких заранее определённых типах задач наблюдаются улучшения и ухудшения?

### Реализация

Извлечь per-task `best_rank` R из существующих metrics либо ranking shards. Для каждой реплики s и каждой ветви:

H_s(K) = mean_x 1[R_s(x) <= K], K=1,...,125.

Для средней кривой усреднять по training seeds. Отдельно считать парную кривую Δ_s(K)=H_s,composition(K)−H_s,control(K). Сохранить все K в таблицу, не только выбранные красивые точки.

Для depth 2 использовать K<=25; для depth 4 — K<=625. Разные d не смешивать без явного правила усреднения.

Существующий Table 9 уже содержит counts both/control-only/composition-only/neither. Его сначала воспроизвести как regression test, затем добавить построчные различия и страты. Это расширение существующего анализа, не «впервые учтённая парность».

### Стратификация

Предварительно определить: family A/B/C/D, field, degree cap, program depth, число правильных программ (например 1,2,>=3), зависимость решения от SH1. Для SH1 различать «есть в одном из решений» и «необходим во всех правильных решениях». Не классифицировать задачу по произвольному canonical witness при нескольких решениях.

На задачу сохранять Δbest_rank, Δcorrect_mass и переход success/failure. На страту — размер, per-seed среднее изменение и интервал. Мелкие страты показывать описательно; не объединять их задним числом по знаку эффекта.

### Статистика

Основной независимый блок — training seed. Для robustness использовать crossed bootstrap: выбирать seeds и task IDs, причём выбранные task IDs одинаковы для обеих ветвей и всех выбранных seeds. Для curve bands указать, pointwise они или simultaneous. Не проверять 125 гипотез и не выбирать минимальное p. H@32 остаётся заранее выбранным endpoint; MRR или area under H(K) можно назначить вторичным.

Для гарантированно разрешимых depth-3 задач:

- H(K) не убывает;
- H(125)=1;
- mean_K H(K)=(126−mean R)/125;
- H(32) совпадает с сохранённым Hit@32;
- ΔH(K) в последней точке равна нулю.

Последний факт означает структурное насыщение конечного пространства, а не отсутствие улучшения модели.

### Выход и готовность

`hitk_by_seed.csv`, `hitk_summary.csv`, `paired_task_deltas.parquet`, `stratified_errors.csv`, графики A–D, paired difference plot и таблица источников. Числа основного H@32 и Table 9 должны совпасть с опубликованными в точности до согласованного rounding; иначе остановка и разбор provenance.

## 5. U2 — Replay и сохранение атомарных навыков

### Основная постановка: буквальная замена примеров

Рекомендуемая единица r — доля атомарных примеров в фиксированных 5000 exposure slots за эпоху:

| r | Composition | Atomic |
|---:|---:|---:|
| 0.0 | 5000 | 0 |
| 0.1 | 4500 | 500 |
| 0.2 | 4000 | 1000 |
| 0.4 | 3000 | 2000 |

Это новый protocol. Если в доступном pool только 4000 композиционных задач, либо расширить train-only pool до 5000 до фиксации протокола, либо выбрать меньший общий N. Не называть дополнительные повторы новыми уникальными задачами.

Для r=0.2 можно сохранить оригинальные 4000 задач, если они доступны и проходят новый leak audit. Для других долей использовать заранее заданные случайные nested subsets/supersets, сбалансированные по исходным полям/глубинам. Не внедрять maximum-coverage selection: этим занимается Дима.

Атомарные записи балансировать по 10 ячейкам: PLAN/APPLY × 5 операций. Источник каждой операции и field distribution сохранить. SH1 не исключать из оценки.

### Что фиксируется

Общая начальная atomic model, пять операций, tokenizer, prompt, target format, learning rate, LoRA, число exposure slots, epochs, effective batch, optimizer schedule, data-order seed pairing и final test. Для новой серии можно взять исходные настройки как стартовые: rLoRA=32, alpha=64, dropout=0.05, lr=1e-4, epochs=2, batch=64. Но resolved config обязан реально управлять loader/trainer.

### Честное сравнение бюджетов

Фиксированные N и steps НЕ гарантируют одинакового числа target tokens: атомарные ответы и TRACE различаются по длине. Сохранять в receipt:

```text
unique_tasks; example_exposures; optimizer_steps;
composition_target_tokens; atomic_plan_target_tokens; atomic_apply_target_tokens;
all_target_tokens; prompt_tokens; total_forward_tokens;
wall_time; peak_memory; examples_per_kind; operation_counts
```

Основной вывод U2 формулируется «при фиксированном числе предъявлений и optimizer steps».

Для выбранного на development контраста провести sensitivity-check при фиксированных target-token budget и steps. В этом режиме количество предъявлений и фактическая доля примеров могут измениться — их измерять, а не обещать сохранение всех бюджетов одновременно. Маскированный padding не является обучающими токенами. При approximate matching заранее задать допустимую погрешность и публиковать её.

Pure-atomic control для сравнения с основной r=0.2 ветвью строить как отдельную budget-matched reference. Он не становится автоматически token-matched контролем для всех четырёх r.

### Оценка

Общая H(K), H@32 и correct-set mass на A–D. Для каждой операции — PLAN accuracy, APPLY exact accuracy, parse rate и semantic accuracy условно на корректно распарсенных ответах; исходные выходы сохраняются. Иначе потеря формата может быть ошибочно названа потерей арифметики.

Считать изменение относительно starting atomic checkpoint и отдельно относительно pure-atomic control. Показывать SH1 собственной строкой, а не только средним по пяти операциям. Для нового диагностического atomic test разумно предусмотреть сотни задач на операцию и распределение по полям/степеням; точный объём рассчитывается из требуемой точности. Пара сотен задач не гарантирует доказательства non-inferiority с margin 2 п.п.

Оценить начальную, промежуточную и финальную модели на development. Финальный checkpoint либо последний по протоколу, либо выбран по заранее объявленному правилу; final test не используется для выбора момента остановки.

### Решение

Построить trade-off: composition gain versus SH1/APPLY loss. Выбрать r по development при заранее обсуждённом допустимом ухудшении, затем проверить на final. Отсутствие статистически значимого ухудшения не доказывает сохранность навыка. Если все r теряют SH1, это самостоятельный результат; опциональный следующий опыт — uniform replay против SH1-weighted replay при том же r, а не произвольное увеличение данных.

### Запуски

Screening: 4 r × 3 seeds = 12 ветвей. Основной atomic control — ещё 3, если подходящего идентичного control нет. Точную r=0.2/full-trace ячейку использовать повторно в U3, когда совпадают все hashes/configs.

## 6. U3 — Польза промежуточных состояний

### Контраст

На тех же задачах, с тем же выбранным правильным witness, compare:

```text
PROGRAM_ONLY:   <OP...> <OP...> <OP...>
PROGRAM_TRACE:  <OP...> <OP...> <OP...>
                TRACE: [s1] -> [s2] -> [s3]
```

Prompt, префикс программы и порядок операций сохраняются; меняется наличие TRACE в supervised target. Для PLAN answer-only означает программу без состояний, а не конечный вектор: конечный target уже находится в prompt.

### Три полезные ячейки

1. Full trace, фиксированные task exposures — reference U2/r=0.2.
2. Program-only, те же task exposures и steps — основной контраст содержания supervision.
3. Program-only, matched target-token budget — sensitivity к количеству обучения; допускается больше повторов/другая упаковка, всё фиксируется в receipt.

Третья ячейка не одновременно matches уникальные задачи, число повторов, длины ответов и FLOPs. Она отвечает на другой контрольный вопрос и должна быть подписана отдельно.

Сравнение старой первой серии со старой второй не заменяет этот эксперимент: у них отличаются data, score, training и retention protocol.

### Реализация

Вынести `render_target(task, witness, format)` в `targets.py`. Эталонные fixtures сравнивают текст и токены. Сохранять раздельные маски operation/trace/EOS для диагностики и суммы token losses. Префикс программы в двух форматах должен быть одинаковым до места, где добавляется TRACE.

Для основной абляции не использовать интерливинг состояний между действиями: это изменит условные распределения, используемые frozen evaluator. Такая постановка допустима позже с отдельным scorer_id.

### Выводы

Если program-only не хуже в практически значимом диапазоне, основная польза может объясняться supervision правильных программ без необходимости TRACE. Если full trace лучше и fixed-exposure, и token-budget control, это более сильное свидетельство дополнительной пользы состояний. Не называть разницу доказательством внутреннего «пошагового рассуждения».

### Выход

Таблица трёх ячеек с H@32, MRR, correct mass, SH1/APPLY, budgets; graph H(K); per-seed paired differences. Минимально нужно ещё 3 program-only runs при reuse full-trace reference, token-matched arm — ещё 3.

## 7. U4 — Всё множество правильных программ

### Цель и область

Проверить, полезно ли обучать модель относить массу ко всему correct set вместо одного witness. Это offline oracle supervision, а не самостоятельное открытие программ моделью.

Для глубины 3 пространство имеет 125 программ. В исходной смеси есть также depth 2 и 4: либо реализовать полный objective для 25/125/625 программ на исходной смеси, либо объявить отдельный depth-3-only subexperiment и обучить все его controls заново. Нельзя фильтровать обучение до depth 3 и сравнивать с историческим mixed-depth baseline как с единственным изменённым loss.

### Определение

Пусть P_d — все программы длины d, C(x) — все программы из P_d, которые достигают target. s_theta(p|x) — сумма operation scores по frozen scoring convention. Определим распределение внутри полного candidate space:

q_theta(p|x) = exp(s_theta(p|x)) / sum_{u in P_d} exp(s_theta(u|x)).

Тогда:

```text
L_single_norm(x) = logsumexp(s_all) - s_witness
L_set(x)         = logsumexp(s_all) - logsumexp(s_correct)
L_uniform(x)     = logsumexp(s_all) - mean(s_correct)
```

Последний objective — cross-entropy с равномерной целью по correct set.

**Различать raw LM sequence mass и candidate-normalized mass.** Эти формулы оптимизируют вторую, соответствующую normalized correct-set mass в основной оценке. Не называть её автоматически вероятностью корректного свободного текстового ответа. Нативные invalid strings вне P_d здесь исключены условием.

### Минимальная матрица

- Existing program-only SFT: практическая reference, но другой тип normalization.
- Normalized single-witness loss: необходимый matched-objective control.
- Set-mass loss: основной новый метод.
- Uniform-correct loss: диагностический дополнительный вариант, если задача включает именно сохранение нескольких решений.

Все normalized arms используют одинаковые candidate sets, data stream, optimizer updates, replay, формат и training seeds. Per-task loss усредняется по задачам, а не по числу правильных программ. Atomic replay loss и его коэффициент одинаковы для этих arms. Если выбран отдельный коэффициент между sequence CE replay и set loss, он фиксируется до final и применяется также к normalized-single control.

### Почему set mass может не повысить diversity

При q(correct set)≈1 loss может быть почти минимальным, даже если одна правильная программа получает почти всю массу. Поэтому «set loss предотвращает collapse» — гипотеза, не следствие формулы.

Нужно отдельно считать распределение q(p | p in C,x), его entropy, exp(entropy), максимальную условную вероятность и |C|. Основной diversity-анализ — на |C|>=2. Рост общей entropy может означать рост массы неправильных программ; его недостаточно.

При необходимости добавить отдельно зарегистрированный `L_set - lambda * H(q(.|C))`. Это supervised regularization твоего пакета, а не entropy reward GRPO Димы. Lambda выбирать по development, показывать весь заранее назначенный небольшой grid, а не только лучший final.

### Реализация tensor scorer

API:

```python
score_program_tree(model, tokenizer, task, scorer_spec) -> Tensor[num_programs]
correct_mask(task, programs, interpreter) -> BoolTensor[num_programs]
set_objective(scores, mask, reduction="mean_tasks") -> Tensor
```

Использовать shared prefixes, устойчивый logsumexp в FP32, microbatching и gradient checkpointing по необходимости. Не обрывать gradients при сборке scores из отдельных prefix batches. Если весь graph не помещается, сначала уменьшить task batch; переход на sampled candidates должен получать другое имя objective и отдельный protocol.

Не передавать gold correct set в prompt. Он используется для построения loss. Training candidate labels включают oracle-информацию обо всех решениях; её стоимость и объём нужно признать при сравнении с одним witness.

### Тесты

- При singleton C set loss и normalized-single совпадают по значению и градиенту.
- При C=P_d normalized set loss равен нулю и имеет нулевой градиент.
- Перестановка candidates не меняет loss.
- Tree scorer и brute-force version совпадают по значениям и gradients на маленькой игрушечной модели/фиксированных logits.
- Оптимизация на одном task повышает q(C) при малом проверочном шаге в численно стабильном режиме.
- Stable finite loss при очень отрицательных scores.
- CPU gold labels совпадают с replay interpreter; C не усечено `limit=2`.
- Save/reload, token-prefix alignment и операция с padding проходят общие тесты U0.

### Бюджет и результат

Normalized-single и set-mass оба считают полное candidate space: это честная пара по evaluator overhead. Обычный sequence SFT может быть намного дешевле; публиковать время, число scored prefixes и peak memory, а не утверждать одинаковые FLOPs по равному числу optimizer steps.

Эксперимент удачен научно и при отрицательном результате: он должен различить «масса правильного множества», «позиция лучшего ответа» и «несколько вероятных правильных ответов».

## 8. U5 — Что делать с критикой verifier-quality / two-dial law

### Сначала provenance и обозначения

Для текущего `prepare_noisy_verifier_data.py` естественная переменная — rho, доля неправильных targets в supervised dataset. Она не равна β=P(accept|incorrect).

Пусть q — доля правильных программ в исходном candidate proposal, α=P(accept|correct), β=P(accept|incorrect). Тогда среди принятых:

rho = beta*(1-q) / (alpha*q + beta*(1-q)).

Например, для uniform proposals при одном правильном ответе из 125, α=1 и β=0.5 дают rho=62/63≈98.4%, а не 50%. Если q=1, conditional beta вообще не наблюдается на предложениях, содержащих только правильные примеры.

Сопоставить script hash, source dataset hash, corruption manifest и output table/figure. Если график получен заменой targets, обозначить его как label-corruption robustness. Не заявлять, что он измерил точку разрушения при определённом false-accept rate. Сохранить историческую маркировку и correction note, не делать вид, что другой опыт был проведён раньше.

### Рекомендуемый обязательный маршрут

Сузить claims: измерены consolidation under oracle supervision и границы исследованных алгоритмов; универсальный two-dial law не установлен. Review допускает этот маршрут вместо большой сетки.

Текущая версия abstract уже осторожнее review-формулировок. Проверить title, introduction, discussion, captions и conclusion на оставшиеся более сильные фразы. Отдельно не писать «стандартный RL не подходит»: имеющиеся данные относятся к конкретному GRPO, budget и schedule.

### Усиление при наличии ресурсов

Вариант A: честный multi-seed sweep label corruption rho∈{0,0.1,0.25,0.5}. Это относительно дешёвая проверка robustness supervised learning, но не закон качества верификатора.

Вариант B: настоящий acceptance process. Создать замороженный proposal bank `(task, candidate, gold_correct)`, а затем принимать кандидаты вероятностно по заданным α/β. Пример небольшого grid:

```text
(1.0, 0.0), (0.5, 0.0), (1.0, 0.1), (1.0, 0.5), (0.5, 0.5)
```

Для каждого уровня сохранять TP/FP/FN/TN, acceptance rate, accepted pollution, число задач без принятых кандидатов, unique accepted programs и число проверок. Random draws привязать к `(task_id, candidate_id, verifier_seed)`, чтобы смена порядка файлов не меняла corruption pattern. Actual acceptance rates имеют sampling error; не ждать точного совпадения с параметрами на конечной выборке.

Gold verifier используется для аудита и финальной оценки, а не для повторного удаления false positives после noisy acceptance. Иначе интервенция будет отменена. Если accepted-wrong program становится target, состояния строятся реальным interpreter для этой программы; нельзя незаметно рисовать траекторию, которая всё равно достигает исходного target.

### Два разных бюджета

Primary: одинаковый объём принятой supervision, а proposal/checker costs и число повторов публикуются. Это отделяет качество labels от недополученного количества training.

Secondary: одинаковый proposal/checker budget, accepted supervision может уменьшиться. Это измеряет весь pipeline. Нельзя обещать одновременно фиксированное число проверок, количество принятых уникальных задач и training tokens при любом α/β.

### Если нужен именно joint quality × exploration map

Повторить заранее выбранные verifier regimes на двух фиксированных proposal banks разного покрытия, полученных из артефактов Димы. Ты реализуешь принятие/обучение/анализ, Дима — единственный владелец генерации exploration arms. Отдельный GRPO sweep не запускать. Определить operational collapse criterion заранее; несколько шумовых уровней без его определения не образуют «закон».

## 9. U6 — Размер модели

Для сравнения в той же исходной dense-линейке Qwen3 следующий размер после 0.6B — 1.7B, затем 4B. Проверено по официальному описанию Qwen3. 1.5B/3B из другой линейки не изолируют размер.

Первый шаг — smoke test настоящего training: загрузка, forward, backward, optimizer step, save/reload, exact depth-3 ranking. Успешный quantized inference не гарантирует, что поместятся LoRA training и activation graphs.

Нужна собственная atomic initialization для каждой модели; adapter 0.6B нельзя переносить в 1.7B. Внутри новой модели обе ветви стартуют из одного frozen atomic checkpoint. Исторический model-size lock не редактировать; новая модель получает другой protocol.

Минимальная replication: pure-atomic control versus full-trace composition при r=0.2 или другом заранее выбранном на development режиме, 3 paired seeds, A–D и atomic retention. Не повторять на большой модели GRPO/motif/diversity программы Димы.

Если 1.7B возможна только в 4-bit, выполнить 0.6B bridge в той же quantization/training configuration. Тогда сравнение размера не смешано только с изменением численной точности. Исторический FP32-master/BF16-compute результат оставить отдельной reference.

Выход: не scaling law, а проверка устойчивости основного контраста к размеру; memory/time receipt и явные различия protocol.

## 10. U7 — Независимая внешняя проверка

### Что этот пакет закрывает

Новые поля, ещё одна тройка motifs или большая модель всё ещё остаются внутри собственного synthetic generator. Для просьбы reviewer о real or independently sourced dataset требуется отдельный внешний источник.

Рабочий кандидат: публичные SyGuS competition benchmarks, в частности подходящий фрагмент PBE/string-program synthesis. Это внешне собранные задачи синтеза программ, а не автоматическое доказательство переноса на реальную математику. Официальный репозиторий предупреждает об ошибках и дубликатах — их проверка входит в работу.

### U7.0. Feasibility до обучения

На небольшой development-подвыборке проверить parser, grammar, typed interpreter, число допустимых программ при заранее объявленном ограничении размера AST, число разрешимых задач и доступность reference/дополнительных тестов. Оценить runtime полного ранжирования.

Не подменять исходную задачу собственной игрушкой из пяти операций ради получения ровно 125 candidates. Допустима явно обозначенная bounded-grammar подзадача, но нужно сохранить исходные constraint files и долю задач, которые в неё не помещаются. Grammar/candidate enumeration не зависят от модельных scores и успешности обученной модели.

До выбора final subset определить критерии включения по данным/грамматике/вычислительной допустимости, а не по знаку эффекта SFT. Количество финальных задач задаётся после инвентаризации, до чтения результатов моделей; нельзя обещать наличие нужных 100/200 независимых tasks без проверки.

### U7.1. Данные и независимость

Сохранить source commit, file hash, original task ID, grammar, constraints, candidate bound и inclusion/exclusion reason. Удалить дубликаты и делать split по source/template families, где возможно. Отдельные input-output пары одной задачи не считать независимыми benchmark tasks.

При PBE correct означает выполнение заданных constraints. Если нет reference function или независимых проверочных IO, нельзя утверждать равенство программ на всех входах. Если дополнительная проверка доступна, заранее разделить constraints для synthesis и semantic holdout и использовать это одинаково во всех arms.

### U7.2. Основная пара

В новой DSL создать атомарную preparation stage и одинаковый starting checkpoint для двух branches. Сравнить atomic-only continuation с composition-supervised continuation при общем budget. Не требовать, чтобы polynomial adapter без адаптации понимал новые string operations: такая неудача не изолирует исследуемый механизм.

Оценивать на independent benchmark tasks: pass of original constraints, best correct rank/H(K) в bounded exhaustive space, normalized correct mass, parse validity, elementary-operation retention. При переменном числе candidates дополнительно дать M и C, curves by M strata или заранее выбранные normalized rank summaries. Не усреднять бессмысленно H@32, если у части задач меньше 32 программ.

### U7.3. Domain-shift diagnostic

Построить две test families в одной новой DSL: внутренние generated tasks и независимые benchmark tasks. На обеих оценить ту же atomic/composition пару. По возможности matched по программе/grammar size, числу constraints и числу правильных решений.

Отдельный positive control: composition supervision из разрешённой внешней train части, evaluated на disjoint external test. Если такое обучение помогает, а обучение на внутреннем generator — нет, это аргумент в пользу distribution mismatch. Если не помогает, проверить representability, elementary skill mastery, parsing и optimizer before claiming mechanism failure. Отрицательный результат сам по себе не позволяет логически доказать единственную причину.

### U7.4. Denominators и выход

Показать весь pipeline: исходных tasks → корректно разобранных → допустимых по grammar bound → имеющих solution → успешно ранжированных. Отдельно дать результат на всём исходном test scope и conditional ranking result на представимых задачах. Не скрывать coverage bound.

Выход: dataset card, frozen manifest, adapter/interpreter tests, paired results, domain-shift table, qualitative failures и предел внешнего claim. Если feasibility не проходит или ресурсов нет, reviewer-пункт остаётся открытым; ещё одна внутренняя synthetic проверка его не заменяет.

## 11. U8 — Изменения статьи и response matrix

| Замечание | Изменение/опыт | Что нельзя утверждать без результата |
|---|---|---|
| Только Hit@32 | U1: полная H(K), парная ΔH(K), MRR/ranks | Одно пересечение при K=32 не означает улучшение для всех K |
| Только агрегаты | U1: reproduce Table 9, release task-level differences, strata | Tasks внутри run не независимые модели |
| Неясна польза full trajectories | U3, одинаковые tasks/witness, exposure и token controls | TRACE-after-program не демонстрирует inference-time reasoning |
| Забывание SH1 | U2, per-operation/mode/parse metrics | p>0.05 для ухудшения не доказывает retention |
| Мало seeds для вторичных эффектов | Заранее определённые replication и отдельная confirmation | Добавлять seeds до significance нельзя |
| Одна модель/размер | U6 | Две точки не scaling law; одна семья не cross-architecture validation |
| Только собственная synthetic среда | U7 | Новые primes или 1.7B сами по себе не external validation |
| Не установлен two-dial law | U5: сузить claim или настоящий quality×proposal experiment | Label-noise sweep не измеряет conditional beta |
| Supervision из exhaustive search названа self-training | U8: разделить oracle discover-and-distill и policy-generated self-training | U4 с полным C не является self-discovery |
| Нет unseen motif generalization | D3; твоя интеграция figures и границ claim | Хороший результат на familiar motifs не новая универсальная композиционность |
| Структура/разнообразие поиска | D2/D1 | Старый matched-diversity result не доказывает причинную пользу конкретного train selector |
| Статус prereg/amended/posthoc | U8: отдельная колонка status на каждый contrast | Новый текст не изменяет момент создания amendment |
| Недостаточно prompt/data details | U0/U8: exact PLAN/APPLY/TRACE fixtures, token map, atomic pool composition, hashes | Нельзя давать выдуманные «типичные» prompts вместо использованных |
| Чрезмерное обобщение GRPO | D1/U8: algorithm/budget/sampler-specific statement | Нельзя делать вывод «стандартный RL не подходит» |

### Предлагаемая редакционная позиция

Рабочее узкое название: **Verified Composition Learning under Exhaustive Program Ranking**. Это предложение, а не требование обязательно переименовать статью.

В introduction разделить три вопроса: улучшение ranking знакомых композиционных семейств; перенос на withheld motifs; самостоятельное обнаружение новых полезных программ. Их не считать эквивалентными.

В methods точно описать источник targets: exhaustive oracle versus model proposals. В discussion заменить буквальные утверждения о «появлении support» операциональными метриками: ranking, finite-budget coverage, mass, entropy. При softmax ненулевая формальная вероятность не равна практической обнаружимости.

В Appendix E отдельно сохранить результаты двух scorers/двух статистических designs. Table 11 и подпись должны согласованно различать original preregistration, disclosed amendment и post-hoc checks; не обозначать E4 независимой заранее зарегистрированной проверкой, если текст сообщает обратное.

Добавить reproduction appendix: scripts/CLI after implementation, package lock, model/tokenizer revisions, data schemas, candidate enumeration, tie breaks, budget receipts, all planned runs including failures, raw atomic outputs, postrочные metric files и источник каждого рисунка.

## 12. Последовательность и вычислительный бюджет

### Волна 0 — без нового обучения

Закрепить ownership → inventory → reproduce H@32/Table 9 → full H(K) → code/protocol tests → provenance noise audit → новая framing/response matrix. Параллельно запустить только feasibility внешнего benchmark и memory smoke большей модели.

Если raw metrics найдены, U1 — CPU analysis. Если их нет, но есть adapters и frozen tasks, нужен inference/rescoring. Если нет adapters или task identities, реальное воспроизведение не объявлять завершённым.

### Волна 1 — обязательные содержательные ablations

U2: 12 runs для четырёх replay levels × 3 seeds.

Pure-atomic reference для основной r=0.2 пары: 3 runs, если нет строго идентичной готовой ячейки.

U3 program-only: 3 runs; full-trace r=0.2 reuse.

Итого ориентир **18 training branches** для initial screening, до дополнительных token-budget controls. Это не оценка времени: runtime надо получить из собственного smoke и pilot. Если нужные initial checkpoints требуют восстановления/переобучения, их стоимость считать отдельно.

### Волна 2 — новый supervised objective

U4 normalized-single versus set: 2 × 3 = 6 runs. Uniform-correct — ещё 3 при необходимости. Начать на depth-3 smoke, затем либо поддержать исходные depths 2/3/4, либо честно открыть отдельный depth-3 protocol с его controls. Не брать полную сетку replay×objective×size×motifs.

### Волна 3 — расширение доказательств

U6: собственная atomic initialization большей модели плюс 2 × 3 continuation branches, и quantization bridge по необходимости.

U7: финальный размер и число arms после feasibility; независимый train/test split фиксируется до model selection.

U5: multi-seed corruption или genuine verifier grid только по выбранной версии claim. Полная joint grid не обязательна, если статья не заявляет universal law.

### Волна 4 — подтверждение выбранного центрального нового вывода

Выбрать один-два важнейших заранее сформулированных contrast по development. Для нового confirmatory claim зафиксировать protocol, final data, число training seeds и stopping rule до доступа к результатам final. Не выдавать screening с выбранными по final параметрами за подтверждение.

### Финальные условия готовности

- Каждый reviewer-пункт имеет пакет, владельца и статус: answered by analysis / answered by experiment / narrowed claim / open limitation.
- Ни одна ячейка Димы не переобучена у тебя без технической причины, записанной в registry.
- Каждый график воспроизводится из task-level файлов, а не вручную введённых средних.
- В каждом новом сравнении записано, какой именно budget matched, а какой только measured.
- Все negative results и failed runs остаются в run index.
- Для дополнительных правильных решений, нового verifier process, новой model precision и внешнего benchmark есть свои проверки, а не только общий training-success status.

## 13. Сообщение для фиксации распределения

> За тобой оставляем GRPO с диагностикой и возможными изменениями exploration/reward, выбор разнообразных траекторий и расширенный sweep исключённых пар. Я беру H(K) и парный/стратифицированный анализ, replay и забывание, program-only против TRACE, обучение на всём correct set, проверку другого размера модели, внешний benchmark и доработку статьи. По verifier noise отдельно проверю семантику и provenance старого эксперимента; новый sweep сделаю только в рамках согласованного claim. Общие split manifests, формат результатов и baselines согласуем один раз. Твои эксперименты не переобучаю; для общих графиков беру твои экспорты.

## Источники для самостоятельной проверки

- `36_The_Verifier_Bottleneck_Can.pdf`: среда и методы с. 4–5; основной результат/retention с. 6–9; per-run paired counts с. 17; различия серий и provenance с. 20–22; atomic formatting/retention с. 26; verifier figure с. 28.
- `main10.tex`: текущий title, abstract, introduction, discussion и Appendix E.
- Review qwGX: screenshot `69538fe0-8a55-410c-941d-efc68ffe477e.png`.
- Review cUuo: screenshot `75a687c0-0600-4873-a709-2e241dde37ff.png`.
- Review EFRP: screenshot `cc91299c-fc4b-4ccd-a781-9348bc021ed7.png`.
- Код: точные пути и диапазоны функций приведены в разделе 2; исторические файлы не изменялись при подготовке плана.
- Внешний источник по размерам: официальный материал Qwen Team «Qwen3: Think Deeper, Act Faster», 29 апреля 2025 года.
- Внешний источник по benchmark: официальный публичный репозиторий `SyGuS-Org/benchmarks`, его README и competition PBE/string tasks. Совместимость конкретной подвыборки с вашим bounded evaluator ещё требуется проверить.
