# Протокол v3: после пилота

Код реализует план [22 сентября](source/22sept_iclr_feedback/RESEARCH_PLAN_RU.md).
Полные модельные эксперименты выполняет оператор на своих GPU. Все значения
по умолчанию — screening: одна маска, один continuation seed, общий checkpoint.
Источник параметров и основного контраста — [research_v3.json](research_v3.json).

## Данные

При отсутствии общего v2 восстановите задачи один раз на CPU:

```bash
uv run python -m iclr.data --out outputs/shared_inputs_feedback_v2/reference_data
uv run python -m iclr.upgrade_data --reference outputs/shared_inputs_feedback_v2/reference_data --out outputs/shared_inputs_feedback_v2/new_tasks
uv run python -m iclr.research_data --source outputs/shared_inputs_feedback_v2/new_tasks --reference outputs/shared_inputs_feedback_v2/reference_data --out outputs/shared_inputs_v3
tar -czf shared_inputs_v3.tar.gz outputs/shared_inputs_v3
```

На других машинах: `tar -xzf shared_inputs_v3.tar.gz` из корня checkout.
Повторная генерация на каждой машине не нужна. Генератор отказывается
перезаписывать данные. `audit.json` содержит полную поддержку программ,
позиционные пары, поля/размерности, кратчайшую длину и статистику поиска.
Новые панели проверяются прямым интерпретатором по всем решениям и состояниям;
обычные задачи v2 копируются с проверкой хешей. Перекрёстные train-панели
допускаются при 64 панелях и >=8 программах; иначе применяется four-TARGET loss.
Обе конструкции оцениваются отдельно. Поиск состояний ограничен и не является
доказательством отсутствия других перекрёстных панелей.

## Веса и стоимость

Q1 читает прежние atomic checkpoints и обе continuation-ветки seed0/1/2
для выбранных размеров модели. Q2 начинает с прежнего composition seed0;
Q3/Q4 — с atomic checkpoint. Хеши проверяются по историческим квитанциям.
Q2 не является повтором прежней 400-шаговой GRPO-серии на Base.

Новые обучения по умолчанию: Q3 — четыре; Q2/Q4 screen — ноль.
Каждый conditional pilot добавляет четыре обучения только явной командой.
До продолжений оценивается исходный checkpoint на тех же данных; atomic
PLAN/APPLY сохраняются до и после обучения. Низкая исходная atomic accuracy
ограничивает интерпретацию утверждений об освоенных атомарных навыках.

Scorer перебирает 125 программ d3 и 625 d4. В budget записываются число
экспозиций, forward-токены, время и память, обращения к меткам для objective
и отдельно для диагностики. Reward-пилот в обеих ветках строит полное local
распределение; это механизм-диагностика, не бюджетная имитация обычного RL.
Меткам соответствует сохранённый CPU correct set; новые вызовы интерпретатора
во время reward-loss не выполняются. Exact имеет oracle-доступ ко всем меткам.

## Q2: reward pilot

После `Q2_diagnostic` изучите `gradient_diagnostic.json` и `gate.json`:

```bash
uv run python -m iclr.research start --queue-name Q2 --stage pilot \
  --gate outputs/Q2_diagnostic/diagnostics/q06_mask1_diagnostic \
  --inputs outputs/shared_inputs_v3 --out outputs/Q2_pilot --gpus 0
```

Допуск требует kernel-проверки, MC mean в пределах четырёх оценённых SE,
изолированных Adam-шагов в FP32 и BF16 и не менее четырёх train-задач.
У всех изолированных шагов одинаковые начальные веса и пустые моменты Adam;
Taylor-прогноз использует фактическое изменение параметров. Dev-функционал
только измеряется, не входит в loss и не фильтрует train-обновления.
CPU или одношаговый tiny smoke не даёт научного допуска.

## Q4: state/SIGReg pilot

```bash
uv run python -m iclr.research start --queue-name Q4 --stage pilot \
  --gate outputs/Q4_diagnostic/diagnostics/q06_mask1_diagnostic \
  --inputs outputs/shared_inputs_v3 --out outputs/Q4_pilot --gpus 0
```

Допуск: >=8 dev-панелей в B и D, положительная нижняя граница парного
95% bootstrap для state-access против обычного режима и token-control.
Состояние вычисляется только после выбранного моделью префикса. Control
получает столько же дополнительных токенов при том же префиксе; совпадение
их числа обеспечивается на уровне token IDs.

Все четыре обучающих условия имеют одинаковую компактную state-проекцию
с выходом в action logits. State loss классифицирует коэффициенты modulo p.
SIGReg — Epps–Pulley по случайным единичным проекциям, согласно
[LeJEPA, §4](https://arxiv.org/html/2511.08544v1#S4); его гарантии на LLM не переносятся.
Повторяющиеся точные состояния дают одну усреднённую точку regularizer.
Оценка включает отключение пути state→action, state reconstruction,
atomic retention, условные метрики и обычное решение задач.

В новом dev-пилоте можно явно задать `--cf-weight`, `--tau`, `--entropy-weight`,
`--state-weight`, `--sigreg-weight`, `--learning-rate` и новый `--out`.
Конфиги после планирования защищены хешами; confirmation/final наследуют выбранные значения.

## Выбор пары и подтверждение

Выберите метод по dev и сохраните основание решения; команда ничего не выбирает автоматически:

```bash
uv run python -m iclr.research select --queue outputs/Q3_dev/queue.json \
  --method ce_cf --reason 'Описание результата dev и выбора метода' --out outputs/selection_v3.json
uv run python -m iclr.research start --queue-name confirm --selection outputs/selection_v3.json \
  --inputs outputs/shared_inputs_v3 --out outputs/confirm_dev --gpus 0 1
```

Пара — baseline семейства и один выбранный метод; маски 1/2, seeds 0/1/2.
Это 12 continuation-ячеек. Они используют общую atomic initialization,
а не 12 независимых полных обучений. Для выбранных reward/state методов
каждая маска сначала проходит свой диагностический допуск.
Параметры, модель и precision наследуются из dev selection.

## Закрытая final-оценка

```bash
uv run python -m iclr.research freeze --queues outputs/confirm_dev/queue.json --out outputs/analysis_lock_v3.json
uv run python -m iclr.research start --queue-name confirm --selection outputs/selection_v3.json \
  --inputs outputs/shared_inputs_v3 --phase final --analysis-lock outputs/analysis_lock_v3.json \
  --trained-from outputs/confirm_dev --out outputs/confirm_final --gpus 0 1
```

Final использует сохранённые финальные adapters; обучающих заданий нет.
Lock фиксирует код, план, данные, выбор метода и checkpoints. Для Q1 заморозьте
её dev-очередь и передайте `--phase final --analysis-lock ...`; `--trained-from`
не нужен. Анализ v2 остаётся историческим: его исходная версия кода — `1566f5d`.

## Второй домен

После выбора пары подготовьте отдельный конечный нелинейный list DSL:

```bash
uv run python -m iclr.research_dsl --out outputs/shared_inputs_listdsl
uv run python -m iclr.research start --queue-name external --selection outputs/selection_v3.json \
  --inputs outputs/shared_inputs_listdsl --out outputs/listdsl_dev --gpus 0
```

Это две continuation-ветки, seed0. Операции: инкремент, квадрат первого
элемента, разворот, сортировка и циклический сдвиг списка. Их определения
включены в каждый prompt; символы OP0–OP4 переиспользуются как токены.
Это новый малый DSL, не результат на опубликованном RobustFill/DeepCoder.
Обучение d3, оценка d3/d4; все решения перебираются точно. Affine-разложение
энтропии к этому домену не применяется. Данные и final lock независимы.
Для reward/state выбранного метода здесь также нужен собственный допуск.

## Результаты

Основной контраст v3 screening: `ce_cf − ce`, local pair alignment,
равное среднее B/D. Перекрёстный I, точность каждой клетки и совместный успех,
H1/H8, correct-set mass, entropy decomposition и температуры — вторичные метрики.
`summary.json` и `analysis/summary.csv` разделяют policy, семьи и типы задач.
Full-score global softmax и local prefix policy не объединяются.

```bash
uv run python -m iclr.research_analysis \
  --runs outputs/Q3_dev/evaluations/q06_mask1_ce_seed0 outputs/Q3_dev/evaluations/q06_mask1_ce_cf_seed0 \
  --out outputs/Q3_contrast.json
```

Можно передать все парные seeds и обе маски. Сначала усредняются seed,
затем 2 000 парных ресэмплов панелей отдельно в B/D. Маски показаны отдельно
и равным средним; CI условен на этих checkpoints. Для Q2 задайте
`--control sampled --treatment exact`, для Q4 — `--control state_ce --treatment state`.

```bash
uv run python -m iclr.research send --out outputs/Q3_dev
uv run python -m iclr.research verify-bundle --archive outputs/send_to_artem_exp_ИМЯ.zip
```

Внешние dev-очереди, решения о выборе метода, analysis locks, диагностические
допуски и записи обучения автоматически включаются в `dependencies/`.
`DEPENDENCIES.json` связывает исходные пути с путями внутри архива. Хеши
исходных квитанций сохраняются; файлы весов, включая state projection, исключены.

Для частичных результатов: `send --allow-incomplete` после обновления статуса
через `run`. Копирование ZIP не отправляет его кому-либо автоматически.
