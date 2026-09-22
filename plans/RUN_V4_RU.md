# Протокол v4: после screening

Источник — [второй отзыв](source/22sept_iclr_feedback2/REVIEW_RU.md).
Параметры и правило успеха зафиксированы в [research_v4.json](research_v4.json).
Полные модельные эксперименты выполняет оператор на своих GPU. Малый пилот
проверяет перспективность метода; отрицательный результат ещё не исключает его пользу.

## Данные

Если общего v2 нет, один раз восстановите его и подготовьте v4 на CPU:

```bash
uv run python -m iclr.data --out outputs/shared_inputs_feedback_v2/reference_data
uv run python -m iclr.upgrade_data --reference outputs/shared_inputs_feedback_v2/reference_data --out outputs/shared_inputs_feedback_v2/new_tasks
uv run python -m iclr.research_data --source outputs/shared_inputs_feedback_v2/new_tasks --reference outputs/shared_inputs_feedback_v2/reference_data --out outputs/shared_inputs_v4
tar -czf shared_inputs_v4.tar.gz outputs/shared_inputs_v4
```

На остальных машинах: `tar -xzf shared_inputs_v4.tar.gz` из корня checkout.
Генератор отказывается перезаписывать данные. `audit.json` содержит поддержку
программ, пары операций, поля, кратчайшую длину и статистику поиска.
Новые панели проверяются интерпретатором по всем решениям и состояниям;
обычные задачи v2 копируются с проверкой хешей. Crossed train применяется
при 64 панелях и >=8 программах; иначе используется four-TARGET loss.

## Метрики и границы выводов

Основной screening-контраст — `ce_cf − ce`, local prefix policy, temperature 1,
depth3, равное среднее B/D. До final фиксируется последовательность проверок:

1. **Совместное использование START/TARGET:** нижняя граница парного 95% CI
   прироста strict crossed joint accuracy >0, прирост crossed cell accuracy >=0.
2. **Улучшение решения задач:** после первой проверки нижняя граница CI прироста
   raw Hit@1 на обычных задачах >0; Hit@8 и correct mass не снижаются.

Вторая проверка становится подтверждающей только после первой (fixed sequence,
alpha .05). Остальные интервалы описательные. Four-TARGET pair accuracy,
interaction I и энтропия сохраняются отдельно. Full-score global softmax
не объединяется с local prefix policy. Без crossed-панелей вывод о совместном
использовании START/TARGET не делается.

Сначала усредняются paired continuation seeds, затем выполняются 2 000 парных
bootstrap ресэмплов целых панелей отдельно в B/D и масках. Четыре клетки одной
панели не считаются независимыми; обычные задачи группируются по (поле, START).
Маски показаны отдельно и равным средним. CI условен на этих checkpoints и
масках; вариативность новых atomic initializations он не оценивает.
Подтверждающий вывод требует final, двух масок и seeds 0/1/2 в каждой.

Train-панели имеют единственное решение: correct-set entropy там равна нулю.
Энтропия нескольких правильных решений анализируется отдельно на обычных задачах
с `correct_count > 1`. Статистика pilot: 64 панели, 256 задач, при 2 эпохах —
128 optimizer steps и 512 предъявлений. Счётчики сохраняются в `budget.json`.

## Q2: история обучения и reward pilot

Первичная Q2 использует исторический composition seed0 и **original** mask.
Он уже видел пары mask1/mask2 и depth4. `exposure.json` и binding каждой оценки
содержат историю пар/длин; старые предъявления реконструированы по сохранённым
данным и конфигу, это не фактический старый training log. История относится
к правильным/обучающим программам; base pretraining неизвестен, а полный scorer
перебирает все допустимые префиксы.

После диагностики проверьте `gradient_diagnostic.json` и `gate.json`:

```bash
uv run python -m iclr.research start --queue-name Q2 --stage pilot \
  --gate outputs/v4_Q2_diagnostic/diagnostics/q06_original_diagnostic \
  --inputs outputs/shared_inputs_v4 --out outputs/v4_Q2_pilot --gpus 0
```

`kernel_check_failed` и `optimizer_check_failed` означают численную проблему.
`mc_inconclusive_rare_reward` означает недостаток информативных sampled-групп:
диагностика ограниченно увеличивает бюджет 128 → 512 → 2048. Все попытки
сохраняются; sampling обучающей ветки не меняется. Допуск требует >=8 смешанных
групп, >=4 информативных блоков, согласия MC/exact в пределах 4 SE, проверки
реальных Adam/Taylor шагов в FP32/BF16 и >=4 train-задач. Диагностика берёт задачи
из того же `manifest.training_panels`, что pilot. Редкий неубедительный MC
закрывает pilot, но сам по себе не доказывает ошибку ядра или отсутствие эффекта.

Для Q2 confirmation на mask1/mask2 добавьте **`--prepare-matching-sft`** к dev
и final командам confirmation ниже. Это явно добавляет два CE-SFT обучения
по 2 эпохи из atomic base, по одному для каждой маски, перед 12 reward-ветками.
Пары обеих веток внутри маски начинают с одного SFT adapter. Каждая маска
проходит свой reward-допуск. Final переиспользует эти SFT, ничего не обучая.
Альтернатива: в копии профиля указать `reward_initializations` с ключами
`mask1`/`mask2` и путями к завершённым matching-mask CE training receipts.
История исключённых пар проверяется до допуска.

## Q4: state/SIGReg pilot

```bash
uv run python -m iclr.research start --queue-name Q4 --stage pilot \
  --gate outputs/v4_Q4_diagnostic/diagnostics/q06_mask1_diagnostic \
  --inputs outputs/shared_inputs_v4 --out outputs/v4_Q4_pilot --gpus 0
```

Допуск: >=8 dev-панелей в B/D, положительная нижняя граница paired 95% CI
state-access против обычного режима и token-control. Состояние вычисляется
после выбранного моделью префикса; control получает столько же token IDs.
Четыре условия имеют одинаковую state-проекцию; оценка включает её отключение,
state reconstruction и atomic retention. SIGReg реализует проекционный
Epps–Pulley regularizer; гарантии LeJEPA на LLM не переносятся.

В новой dev-очереди можно задать `--cf-weight`, `--tau`, `--entropy-weight`,
`--state-weight`, `--sigreg-weight`, `--learning-rate`, `--epochs` и новый `--out`.
Confirmation/final наследуют выбранные значения; конфиги защищены хешами.

## Выбор пары и кривая бюджета

Команда сохраняет явное решение по dev, не выбирает метод автоматически:

```bash
uv run python -m iclr.research select --queue outputs/v4_Q3_dev/queue.json \
  --method ce_cf --reason 'Описание dev-результата и выбора метода' --out outputs/selection_v4_screen.json
```

Для перспективной **Q3/Q4 пары** до confirmation доступны отдельные dev-прогоны
с одинаковой исходной инициализацией, одной маской и seed0 на 1/2/4 эпохах:

```bash
for epochs in 1 2 4; do
  uv run python -m iclr.research start --queue-name budget --epochs "$epochs" \
    --selection outputs/selection_v4_screen.json --inputs outputs/shared_inputs_v4 \
    --out "outputs/v4_budget_${epochs}" --gpus 0
done
```

Затем выполните `select` от очереди с выбранным по dev бюджетом, например:

```bash
uv run python -m iclr.research select --queue outputs/v4_budget_2/queue.json \
  --method ce_cf --reason 'Выбор 2 эпох по dev-кривой 1/2/4' --out outputs/selection_v4.json
```

Если кривая не нужна или выбрана Q2, сохраните первичный `select` сразу
в `outputs/selection_v4.json`. Q2 использует отдельный фиксированный reward budget.

## Подтверждение и закрытая final

```bash
uv run python -m iclr.research start --queue-name confirm --selection outputs/selection_v4.json \
  --inputs outputs/shared_inputs_v4 --out outputs/v4_confirm_dev --gpus 0 1
uv run python -m iclr.research freeze --queues outputs/v4_confirm_dev/queue.json --out outputs/analysis_lock_v4.json
uv run python -m iclr.research start --queue-name confirm --selection outputs/selection_v4.json \
  --inputs outputs/shared_inputs_v4 --phase final --analysis-lock outputs/analysis_lock_v4.json \
  --trained-from outputs/v4_confirm_dev --out outputs/v4_confirm_final --gpus 0 1
```

Пара — baseline семейства и выбранный метод; две маски × три paired seed =
12 continuation-ячеек. Для одной GPU передайте `--gpus 0`.
Q2 требует описанного выше matching-mask SFT, Q2/Q4 — диагностики каждой маски.
Final использует сохранённые adapters; обучающих заданий нет.
Lock фиксирует код, иерархию метрик, данные, выбор и checkpoints.
Для Q1 заморозьте её dev-очередь и передайте `--phase final --analysis-lock ...`;
`--trained-from` не нужен.

Старые планы и locks не переписываются. Для воспроизведения v3 используйте
отдельный checkout `3d02434`, для v2 — `1566f5d`; новый анализ к ним не применяется.

## Второй домен

```bash
uv run python -m iclr.research_dsl --out outputs/shared_inputs_listdsl_v4
uv run python -m iclr.research start --queue-name external --selection outputs/selection_v4.json \
  --inputs outputs/shared_inputs_listdsl_v4 --out outputs/v4_listdsl_dev --gpus 0
```

Сначала один **общий atomic PLAN/APPLY warm-up**: 64 задачи на операцию,
2 эпохи, только поля 7/11/13; withheld 17/19 исключены. Dev-проверка требует
>=20 задач и PLAN/APPLY accuracy >=0.8 для каждой из пяти операций.
Только затем один общий adapter разветвляется в baseline/метод (seed0).
Итого три обучения. При слабых навыках продолжения остановятся; новый dev
warm-up можно явно задать через `--atomic-epochs` в новой очереди.
Reward/state метод дополнительно проходит собственную диагностику.

OP0–OP4 здесь означают инкремент, квадрат первого элемента, разворот,
сортировку и циклический сдвиг списка. Это авторский малый нелинейный DSL.
Обучение d3, оценка d3/d4, точный перебор решений. Crossed-панелей в нём нет:
результат показывает адаптацию метода к домену, не подтверждает START×TARGET
или результат на опубликованном benchmark. Его данные и final lock отдельные.

## Анализ и передача результатов

```bash
uv run python -m iclr.research_analysis \
  --runs outputs/v4_Q3_dev/evaluations/q06_mask1_ce_seed0 outputs/v4_Q3_dev/evaluations/q06_mask1_ce_cf_seed0 \
  --out outputs/v4_Q3_contrast.json
```

Для confirmation передайте все парные seeds и обе маски. Q2:
`--control sampled --treatment exact`; Q4: `--control state_ce --treatment state`.
В `summary.json`/`analysis/summary.csv` разделены policy, семьи и типы задач.

```bash
uv run python -m iclr.research send --out outputs/v4_Q3_dev
uv run python -m iclr.research verify-bundle --archive outputs/send_to_artem_exp_ИМЯ.zip
```

Папка и ZIP создаются автоматически по завершении очереди. Данные, rankings,
prefixes, exposure ledgers, бюджеты, логи и квитанции входят целиком; зависимости
(dev-очереди, selection, locks, diagnostics, SFT/warm-up receipts) — в `dependencies/`.
`DEPENDENCIES.json` связывает исходные и переносимые пути. Веса исключены.
Отдельный результат анализа сохраните в папку очереди до повторного `send`.
Для частичной очереди: `send --allow-incomplete` после обновления статуса через
`run`. ZIP остаётся локальным файлом для передачи.
