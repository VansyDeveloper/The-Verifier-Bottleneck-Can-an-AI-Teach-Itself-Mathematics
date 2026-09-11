# Проверка результатов и H(K)

Перед анализом проверьте каждый завершённый расчёт и baseline. Например:

```bash
uv run python -m iclr.validate \
  --run outputs/q8b/runs/Qwen_Qwen3-8B_ce_trace_r20_examples_mixed_seed0 \
  --data outputs/data
```

Команда сверяет хеши файлов и состав задач, заново проверяет все программы и пересчитывает метрики из рангов и атомарных ответов. У обученного расчёта также сверяет сохранённые веса и бюджет. GPU и загрузка модели не нужны. После копирования всего `outputs/` достаточно указать новые пути. PASS означает согласованность сохранённых данных; сами модельные вероятности заново не вычисляются.

| Файлы расчёта | Для чего нужны |
|---|---|
| `config.resolved.json`, `environment.json`, `resolved_model.json` | Настройки, seed, версия кода, зависимости и фактическая модель |
| `training_metrics.jsonl`, `training_stream.jsonl`, `budget.json` | Loss и нормы градиентов, предъявленные задачи, маски и отдельные CE-суммы operation/TRACE/EOS/APPLY, фактический бюджет |
| `adapter/` или `checkpoint/`, `reload_check.json` | Повторная оценка модели и проверка сохранения весов; атомарный checkpoint тоже нужно вернуть |
| `eval/rankings.jsonl`, `eval/metrics.jsonl` | Все 125 программ для каждой задачи глубины 3: оценки, правильность, ранги и метрики |
| `eval/atomic_plan.jsonl`, `eval/atomic_apply.jsonl`, `eval/summary.json` | Предсказания операций, исходные ответы APPLY, ошибки формата и сохранение каждого навыка |
| `mid_adapter/`, `mid_eval/`, `midpoint.json` | Снимок и dev-оценка посередине continuation; при одном шаге и при atomic initialization не создаются |
| `DONE`, `TRAINED`, `FAILED`, `stdout.log`, `stderr.log` | Завершение, целостность файлов и причины ошибок |

Задачи и их manifest лежат отдельно в `outputs/data/`. `run_index.json` содержит статусы всех ячеек, включая ошибки; `results_*.csv` — метрики завершённых. `wall_seconds` и peak memory включают промежуточную оценку, когда она выполняется (`measurement_scope` в budget). Для проверки нужны исходные файлы из таблицы. После исправления анализа или оценщика сохраняйте новый результат в отдельный каталог. В `iclr.evaluate` передайте перенесённые `--base`, `--adapter`, `--data` и новый `--out`, сохранив исходные `--dtype` и `--prefix-batch`.

Если есть `TRAINED`, повтор исходной команды продолжает оценку без обучения. После сбоя внутри обучения оно начинается заново: состояние оптимизатора по шагам не сохраняется. Продолжение очереди привязано к прежним абсолютным путям; перенос поддерживается для проверки, нового анализа и новой оценки с явно заданными путями.

Для новых расчётов очередь сама создаёт список сравниваемых пар:

```bash
uv run python -m iclr.analyze --manifest outputs/q06/comparison_Qwen_Qwen3-0.6B.json \
  --output outputs/q06/hitk --plots
```

Получатся кривые K=1…125, парные различия и группы ошибок. Полосы кривых — pointwise seed-t; отдельно сохраняются crossed seed×task bootstrap для H@32, параметры и samples в `analysis.json`/`bootstrap_hit32.npz` (по умолчанию 20 000 повторов; нужен минимум 2 seed). Это проверка устойчивости endpoint, не одновременная полоса для всей кривой. Каждый размер модели анализируется отдельно, `dtype` внутри сравнения должен совпадать.

Для replay, TRACE и всех правильных программ очередь создаёт отдельные `comparison_*.json` и печатает команды с названиями сравниваемых вариантов. Например:

```bash
uv run python -m iclr.analyze --manifest outputs/q06/comparison_Qwen_Qwen3-0.6B_set_mass.json \
  --arms single_norm set_mass --output outputs/q06/set_mass_analysis --plots
```

Разность всегда равна второму варианту минус первый. После добавления seed укажите новый каталог анализа.

[Локальная вторая серия](local_series/README.md) уже содержит построчные результаты и кривые. Для основной шестизапусковой серии есть только агрегаты: 45,32% → 72,72% Hit@32, разность 27,40 п.п., p=0,03125. Их проверка вместе с Table 9:

```bash
uv run python -m iclr.analyze --historical-summary evidence/PRIMARY_ANALYSIS.json \
  --output outputs/historical_check
```

Для полной H(K) основной серии нужен `stage4_composition_confirmation_full.zip` с исходными рангами либо задачами и весами. В этой ветке его нет; локальная серия его не заменяет.
[Отправленная статья](../paper/submitted.pdf); [происхождение сохранённого кода](../legacy/SOURCES.json).
