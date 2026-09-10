# Источники и анализ

`paper/submitted.pdf` и три скриншота — материалы пользователя.
Здесь лежат неизменённые сводки основной серии из commit `42c70c8`.
`legacy/SOURCES.json` связывает сохранённые исходники с тем же commit.

Проверка арифметики Hit@32 и всех шести строк Table 9:

```bash
uv run python -m iclr.analyze --historical-summary evidence/PRIMARY_ANALYSIS.json \
  --output outputs/historical_check
```

Она воспроизводит 45,32% → 72,72%, разность 27,40 п.п. и exact p=0,03125.
Это проверка агрегатов. Полной H(K) основной шестизапусковой серии здесь пока нет.

Для локальной второй серии найдены 12 построчных файлов, и полные кривые уже
построены: [локальные результаты](local_series/README.md). Там три пары моделей,
другой scorer и Qwen3-0.6B-Base; это отдельное сравнение.

Для неё нужны исходные task-level metrics или frozen задачи, atomic export и
адаптеры шести пар моделей. В `archive_manifest.json` указан архив
`stage4_composition_confirmation_full.zip`; самого архива в рабочей папке нет.
Восстанавливать промежуточные K из нескольких Hit@K нельзя.

После основной новой пары очередь создаёт `comparison_Qwen_Qwen3-0.6B.json`:

```bash
uv run python -m iclr.analyze --manifest outputs/q06/comparison_Qwen_Qwen3-0.6B.json \
  --output outputs/q06/hitk --plots
```

Для своих построчных результатов формат manifest простой:

```json
[
  {"seed": 0, "arm": "atomic_control", "metrics": "control/eval/metrics.jsonl"},
  {"seed": 0, "arm": "composition", "metrics": "composition/eval/metrics.jsonl"}
]
```

Повторите пару для каждого seed. Пути считаются от manifest; новый evaluator
сам сохраняет происхождение scorer/tokenizer/prompt. Для старых данных эти поля
надо указать явно после проверки источника. `tasks` — необязательный путь к
исходным задачам для восстановления групп ошибок.

Анализ сохраняет все K, парные различия по задачам, таблицы по группам и
точечные 95% интервалы по training seeds. При одном seed интервала нет.
Модели 0.6B и 1.7B анализируются отдельно; задания должны совпадать внутри пары.

Новые final-наборы автоматически не оцениваются. Для заранее выбранного
checkpoint используется `iclr.evaluate --split final` с отдельным `--out`.
Опубликованные числа и новые screening-результаты сохраняют разный статус.
