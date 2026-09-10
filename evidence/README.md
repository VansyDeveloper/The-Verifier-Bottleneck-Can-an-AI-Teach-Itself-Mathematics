# H(K) и исходные результаты

Для новых расчётов очередь сама создаёт список сравниваемых пар:

```bash
uv run python -m iclr.analyze --manifest outputs/q06/comparison_Qwen_Qwen3-0.6B.json \
  --output outputs/q06/hitk --plots
```

Получатся кривые K=1…125, парные различия и группы ошибок. Интервалы считаются по seed обучения; модели 0.6B и 1.7B анализируются отдельно.

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
