# Пакет следующей итерации Q1/Q3

`BRANCH_NEXT_STEPS_RU.md` — предлагаемое ТЗ по файлам, тестам и запускам. Оно не внесено в удалённую ветку.

`independent_checks.json` — повтор компактной проверки и новая post-hoc диагностика знака I.

`check_feedback_and_diagnostics.py` — самостоятельный Python/NumPy-скрипт; запуск:

```bash
python check_feedback_and_diagnostics.py /path/to/feedback_Q1_Q3_20260923 --out checks.json
```

`SOURCE_RESULTS_RU.md` — неизменённый отчёт из пользовательского feedback-пакета; проверки сырых logits и исходных ZIP, описанные в нём, здесь заново не повторялись.

`source_comparison.json` — совпадение семи приложенных Python-фрагментов и плана с c37cfe0.
