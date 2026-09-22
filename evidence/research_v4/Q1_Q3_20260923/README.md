# Неизменяемое dev-evidence от 22–23 сентября

`feedback_Q1_Q3_20260923.zip` сохранён побайтно, вместе с исходной SHA256. Внутри находятся оригинальный CPU analyzer, `check_primary.py`, primary observations, таблицы, полный русский разбор и снимок v4 protocol. Проверка по-прежнему воспроизводит 34 file hashes и 35 endpoint comparisons.

`shared_inputs_v4.tar.gz` содержит побайтную копию `results/data` первичного Q3 ZIP, упакованную как `shared_inputs_v4/`. Сохраняются исходные manifests, protocol и audit. Новые replay данные лежат отдельно в `evidence/research_v5/amendments/`; v4 JSONL не менялись.

`ARCHIVES.json` содержит hashes, происхождение и ссылки по SHA256 на два больших первичных ZIP. Они не дублируются в компактном review-архиве. Старые результаты остаются dev-evidence: Q3 — одна mask/seed, победитель не подтверждён; для Q1 не восстановлена история шести фактических adapters.

Повторный feedback сохранён в `plans/source/23sept_iclr_feedback1/`. Новая интерпретация и новые запуски относятся к v5, а не к пересмотренному задним числом primary v4.
