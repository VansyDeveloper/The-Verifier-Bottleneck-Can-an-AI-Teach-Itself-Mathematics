# После feedback2: проверка исправленного recovery и запуск R0–R3

Пожалуйста, дайте явный итог **OK / NOT OK для запуска R0–R3** после recovery-fix. Если NOT OK — конкретный дефект, файл и минимальная необходимая поправка. Затем отдельно укажите **что запускать сейчас, что только после gate и что пока не запускать**. Дизайн R0–R3 и Q1 post-hoc уже одобрены во втором feedback; эта дополнительная проверка не вводит нового обязательного допуска или этапа научного проектирования.

В пакете — состояние ветки `artem_iclr` после второго feedback от 23 сентября, оба исходных feedback, frozen v4 evidence, post-hoc результаты, код, тесты и готовые данные без весов. Новые большие scientific trainings не выполнялись. Локальные изменения закоммичены; push не выполнялся. Commit и состав архива указаны во внешнем `REVIEW_ARCHIVE_MANIFEST.json`.

Исправлено:

- `resume_from` всегда удаляется из runtime config до проверки ключей; явный аргумент имеет приоритет.
- В `resume_state.pt`, защищённом hash в checkpoint receipt, сохраняются byte offsets и SHA256 трёх журналов. Проверка всех сохранённых префиксов предшествует обрезанию хвостов. Повреждённая или недостающая зафиксированная история вызывает отказ без изменения журналов.
- `latest` и проверка более поздних checkpoints игнорируют `.partial`; явное восстановление из `.partial` тоже отклоняется.
- Реальная tiny-модель сравнивается в четырёх режимах: непрерывно, с monitor, с чистым resume и после ошибки перед публикацией checkpoint плюс оборванных JSONL хвостов. Проверяются итоговые веса, Adam, курсор, счётчики и composition/replay streams. Это fault injection в реальный model/optimizer path, не системный тест отключения питания или физической durability.

Научные планы, данные, replay/monitor manifests, objectives и пороги не менялись. Старые receipts остаются историческими. После исправления строится новая очередь; checkpoints `056fd9a` без log snapshots не продолжаются новым кодом.

Порядок чтения:

1. `README.md`, `plans/RESEARCH_STATUS.md` — что уже запускалось и что только подготовлено.
2. `plans/research_v5.json`, `plans/RUN_V5_RU.md` — amendment, admission criteria и команды.
3. `iclr/research_train.py`, `tests/test_research_resume_recovery.py`, `tests/test_research_v5.py` — recovery-fix, проверки повреждения журналов и actual forward/backward/resume checks.
4. `evidence/research_v5/POSTHOC_RESULTS_RU.md` и `q1_posthoc/` — новая арифметика Q1 и raw scores для независимого CPU-пересчёта. `atomic_audit/` — raw ошибки и проверка меток.
5. `evidence/research_v5/feedback2/VALIDATION.json` — новые проверки и logs; прежний `evidence/research_v5/VALIDATION.json` относится к предыдущему code hash. `plans/source/23sept_iclr_feedback2/` — неизменённый источник замечаний. `evidence/research_v4/Q1_Q3_20260923/feedback_Q1_Q3_20260923.zip` — неизменённый прежний пакет с оригинальным analyzer и `check_primary.py`.

Проверьте, пожалуйста:

- Закрыты ли оба дефекта resume и случай незавершённой публикации checkpoint? Корректен ли отказ при повреждении зафиксированного префикса?
- Достаточны ли реальные CPU/CUDA регрессии для следующего ограниченного запуска? Какие остаются конкретные дефекты, если они есть?
- Соответствуют ли README и `RUN_V5_RU.md` принятому порядку: R0–R3 сейчас, matched CE/CF после full-dev retention/progress, без автоматического расширения?

Следующий запуск после успешной технической регрессии: R0–R3 на одной фактической 0.6B базе с ожидаемым hash, mask1/seed0, максимум 128 updates. Восстановление всей истории Q1 и новая atomic initialization не являются предварительным условием. CE/CF — после full-dev retention/progress admission; atomic skills после CF проверяются заново. Progress 0,001 означает 0,1 п.п. correct probability mass; TRAIN-only PASS подтверждает fitting. Monitor с 20 задачами на операцию не заменяет full dev. Replication/final, 8B, дополнительные epochs и SIGReg сейчас не запускать автоматически. Q2 требует verified history своей initialization и отдельного numerical gate.
