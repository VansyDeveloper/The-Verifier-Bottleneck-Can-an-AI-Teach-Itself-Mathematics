# Повторный feedback: корректность кода и порядок запусков

Пожалуйста, дайте явный итог **OK / NOT OK для запуска R0–R3**. Если NOT OK — конкретный дефект, файл и минимальная необходимая поправка. Затем отдельно укажите **что запускать сейчас, что только после gate и что пока не запускать**.

В пакете — состояние ветки `artem_iclr` после исправлений feedback от 23 сентября, исходный feedback, frozen v4 evidence, новые post-hoc результаты, код, тесты и готовые данные без весов. Новые большие scientific trainings не выполнялись. Локальные изменения закоммичены; push не выполнялся. Commit и состав архива указаны во внешнем `REVIEW_ARCHIVE_MANIFEST.json`.

Порядок чтения:

1. `README.md`, `plans/RESEARCH_STATUS.md` — что уже запускалось и что только подготовлено.
2. `plans/research_v5.json`, `plans/RUN_V5_RU.md` — amendment, admission criteria и команды.
3. `iclr/research_train.py`, `research_evaluate.py`, `research_io.py`, `research_data.py`, `research_stability.py`, `research.py` — реализация и bindings; `tests/test_research_v5.py` — actual forward/backward/resume checks.
4. `evidence/research_v5/POSTHOC_RESULTS_RU.md` и `q1_posthoc/` — новая арифметика Q1 и raw scores для независимого CPU-пересчёта. `atomic_audit/` — raw ошибки и проверка меток.
5. `evidence/research_v5/VALIDATION.json` — выполненные проверки кода. `evidence/research_v4/Q1_Q3_20260923/feedback_Q1_Q3_20260923.zip` — неизменённый прежний пакет с оригинальным analyzer и `check_primary.py`.

Проверьте, пожалуйста:

- Сохраняет ли checkpoint/resume фактическую траекторию, включая Adam/RNG/cursor, и привязана ли evaluation к своему шагу? Достаточны ли tiny CPU/GPU regression checks?
- Корректны ли train-only replay, исключение совпадений состояний/задач, PLAN/APPLY mean-token CE с EOS, раздельный budget? Действительно ли matched arms получают один режим?
- Подходит ли bounded R0–R3 и критерий retention ≤5 п.п. плюс gain correct mass ≥0,001 на train/A? Достаточен ли monitor для решения о дополнительных full dev оценках? Эти пороги — инженерное предложение, не установленный оптимум.
- Корректны ли Q1 additive diagnostic, reference из 64 разрешённых dev A, нормировка вероятностей, crossed nuisance decomposition и точное разложение массы по задачам? Нет ли завышенных выводов? Первичные v4 endpoints не заменены.
- Достаточно ли ограничены claims при неизвестной истории шести Q1 adapters и declared-only базе? Какие реальные receipts нужно запросить у оператора до Q2 или stronger transfer claims?
- Есть ли причина сейчас готовить новую общую atomic initialization, или сначала достаточно проверить сохранение уже сильных операций и отдельно разобрать SH1/SC2 APPLY?

Предлагаемое решение: после OK запустить только R0–R3 на 0.6B, mask1/seed0, максимум 128 updates; параллельно восстановить происхождение weights. CE/CF — после retention/progress admission. Replication/final, 8B, дополнительные epochs и SIGReg сейчас не запускать автоматически. Если вы рекомендуете другой порядок, укажите конкретный минимальный запуск и критерий остановки.
