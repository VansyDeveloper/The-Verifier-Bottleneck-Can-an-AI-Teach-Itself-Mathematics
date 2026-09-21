# Проверки версии после review 21 сентября

Новые научные GPU-эксперименты на A100 ещё не выполнены. Здесь сохранены
CPU-пересчёт прежнего dev, полный аудит новых задач, проверки очереди и
технический CUDA smoke на настоящей Qwen3-0.6B.

| Проверка | Результат / артефакт |
|---|---|
| Полный pytest | `pytest.log`: 38 тестов и 14 subtests; включает dev/final, все TARGET donors, ties, invariant alignment, offline rescore, START-bootstrap, многопроцессную очередь, resume и ZIP без весов |
| Исправленный тест final reuse | `final_reuse_pytest.log`; сначала фиктивной тестовой квитанции не хватало обязательного file hash (`pytest_initial.log`), исправлена только fixture, не проверка настоящих receipt |
| Полные данные | `full_data_audit.json`: 94 000 строк, из них 30 000 — переиспользованный atomic train и 64 000 — новые задачи; шесть datasets, 2 400 четырёхцелевых панелей |
| Witness manipulation | В `full_data_audit.json`, dataset witness: квоты, глобальная/START-специфическая эквивалентность, actual fixed/balanced за две эпохи seed0 |
| Поддержка программ и пустые страты | `design_preflight.json`: counts, program union и reliable eligibility для каждого dev/final/panel split; никаких модельных оценок |
| Команды нескольких серверов | `operator_plan_checks.json`: 10 реальных CLI-планов на полном v2; pilot/remaining покрывают 16 разных mask training cells; final не содержит обучения |
| Прежние Qwen0.6 / Qwen8B | `q06/`, `q8b/`: A-only crossfit/dev, static controls, positional controls, H(K), коэффициенты и receipts; inference не повторялся |
| Настоящая локальная CUDA | `cuda_validation.json`: Qwen0.6 BF16, RTX3050Ti 4 GiB вне sandbox, 4 примера/2 обновления balanced witness, save/reload; новый dev evaluator: 52 задачи, 8 unique TARGET panels, actual prefix probabilities, atomic и true-intermediate execution |

`analysis_lock.json` фиксирует текущий план, код, данные и прежние квитанции.
`data_protocol.json` и `input_manifest.json` идентифицируют общий TAR для
операторов. После изменения кода старый lock и готовые очереди намеренно
не принимаются как новая версия.

CPU-контроли старого dev совпали с review. `headline_controls.json` содержит
компактную таблицу и хеш прежнего training pool. Подробные `task_metrics.csv`
и исходные старые rankings остаются в исходных outputs; receipts сохраняют
их хеши. Эти компактные evidence-каталоги не являются полными копиями
исходных директорий анализа. Сырой CUDA smoke дополнительно включён без
весов в review-архив.

Все модельные значения внутри `cuda_*` — технический smoke на малом пуле,
а не оценка эффекта SFT или атомарной компетентности. Доступ к удалённому
A100 и его фактические свободные GPU проверяет оператор. Локально проверены
разделение заданий, зависимости, сохранность данных и исполнение кода;
производительность/память полных 8B запусков на A100 здесь не измерялись.

Проверки выполнены до коммита: environment может указывать родительский
`d976b30` и dirty checkout. Совпадение окончательного Python-кода проверяется
по `code_hash` в receipts, analysis lock и operator_plan_checks.
