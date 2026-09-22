# Статус исследований на 23 сентября 2026

| Работа | Статус | Что это позволяет сказать |
|---|---|---|
| v4 Q1, 0.6B, original, dev, baseline + 6 adapters | Выполнено оператором, исходные ZIP получены | Conditional panel и ordinary ranking изменились по-разному. Все seeds используют общую базу. |
| v4 Q3, 0.6B, mask1/seed0, initial + 4 conditions, 128 steps | Выполнено оператором, исходные ZIP получены | Основной CE+CF−CE эффект не подтверждён; существенная потеря atomic skills. |
| Архивный v4 replay | Повторен локально | Неизменённые 34 file hashes / 35 endpoint comparisons воспроизводятся. |
| Q1 nuisance/additive/pool diagnostics | Выполнено локально на CPU | Конкретная средняя program-score поправка не воспроизводит весь рост joint; численные результаты и raw subset сохранены. |
| Q3 atomic APPLY audit | Выполнено локально на CPU | Canonical labels/parser outcomes совпали; 20 raw initial ошибок сохранены. Причина ошибок модели не установлена. |
| Восстановление истории шести Q1 adapters и initial base | Не завершено: исходные training artifacts отсутствуют | Описательные сравнения допустимы; claims о невиденных парах/длине/историческом рецепте блокируются. |
| v5 replay/monitor manifests mask1 и mask2 | Сгенерированы и проверены | По 240 train-only atomic rows, баланс 5 операций × 6 известных полей × 8; exclusions против всех parent task/trajectory states. |
| v5 код после feedback1 | Исторические CPU/GPU smoke сохранены | Проверки снимка `056fd9a` — в `evidence/research_v5/VALIDATION.json`; второй feedback обнаружил непокрытые случаи resume. |
| Recovery-fix после feedback2 | Исправлены config + CLI, torn JSONL tails, `.partial` checkpoints | Новые CPU/CUDA проверки и code hash — в `evidence/research_v5/feedback2/VALIDATION.json`. Траектория сравнивается с непрерывным запуском, включая Adam и оба потока. |
| Полные R0–R3 на исходной 0.6B | Подготовлены, не запущены | Следующая научная очередь после успешной регрессии; дополнительное внешнее ревью не обязательно. Нужен фактический pinned base на машине оператора. |
| Matched CE/CF на сохранённом режиме | Код готов, gate ещё не пройден | Запускать после full dev retention и nonzero composition progress одного CE режима. |
| v5 2 masks × 3 seeds и final | Код готов, научный выбор ещё не сделан | После выбора одной пары; final закрыт для tuning. |
| Q2 / Q4 / 8B / дополнительные epochs | Не запущены в этой итерации | Q2 требует verified history и numerical gate; Q4 — положительного state gate. Остальное не является автоматическим продолжением. |

Все новые локальные GPU-прогоны — tiny random model, BF16, RTX 3050 Ti 4 GiB, ограниченные smoke budgets. Они проверяют реальные forward/backward, optimizer, save/reload, checkpoint/resume, evaluation и очередь. Они не заменяют научные R0–R3 на исходной базе и не являются независимыми replications.

Original v4/v5 plans, endpoint hierarchy, replay/monitor manifests, исходный компактный ZIP и его analyzer сохранены без изменений. После изменения code hash создаётся новая очередь; старые receipts/checkpoints не мигрируются. Полная методика и команды находятся в `RUN_V5_RU.md`, запрос reviewer — в корневом `REVIEW_REQUEST_RU.md`. Оба исходных feedback сохранены в `source/23sept_iclr_feedback1/` и `source/23sept_iclr_feedback2/`.
