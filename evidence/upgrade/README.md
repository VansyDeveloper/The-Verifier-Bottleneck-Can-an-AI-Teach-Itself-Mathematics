> Исторические проверки снимка d976b30. Вывод по калибровке и очередь пересмотрены после review 21 сентября; актуальные контроли и проверки — [feedback_v2](../feedback_v2/README.md).

# Проверки дополнения от 21 сентября

Эти файлы различают проверенные прежние результаты, проверку новых данных
и локальные технические запуски. Новых научных GPU-серий в этом каталоге нет.

| Артефакт | Установлено |
|---|---|
| `previous_q06.json`, `previous_q8b.json` | По шесть прежних runs, raw/metrics/summary/внутренний CSV согласованы; проверены файловые квитанции; перечислены неподтверждённые старые ячейки |
| `source_packages.json` | SHA-256 исходного ZIP 8B, исходный каталог 0.6B, хеш восстановленного старого manifest |
| `calibration_q06/`, `calibration_q8b/` | A-only alpha=1 на старом dev, результаты каждого seed, H(K), коэффициенты и исходные хеши |
| `data_protocol.json`, `finite_state_preflight.json` | Восемь новых наборов, маски, квоты, исключение исчерпанной F5/degree3 из matched train |
| `full_data_audit.json` | PASS: полный независимый перебор 72 000 новых задач; minimum depth, correct sets, все состояния, ограничения, совпадение страт и частот операций |
| `exposure_check.json` | PASS: на пяти полных matched наборах × двух seed смесь 6 250 содержит все 5 000 composition задач ровно по одному разу |
| `reward_gradient_check.json` | Точное перечисление всех IID-групп G=2/3/5; максимум ошибки градиента 5,56e-16, float64 |
| `external_model.json` | Настоящий config и tokenizer SmolLM2-1.7B на pinned revision; пять токенов операций и prefix-контракт проверены; полные веса не загружались |
| `queue_allocation.json` | Шесть блоков проверены отдельно: ацикличные зависимости, только необходимые старые веса, отсутствие повторных старых обучений |
| `shared_inputs_archive.json` | Хеш общего TAR для всех компьютеров; 139 файлов проверены, веса отсутствуют |
| `local_validation.json` | Команды/конфиги, budgets, хеши и reload реальной Qwen0.6; результаты тестов; явная граница smoke и научных runs |

В калибровочных каталогах в Git сохранена компактная выборка отчётов.
`receipt.json` относится к полному локальному анализу и перечисляет также
`task_metrics.csv`, который лежит в `outputs/upgrade_archive_v2/<model>`.
Это не самодостаточная копия всех raw старой серии; исходные файлы нужны для
повторного вычисления. Веса не входили в исходные пакеты и при аудите не
проверялись. Фактические веса проверяются новой очередью до inference.

## Повторить анализ старого dev

Из корня checkout, подставив каталог распакованного пакета:

```bash
python -m iclr.previous_runs --root outputs/q06_bf16_mb16_pb32 \
  --out outputs/audit_q06.json
python -m iclr.previous_runs --root outputs/q8b_bf16_mb16_pb32 \
  --external-csv '/path/to/results_Qwen_Qwen3-8B (2).csv' \
  --out outputs/audit_q8b.json
python -m iclr.calibration \
  --comparison outputs/q06_bf16_mb16_pb32/comparison_Qwen_Qwen3-0.6B.json \
  --out outputs/calibration_q06_new
```

Имя comparison-файла уточняется по `comparison_*.json` в пакете. Для 8B
используется такой же вызов с его comparison. Старый dev допускает crossfit
по A; новая закрытая проверка использует отдельный `calibration_A.jsonl`.
Для повторного анализа всегда выбирайте новый каталог вывода.

## Локальные проверки

Полный pytest: 36 тестов и 14 subtests прошли. Он покрывает prefix scoring,
калибровку, семантическую генерацию/аудит, witness, SGD/AdamW reward control,
закрытый evaluator, Llama init/continuation/reload, параллельную очередь,
resume и экспорт. После перемещения исходных outputs ZIP всё ещё проверяется;
искажение файла обнаруживается по SHA-256.

Настоящая Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`,
проверена вне sandbox на RTX 3050 Ti Laptop, 4 ГБ, CUDA/BF16:

* Atomic init на двух примерах, один update, merge/save/reload.
* Uniform witness, balanced witness и MML: по четыре предъявления, два update,
  промежуточный и последний eval, сохранение и повторная загрузка adapter.
* Новый prefix evaluator: 72 задачи, отдельные 5 calibration A, глубина 4,
  matched-START/wrong-TARGET и 61 строка исполнения; все проверки прошли.
* Ранее выполнен полный старый `iclr.smoke` в CUDA/BF16 на tiny Qwen.

Попытка atomic init на четырёх примерах и двух update исчерпала 4 ГБ на
втором backward (дополнительная аллокация 298 МиБ). Ошибка сохранена;
успешный сокращённый тест не подтверждает возможность полного обучения
на этой карте. Полные веса 8B и SmolLM2 здесь не запускались.

Данные полного аудита локально находятся в `outputs/upgrade_data_matched_v2`,
исходный восстановленный набор — `outputs/reference_reconstructed`.
Ранние каталоги прототипов не используются новой очередью. На удалённой
машине полный набор готовится и проверяется заново до любого inference;
для точного побайтного повтора переносится сохранённый каталог.

[План, ограничения, команды сервера и скачивания](../../plans/GOAL_TO_ICLR.md).
