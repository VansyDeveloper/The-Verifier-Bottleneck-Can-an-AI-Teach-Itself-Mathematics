# Q1: что объясняет средняя поправка к scores

Пересчёт выполнен на CPU по исходным Q1 scores трёх paired seeds. Для каждого seed и каждой policy взяты одни и те же первые 64 ordinary dev A задачи, отсортированные по task_id, только на известных полях. Это новый post-hoc reference из разрешённого A-условия. B/D и новые поля не использовались для оценки поправки. Коэффициент alpha=1; он не подбирался.

Для каждой программы вычислено `delta_b = mean_reference(score_composition − score_control)`. Таблица ниже — local policy, среднее по seeds и равным B/D; проценты, кроме последнего столбца со средней вероятностной массой.

| Scores | Strict joint | Four-target pair | Hit@8 | Hit@32 | Correct mass |
|---|---:|---:|---:|---:|---:|
| Control | 9,375% | 70,667% | 7,167% | 25,583% | 0,0009477 |
| Composition | 42,708% | 82,250% | 0,833% | 9,583% | 0,0027539 |
| Control + delta_b | 11,458% | 70,667% | 0,750% | 19,250% | 0,0024926 |
| Composition − delta_b | 9,375% | 82,250% | 0,583% | 29,083% | 0,0000501 |
| Только delta_b | 0% | 50% | 0% | 13,333% | 0,0013312 |

Hit@1 равен нулю во всех пяти вариантах. После каждой модификации scores вероятности заново нормированы по всем 125 программам. Tie-breaking для Hit@K совпадает с v4: score, затем лексикографический порядок программы. Full-vocabulary/global-leaf policy посчитана отдельно в JSON и не объединялась с local.

Средняя поправка заметно меняет массу правильных программ и ranking, но её перенос на control не воспроизводит observed joint 42,71%. Удаление этой же поправки из composition снижает joint до 9,38%. Это асимметричный результат: данная аддитивная модель связана с эффектом, но одной её недостаточно. Из него нельзя вычислять «долю причинного эффекта prior» или заключать о существовании отдельного нейронного модуля.

Аддитивная program-поправка в точности сохраняет I каждой crossed-панели и four-target cycles. У исходных моделей условная структура отличается: доля положительных I выросла с 70,83% до 88,54%; 21 panel-seed запись перешла из неположительного I в положительное, четыре — обратно. Это 32 общие панели × три seed, не 96 независимых задач. Среднее I при этом снизилось с 8,2095 до 6,9981; знак среднего изменения и доля положительных панелей отвечают на разные вопросы. Новый post-hoc показатель не заменяет первичный joint.

В `crossed_decomposition.jsonl.gz` сохранены все пять вариантов: 4×2 score matrices, четыре signed margins, minimum margin, a/b/c/h, I и scalar-offset gap. Положительный gap означает существование исправляющего offset для конкретной панели; это диагностическое свойство, не разрешённая коррекция с использованием ответа.

## Где выросла correct mass

Для каждой ordinary B/D задачи `m = pi * r`, где pi — масса пула программ с исключёнными парами, r — условная масса правильной программы внутри него. Использовано точное симметричное разложение:

`delta_m = (pi_t − pi_c)(r_t + r_c)/2 + (r_t − r_c)(pi_t + pi_c)/2`.

Усреднение выполнено **после** разложения по задачам. Local policy: общий прирост 0,00180615, между пулами — 0,00114649 (63,5%), внутри пула — 0,00065966 (36,5%). Это алгебраические слагаемые, не причинные вклады. Все paired строки сохранены в `pool_decomposition.jsonl.gz`.

Exact IID sampled success при K=32 вырос с 2,380% до 7,644%, тогда как ranked Hit@32 снизился. Эти search contracts сохранены отдельно. Ни sampled success, ни restricted candidate pool не становятся новым primary inference методом.

## Atomic audit и границы

Для всех пяти Q3 dev evaluations повторены canonical labels и parsing raw APPLY completions. Расхождений с сохранёнными semantic outcomes нет. В `atomic_audit/` — operation × field × degree table и 20 initial ошибок SH1/SC2 с prompt, expected, raw и parsed ответом. Код canonical prompts, interpreter и tokenization validation побайтно совпадает с архивной версией; исходные base-training prompts/tokenizer/receipt не предоставлены. Поэтому причина отсутствия SH1 APPLY не установлена.

Сохранять сильные PLAN/REV/AC1/AX1 навыки и обучать отсутствующие SH1/SC2 APPLY — разные задачи. На этих данных нельзя писать, что initial уже освоил все примитивы. История Q1 adapters и task-training история базы остаются неполными; details — `initialization_provenance.json`.

## Повторить числа

```bash
uv run python analysis/q1_prior_residual.py --results evidence/research_v5/q1_posthoc --out outputs/q1_posthoc_check
```

`source_scores.jsonl.gz` — lossless subset исходных scores для reference и B/D evaluation; `input_manifest.json` содержит SHA256, parent manifest и source bindings. Внутренняя арифметика полностью воспроизводится из пакета. В исходном запуске анализатор проверил оригинальные evaluation receipts; повтор их полного file audit требует первичных ZIP. Новый анализ не является новым inference или training.
