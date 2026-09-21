# Эксперименты Артёма к ICLR — после review 21 сентября

**Прежние Qwen3-0.6B и Qwen3-8B уже обучены. Старую очередь из
`artem_iclr` и прежние atomic/SFT обучения повторять не нужно.**
Сейчас приоритет — проверить, отличает ли модель TARGET, когда общий
рейтинг программ уже не может объяснить результат.

| Уже выполненная серия | Зачем она нужна | Что сохранено |
|---|---|---|
| Qwen3-0.6B | Основной парный контраст atomic control / composition | Обе ветки × seed 0/1/2, midpoint/final, raw, конфиги и квитанции |
| Qwen3-8B | Тот же контраст при большем размере Qwen | Обе ветки × seed 0/1/2; согласованные raw/summary внутри исходного ZIP |

Высокий Hit@32 после калибровки сам по себе **не подтверждает восстановление
композиционного умения**. На старом B у SFT full-калибровка даёт 56,00% / 58,33%,
а рейтинг `−bias-only`, не читающий задачу, — 66,50% / 64,50% для 0.6B / 8B.
Для unseen-first контроль на B составляет около 69,57%. Это результат
ранжирования без возвращения, не IID pass@32. Новая основная метрика —
различение четырёх TARGET при одном START: парный контраст сокращает общий
приоритет программы и общий сдвиг score промпта.

[Ответ на review и границы выводов](FEEDBACK_21SEPT_RU.md).
[Зафиксированный план анализа](upgrade_analysis_plan.json).
Исходный review сохранён [без изменений](source/21sept_iclr_feedback/short_feedback.md).
Версия `d976b30` остаётся в истории; её наборы данных не перезаписываются.

## Подготовить сервер

Нужны Linux, Git, uv, Python 3.11 и CUDA. Целевые удалённые серверы — с A100;
проверьте фактическую карту и свободную память. Адрес/SSH сервера не задан
в репозитории: используется сервер оператора с сохранёнными весами.

```bash
git switch artem_iclr
git pull --ff-only
uv sync --locked
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
uv run python -m pytest -q
```

| Размер | Прежний atomic checkpoint | Прежние SFT/atomic-control adapter |
|---|---|---|
| 0.6B | `outputs/atomic_06_bf16/checkpoint` | Шесть `*/adapter` в `outputs/q06_bf16_mb16_pb32` |
| 8B | `outputs/atomic_8b_bf16/checkpoint` | Шесть `*/adapter` в `outputs/q8b_bf16_mb16_pb32` |

Исторический корень — `/workspace/verifier-bottleneck`; это не подтверждение
текущего размещения. При других путях скопируйте
[профиль](upgrade_models.json), измените `base`/`previous` и добавьте
`--models path/to/models.json` к командам. ZIP `no_models` весов не содержит.
Хеши обязательны: другой checkpoint не будет молча принят за прежний.
Блокам 02/04 нужен только atomic 0.6B; прежние adapters им не нужны.

## Один общий набор данных на все компьютеры

Используйте переданный вместе с review `shared_inputs_feedback_v2.tar.gz`.
Распакуйте его **из корня checkout** на каждом сервере:

```bash
tar -xzf shared_inputs_feedback_v2.tar.gz
```

Получится `outputs/shared_inputs_feedback_v2/{reference_data,new_tasks,data_audit}`.
Если нужно подготовить набор самостоятельно, сделайте это **один раз**:

```bash
mkdir -p outputs/shared_inputs_feedback_v2
cp -a outputs/data outputs/shared_inputs_feedback_v2/reference_data
uv run python -m iclr.upgrade_data \
  --reference outputs/shared_inputs_feedback_v2/reference_data \
  --out outputs/shared_inputs_feedback_v2/new_tasks
uv run python -m iclr.upgrade_audit \
  --reference outputs/shared_inputs_feedback_v2/reference_data \
  --data outputs/shared_inputs_feedback_v2/new_tasks \
  --out outputs/shared_inputs_feedback_v2/data_audit
tar -czf shared_inputs_feedback_v2.tar.gz outputs/shared_inputs_feedback_v2
```

При отсутствии прежнего `outputs/data` восстановите его командой
`uv run python -m iclr.data --out outputs/data`. Ожидаемый SHA-256 manifest:
`8a1a9cbc63fa4d9fc5bbf4b82db8d87190fff44f3a88ba50aa550111e161a866`.
Это генерация задач, не повтор обучения. Не запускайте независимую генерацию
новых задач на каждой машине: MILP с одним seed не гарантирует одинаковые
байты в разных окружениях. Каждый сервер получает один и тот же TAR.
Новый код отклоняет прежний протокол v1.

## Что запускать сейчас

| Блок | Команда/этап | Новых обучений |
|---|---|---:|
| 01 | Старые checkpoint 0.6B: новые dev-панели, затем тот же анализ 8B | **0** |
| 02 pilot | mask1/2 × atomic_control/composition × seed0, после анализа 01 | **4** |
| 02 remaining | mask3/4 seed0 и все четыре маски seed1, после интерпретации pilot | **12** |
| 04 pilot | Новый witness-пул, fixed/balanced × seed0 | **2** |
| 03 | Старые triple/position не различают условный выбор; запуск отключён до нового дизайна | **0** |
| 05 | Отложен до гипотезы о градиенте и численной проверки; автоматический pilot → full удалён | **0** |
| 06 | Условная репликация: центральная пара и fixed/balanced на Qwen/SmolLM2, seed0, одна новая Smol atomic init | **9**, только явно |

**Команда по умолчанию запускает только 01 на 0.6B, в фазе dev.**
Очередь из 39/60 обучений больше не является планом запуска.
Сначала выполните на сервере с нужными старыми весами:

```bash
uv run python -m iclr.upgrade start \
  --inputs outputs/shared_inputs_feedback_v2 \
  --model-names q06 --out outputs/01_old_q06_dev --gpus 0 1
```

После анализа 0.6B тот же набор и правила для 8B:

```bash
uv run python -m iclr.upgrade start \
  --inputs outputs/shared_inputs_feedback_v2 \
  --model-names q8b --out outputs/01_old_q8b_dev --gpus 0 1
```

Один процесс занимает одну GPU. Укажите реально свободные индексы; при одной
A100 используйте `--gpus 0`. Диспетчер удобно держать в tmux. По умолчанию
BF16, prefix batch 32; при нехватке памяти задайте `--prefix-batch 8` в новой
папке очереди. Чтобы проверить состав без запуска, замените `start` на `plan`
и уберите `--gpus`. `queue.json` содержит все задания и зависимости.

## Пилот на отдельных серверах A100

После содержательного разбора блока 01 можно одновременно поставить две
маски на разные серверы. В каждой очереди остаются обе сравниваемые ветки:

```bash
# Сервер A: mask1, две ветки, seed0.
uv run python -m iclr.upgrade start \
  --experiments 02_new_pair_masks --stage pilot --masks mask1 \
  --inputs outputs/shared_inputs_feedback_v2 \
  --out outputs/02_pilot_mask1_seed0_dev --gpus 0

# Сервер B: mask2, две ветки, seed0.
uv run python -m iclr.upgrade start \
  --experiments 02_new_pair_masks --stage pilot --masks mask2 \
  --inputs outputs/shared_inputs_feedback_v2 \
  --out outputs/02_pilot_mask2_seed0_dev --gpus 0
```

Без `--masks` pilot включает обе маски. После интерпретации pilot:

```bash
# Сервер A: только недостающие mask3/4, seed0.
uv run python -m iclr.upgrade start \
  --experiments 02_new_pair_masks --stage remaining --seeds 0 \
  --inputs outputs/shared_inputs_feedback_v2 \
  --out outputs/02_remaining_seed0_dev --gpus 0 1

# Сервер B: все четыре маски, seed1.
uv run python -m iclr.upgrade start \
  --experiments 02_new_pair_masks --stage remaining --seeds 1 \
  --inputs outputs/shared_inputs_feedback_v2 \
  --out outputs/02_remaining_seed1_dev --gpus 0 1
```

Pilot и remaining не повторяют обучающие ячейки. Не назначайте одну ячейку
`(модель, dataset, arm, seed, phase)` двум машинам и не пишите с разных
машин в одну папку очереди. Повтор baseline при отдельной очереди нужен
для привязки к её данным; это только оценка, без обучения.

Исправленное вмешательство после проверки основного контраста:

```bash
uv run python -m iclr.upgrade start \
  --experiments 04_correct_program_choice --stage pilot \
  --inputs outputs/shared_inputs_feedback_v2 \
  --out outputs/04_fixed_vs_balanced_seed0_dev --gpus 0 1
```

CPU-аудит проверяет весь witness-пул и фактический эффект выбора меток за
две эпохи до GPU. Перед новым обучением очередь оценивает atomic baseline
на тех же данных. Если атомарное исполнение не проходит критерий, результаты
остаются диагностикой распределения программ; утверждать композицию уже
надёжно освоенных навыков нельзя.

Блок 06 запускается отдельно и только после выбора центрального эффекта и
вмешательства: `--experiments 06_other_model_family --stage complete`.
Держите его на одном сервере/в одной очереди с несколькими GPU: используется
общий новый Smol atomic checkpoint. Это 9 обучений, не прежняя широкая сетка.

## Final и продолжение очереди

Dev-команды не открывают final. До окончательной оценки один раз сохраните
замок анализа и скопируйте его с общими данными на другие компьютеры:

```bash
uv run python -m iclr.upgrade freeze-analysis \
  --inputs outputs/shared_inputs_feedback_v2 \
  --out outputs/analysis_lock_feedback_v2.json
```

Он фиксирует scorer, contrast и правило checkpoint через хеш плана, код,
данные и квитанции старых моделей. Поправки обучаются только на отдельном A,
alpha=1. Итоговая проверка старой 0.6B:

```bash
uv run python -m iclr.upgrade start \
  --inputs outputs/shared_inputs_feedback_v2 --model-names q06 \
  --phase final --analysis-lock outputs/analysis_lock_feedback_v2.json \
  --out outputs/01_old_q06_final --gpus 0 1
```

Для final новых обучений повторите селекторы соответствующей dev-очереди,
добавьте `--phase final --analysis-lock outputs/analysis_lock_feedback_v2.json
--trained-from outputs/ИМЯ_DEV_ОЧЕРЕДИ` и укажите новый `--out`.
**Final использует готовый финальный adapter, не запускает обучение повторно.**
Midpoint не выбирается по лучшему B/D. После изменения кода/плана нужен новый
протокол; нельзя подменять файлы в готовой очереди.

Для возобновления используйте ту же версию кода:

```bash
uv run python -m iclr.upgrade run \
  --queue outputs/01_old_q06_dev/queue.json --gpus 0 1
```

Готовые результаты проверяются по хешам, живые процессы не дублируются.
Причина сбоя — в `status.json` и `logs/<job>.log`; после исправления добавьте
`--retry-failed`. Без TRAINED прерванное обучение начинается с исходных весов;
после TRAINED продолжается оценка. Частичные данные сохраняются, новый
набор создаётся в новом каталоге.

## Что отправить Артёму

После `start`/`run` автоматически создаются:

```text
outputs/send_to_artem_exp_ИМЯ_ОЧЕРЕДИ_ДАТА_ВРЕМЯ/
outputs/send_to_artem_exp_ИМЯ_ОЧЕРЕДИ_ДАТА_ВРЕМЯ.zip
```

**Отправьте ZIP целиком; веса исключены.** В нём сводки, target alignment,
все score и префиксы, три подмены TARGET, static controls, исходные задачи,
результаты исполнения, конфиги, бюджеты, логи, квитанции и код с протоколом.
Начните с `START_HERE_RU.md`, `results/analysis/runs.csv`, затем
`results/experiments/<block>/evaluations/<run>/target_alignment.json`.
Не объединяйте разные data hash, фазы, маски или scorer как независимые повторы.

```bash
uv run python -m iclr.upgrade send --out outputs/01_old_q06_dev
python -m iclr.upgrade verify-bundle --archive send_to_artem_exp_....zip
```

Для остановленной неполной очереди `send --allow-incomplete` делает явно
помеченную partial-посылку. Проверка ZIP работает без весов и CUDA.
Новые контроли из сохранённых rankings можно пересчитать на CPU:

```bash
uv run python -m iclr.upgrade_analysis rescore \
  --source outputs/01_old_q06_dev/experiments/01_recheck_existing_models/evaluations/q06_previous_composition_seed0 \
  --data outputs/01_old_q06_dev/data/original --out outputs/reanalysis_q06_seed0
```

`iclr.upgrade_analysis compare --alignment <atomic JSON> <composition JSON>
--out <contrast.json>` считает парный START-bootstrap основного контраста;
можно передать все три пары seed одной модели/маски/фазы. Для witness
добавьте `--treatment balanced --control fixed`. На d3 требуется 31 префикс,
на d4 — 156; новые аналитические контроли не требуют повторного inference.

[Проверки этой версии](../evidence/feedback_v2/README.md).
[Архивный план до review](GOAL_TO_ICLR.md) сохранён для истории.
