# R0–R3: один сервер или несколько отдельных A100

Для snapshot `8e37d60`, code hash:
`a3e70a7a4e344f88ce67176233d59b5fd0822b6c7f84316faccd32779883e2cc`.

Ниже — план работы, не отчёт о выполненном GPU-запуске. Команды используют существующие entrypoints проекта. Отдельной multi-host реализации в код не добавлено. Удалённое выполнение, сетевое копирование и A100 здесь не тестировались.

## 1. Только четыре условия

| Условие | LR | Replay weight | Max updates | Seed | Mask |
|---|---:|---:|---:|---:|---|
| R0 | 1e-4 | 0 | 128 | 0 | mask1 |
| R1 | 3e-5 | 0 | 128 | 0 | mask1 |
| R2 | 1e-4 | 0.25 | 128 | 0 | mask1 |
| R3 | 3e-5 | 0.25 | 128 | 0 | mask1 |

У всех одинаковые 64 панели, initialization, precision, tokenizer и composition order. Checkpoints 0/8/16/32/64/128 создаются ВНУТРИ одного run, это не шесть отдельных обучений. R2/R3 дополнительно предъявляют 128 atomic задач как PLAN и APPLY. Replay weight — коэффициент loss, не процент примеров.

Одна GPU — один worker process. Не использовать DDP для объединения четырёх независимых условий в одно обучение.

## 2. Общая подготовка

На каждой машине должны совпадать Python/uv.lock environment, код, inputs, amendments, actual base hash, dtype и prefix_batch. Начать со штатного prefix_batch=16; большая память A100 не является причиной менять panel_batch, learning rate или градиентный бюджет.

В примерах код находится по реальному абсолютному пути `/workspace/artem_iclr`, модель — `/models/atomic_06_bf16/checkpoint`. Это выбранная схема каталогов, а не существующие пути вашей машины. Для multi-host одинаковые абсолютные пути важны: planner сохраняет абсолютный путь amendment; receipts не переносить редактированием строк. Реальные каталоги или одинаковые bind mounts предпочтительнее symlink с разными resolved targets.

```bash
cd /workspace/artem_iclr
uv sync --locked
mkdir -p outputs
tar -xzf evidence/research_v4/Q1_Q3_20260923/shared_inputs_v4.tar.gz -C outputs
cp plans/research_v5_models.example.json models.local.json
```

В `models.local.json` заменить только `base` на фактический абсолютный путь. Ожидаемый hash базы оставить:
`a3ca02aa9e41ea7899c5217bf766a596d369b9ff0f92d51320d5ad3d4327d938`.

Проверить фактические веса и код до выделения длинного слота:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from iclr.common import code_hash, tree_hash
expected = 'a3e70a7a4e344f88ce67176233d59b5fd0822b6c7f84316faccd32779883e2cc'
assert code_hash() == expected, 'Другой snapshot: не смешивать эту инструкцию и очередь'
p = next(p for p in json.loads(Path('models.local.json').read_text()) if p['name'] == 'q06')
assert tree_hash(p['base']) == p['base_hash'], 'Не та база'
print('Code and base payload verified')
PY
```

На новом программном окружении полезно один раз повторить короткую техническую регрессию; не запускать полный science pilot на каждой карте для «проверки окружения»:

```bash
ICLR_SMOKE_DEVICE=cuda uv run pytest tests/test_research_v5.py tests/test_research_resume_recovery.py -q --basetemp /tmp/artem_recovery_a100
```

Эта регрессия на одной машине не доказывает bit-exact поведение между всеми моделями GPU. Не переносить незавершённый run на другое железо незаметно для provenance.

## 3. Вариант A — один сервер, 1/2/4 A100

Планировать один раз в НОВОЙ output directory:

```bash
uv run python -m iclr.research plan --queue-name stability \
  --inputs outputs/shared_inputs_v4 \
  --amendments evidence/research_v5/amendments \
  --models models.local.json --model-names q06 \
  --out outputs/v5_stability_feedback2
```

Должно получиться: `jobs=10`, `training_jobs=4`, `phase=dev`.

```bash
# Четыре карты ОДНОГО сервера:
uv run python -m iclr.research run \
  --queue outputs/v5_stability_feedback2/queue.json --gpus 0 1 2 3

# На двух картах вместо предыдущей команды:
# uv run python -m iclr.research run --queue outputs/v5_stability_feedback2/queue.json --gpus 0 1

# На одной карте:
# uv run python -m iclr.research run --queue outputs/v5_stability_feedback2/queue.json --gpus 0
```

Это альтернативы, не три команды для последовательного запуска. GPU indices должны соответствовать видимым устройствам процесса. Не запускать второй dispatcher поверх уже работающего.

Граф зависимостей:

```text
import frozen inputs
         |
initial full dev ONCE
         |
   +-----+------+------+
   |     |      |      |
  R0    R1     R2     R3        independent train + built-in monitors
   |     |      |      |
 eval0 eval1  eval2  eval3       full dev for final saved adapters
   +-----+------+------+
    choose a candidate from retention / TRAIN / A
         |
additional full dev only when selecting an intermediate checkpoint
         |
full-dev retention + progress admission
```

В начале работает только initial evaluation; остальные GPU могут временно простаивать — таков текущий DAG. На двух GPU порядок не строго «все train, затем все eval»: готовые evaluations могут занять слот раньше R2/R3. Это не лишние runs.

При прерывании сначала убедиться, что прежний worker действительно завершился. Повторить штатную команду с `--retry-failed`. Живой процесс не дублировать. Code/config/data работающей очереди не менять.

## 4. Вариант B — четыре разных сервера с одной A100 на каждом

**Не выполнять полную `research run` на каждом сервере.** Так будут посчитаны четыре копии всей серии. Общий `.queue.lock` также не превращает native runner в распределённую очередь.

Используем одну master queue/configs и ручное назначение ровно одного условия на worker. На всех хостах одинаковая структура путей. Один coordinator собирает результаты; остальные не изменяют общий status.json.

| Host / GPU | Роль | Train job | Eval job |
|---|---|---|---|
| A100-0 | common initial, затем R0 | q06_mask1_R0_seed0_train | q06_mask1_R0_seed0_eval |
| A100-1 | R1 | q06_mask1_R1_seed0_train | q06_mask1_R1_seed0_eval |
| A100-2 | R2 | q06_mask1_R2_seed0_train | q06_mask1_R2_seed0_eval |
| A100-3 | R3 | q06_mask1_R3_seed0_train | q06_mask1_R3_seed0_eval |

На двух отдельных серверах: A100-0 выполняет R0, затем R2; A100-1 — R1, затем R3. Модель каждого условия всегда начинается с общей initialization, а не с результата предыдущего условия.

### B1. Coordinator создаёт очередь и считает common jobs

Планирование — ровно команда из раздела 3. Пока НЕ запускать `research run`.

На coordinator, из корня проекта, выполнить ровно две общие задачи из сгенерированной очереди:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
uv run python - <<'PY'
import json, subprocess, sys
from pathlib import Path
from iclr.common import ROOT, code_hash, file_hash
from iclr.upgrade import verify_job
q = json.loads(Path('outputs/v5_stability_feedback2/queue.json').read_text())
assert q['source_code_hash'] == code_hash()
for name in ('inputs', 'q06_mask1_initial'):
    j = next(j for j in q['jobs'] if j['id'] == name)
    if j.get('config_path'):
        assert file_hash(j['config_path']) == j['config_sha256']
    subprocess.run([sys.executable, *j['argv']], cwd=ROOT, check=True)
    print(name, verify_job(j), flush=True)
PY
```

После этого распространить на workers точные копии code/amendments, models.local.json, inputs и каталога `outputs/v5_stability_feedback2`, включая `data`, `configs`, `queue.json` и `evaluations/q06_mask1_initial`. Каждая копия лежит в одинаковом пути. Веса общей базы передаются отдельно. Код и manifests сверить по hash.

Если используются независимые диски, status.json в копиях остаётся вспомогательной master-записью: на workers native dispatcher НЕ запускается. Если используется общий диск, каждому worker разрешена запись только в каталог своего условия и свой лог.

### B2. Worker считает только назначенное условие

Например, A100-1:

```bash
cd /workspace/artem_iclr
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
ARM=R1  # На других хостах ровно R0, R2 или R3 согласно таблице.
Q=outputs/v5_stability_feedback2
mkdir -p "$Q/worker_logs"

# До worker старта: не менять config и проверить его hash по master queue.
ARM="$ARM" uv run python - <<'PY'
import json, os
from pathlib import Path
from iclr.common import code_hash, file_hash
from iclr.upgrade import verify_job
q = json.loads(Path('outputs/v5_stability_feedback2/queue.json').read_text())
assert q['source_code_hash'] == code_hash()
assert os.environ['ARM'] in ('R0','R1','R2','R3')
for name in ('inputs', 'q06_mask1_initial'):
    verify_job(next(j for j in q['jobs'] if j['id'] == name))
for suffix in ('train','eval'):
    j = next(j for j in q['jobs'] if j['id'] == f'q06_mask1_{os.environ["ARM"]}_seed0_{suffix}')
    assert file_hash(j['config_path']) == j['config_sha256']
print('Assignment and prerequisites verified')
PY

uv run python -m iclr.research_train \
  --config "$Q/configs/q06_mask1_${ARM}_seed0_train.json" \
  2>&1 | tee -a "$Q/worker_logs/${ARM}_train.log"

uv run python -m iclr.research_evaluate \
  --config "$Q/configs/q06_mask1_${ARM}_seed0_eval.json" \
  2>&1 | tee -a "$Q/worker_logs/${ARM}_eval.log"
```

Каждый ARM запускается только на одном worker. Прямой worker entrypoint не защищает от двух операторов, одновременно выполняющих один config: назначение владельца обязательно. Resume применим только после завершения/аварийного прекращения предыдущего процесса. Training config уже содержит `resume_from: latest`.

Внешние stdout-логи держать в `worker_logs`, а не внутри sealed training/evaluation directory: receipts включают файлы результата.

### B3. Собрать outputs, не переписывая receipts

С каждого worker вернуть на coordinator:

```text
outputs/v5_stability_feedback2/training/q06_mask1_Rn_seed0/   whole tree
outputs/v5_stability_feedback2/evaluations/q06_mask1_Rn_seed0/ whole tree
outputs/v5_stability_feedback2/worker_logs/Rn_*.log
```

Не копировать чужой status.json поверх master. Не переименовывать Rn и не править DONE. Не использовать `--delete` для всего общего results directory.

**Нужны веса адаптеров и промежуточных checkpoints.** Weight-free feedback ZIP достаточен для части арифметики, но не для full-dev выбранного шага или resume. `resume_state.pt` и committed logs хранить вместе. Экспорт без весов — отдельная операция после сохранения полного run store.

До любых действий native dispatcher проверить, что все 10 results действительно собраны:

```bash
cd /workspace/artem_iclr
uv run python - <<'PY'
import json
from pathlib import Path
from iclr.common import code_hash, file_hash
from iclr.upgrade import verify_job
p = Path('outputs/v5_stability_feedback2/queue.json')
q = json.loads(p.read_text())
assert q['source_code_hash'] == code_hash()
state = json.loads(p.with_name('status.json').read_text())
assert state['queue_sha256'] == file_hash(p)
for j in q['jobs']:
    if j.get('config_path'):
        assert file_hash(j['config_path']) == j['config_sha256']
    print(j['id'], verify_job(j))
print('ALL 10 RESULTS VERIFIED; no missing training/evaluation outputs')
PY
```

Только после успеха всей проверки можно один раз запустить native runner на coordinator для согласования master status и обычного export:

```bash
uv run python -m iclr.research run \
  --queue outputs/v5_stability_feedback2/queue.json --gpus 0
```

Почему это не повтор науки: `start_run` возвращает completed до загрузки модели при совпадающих config/binding/receipt/payload. Worker modules будут вызваны, но full completed результаты пропустятся. Если проверка выше нашла незавершённый job — НЕ использовать эту команду как бездумный «сборщик»: она вправе запустить недостающую работу. Сначала установить её владельца и закончить/перенести run осознанно.

Сборка/verify здесь проверены по коду, не сетевым end-to-end запуском. Для следующей большой серии удобнее отдельный тестированный `run-job/collect-existing` interface, но он не обязателен для четырёх вручную назначенных условий.

## 5. Как не умножить стоимость после R0–R3

Исходная очередь уже выполняет 1 initial + 4 final full dev. Дополнительный full dev не нужен для шага 128.

Для шага 8/16/32/64: сначала выбрать кандидата по monitor retention/TRAIN/A, затем full dev. Не делать full dev всех 24 checkpoint «на всякий случай». B/D не использовать для выбора устойчивого recipe.

Пример для R2, step32 — это пример, НЕ заранее выбранный победитель:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m iclr.research_stability evaluate-step \
  --training outputs/v5_stability_feedback2/training/q06_mask1_R2_seed0 \
  --step 32 --out outputs/v5_R2_step32_full

uv run python -m iclr.research_stability select \
  --queue outputs/v5_stability_feedback2/queue.json --condition R2 --step 32 \
  --full-evaluation outputs/v5_R2_step32_full \
  --out outputs/v5_stability_selection.json
```

Текущий gate: <=5 п.п. падения на каждой PLAN и исходно сильных APPLY; correct mass gain >=0.001 на фиксированном TRAIN probe ИЛИ полном dev A. Monitor с 20 задачами/op не устанавливает full-dev PASS. TRAIN-only PASS — fitting, не перенос.

После PASS — только CE и CE+CF на одном выбранном recipe. На двух GPU: одна arm на каждую. Третья/четвёртая карта не обязана быть занята новой неподготовленной гипотезой. Отдельная intermediate evaluation или независимая уже допущенная диагностика возможны, но не новый обязательный sweep.

Если PASS нет: не масштабировать CF, не снижать порог и не добавлять эпохи задним числом; разбирать имеющуюся динамику.

## 6. Что сейчас не запускать

Не повторять Q1 post-hoc на GPU — scores и CPU-разбор уже сохранены.
Не запускать полный старый Q3-факторный пилот заново до устойчивости.
Не запускать автоматически entropy sweeps, 8B, 256 steps, SIGReg или confirmation.
Q2 возможна отдельно только с verified initialization history и своим numerical gate. Не расходовать четыре свободные GPU на четыре неподтверждённых ответвления только ради загрузки устройств.

## 7. Время и учёт ресурсов

В архиве нет production throughput/VRAM measurements для A100. Число шагов не равно полному времени: monitor, full ranking, atomic generation, checkpoint I/O и hash verification имеют отдельную стоимость. Не обещать длительность по tiny smoke.

Сохранять фактические host/GPU/dtype/environment каждого run, stdout, budget и время всех попыток. Текущее elapsed в trainer включает monitor и начинается заново после resume; суммарные GPU-hours нужно учитывать отдельно. Нельзя незаметно считать переезд на другое устройство тем же проверенным bit-exact экспериментом.
