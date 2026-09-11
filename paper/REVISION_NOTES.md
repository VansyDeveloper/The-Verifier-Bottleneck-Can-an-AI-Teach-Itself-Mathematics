# Правки по трём рецензиям

Прочитаны сами [review_1.png](review_1.png) (qwGX), [review_2.png](review_2.png) (cUuo) и [review_3.png](review_3.png) (EFRP), сверены с [отправленной статьёй](submitted.pdf). Исторический PDF сохранён. Готовность кода ниже не означает, что новые научные результаты уже получены.

| Рецензент и замечание | Что есть и что осталось |
|---|---|
| qwGX: парные различия и группы ошибок | `iclr.analyze` сохраняет различия по задачам/seed и страты. Для полной H(K) основной серии нужен исходный полный архив. |
| qwGX, cUuo: внешний набор и domain shift | Открыто. Отдельный внешний benchmark не согласован; новые поля и 8B его не заменяют. |
| cUuo: replay и забывание SH1 | Очередь `replay`, оценки PLAN/APPLY по всем операциям и ошибки формата готовы. Ожидаются полные прогоны. |
| cUuo: размер модели и дополнительные seed | Пара 0.6B/8B в BF16 готова к запуску. Три seed — предварительная серия; полного запуска 8B ещё нет. |
| cUuo: статус второй серии и границы GRPO | Ниже готовые исправления статусов и формулировок. Диагностику GRPO выполняет Дима. |
| EFRP: программа без промежуточных состояний | Очередь `trace` сравнивает одинаковые задачи/witness при равном числе примеров и с отдельным контролем токенов. Ожидаются результаты; TRACE после программы не проверяет рассуждение во время вывода. |
| EFRP: H(K), точные prompts и atomic control | Оценка всех K=1…125 и сохранение исходных ответов реализованы. PLAN/APPLY/TRACE задаются исполняемым кодом, состав обучения и токены сохраняются. |
| EFRP: verifier-quality law и источник supervision | [Аудит шума](../evidence/noise_audit.json) подтвердил загрязнение labels. Ниже исправления; новый quality×exploration опыт не проведён и не добавлен в очередь. |

Готовые замены для редакции статьи:

**Figure 9, ось x:** `Fraction of incorrect composition labels, ρ`.

**Figure 9, caption:**

> Composition supervision with corrupted training labels. The horizontal axis shows the fraction ρ of the 4,000 composition examples whose witness program was replaced by an incorrect program: 0, 0.10225, 0.25225 and 0.49725. The full training mixture contains 5,000 examples. Evaluation uses the same 300 depth-three tasks over fields 11 and 17. Each corruption level uses one training seed, so the result is exploratory and has no interval over training seeds. This intervention measures robustness to label corruption; it does not measure the conditional false-accept rate of a verifier.

Связь исходных файлов и результатов подтверждена конфигами и пересчётом 251/248/227 попаданий из 300. Старые training manifests не фиксируют хеши исходного кода и обучающих данных; полная provenance поэтому не восстановлена.

**Table 11, статусы строк:**

| Study | Replacement status |
|---|---|
| E1 | Confirmatory |
| E2 | Confirmatory |
| E3 | Preregistered sampled endpoint; amended exhaustive endpoint, timing unresolved |
| E4 | Disclosed extension; hypothesis informed by an opened result |
| E5 | Post-hoc |

**Table 11, caption:**

> Status of the second-series studies. E1 and E2 were preregistered. E3 originally specified sampled pass@32; an amendment shifted emphasis to exhaustive Hit@32, but the artifacts do not establish whether this change preceded access to its results. E4 was introduced by the same amendment and used motifs selected after the main series had been opened. E4 therefore tests an informed hypothesis on independently generated data. E5 contains post-hoc controls, with one training seed per label-corruption level and three seeds for its other controls.

**Methods/introduction, источник supervision:**

> The supervised contrasts use correct programs supplied by exhaustive search and verified by the interpreter. They measure consolidation under externally supplied supervision. Discovery by the current policy is studied separately through the search and GRPO experiments.

**Discussion, границы GRPO и verifier-quality:**

> Under the tested 400-step GRPO schedule and evaluation samplers, per-candidate accuracy increased without a clear increase in held-out coverage. This result concerns that algorithm, budget and sampling setup. The label-corruption study tests the robustness of supervised learning; it does not establish a verifier-quality threshold or a universal two-dial law. The positive supervised result concerns familiar composition families, while transfer to the withheld motifs remains unconfirmed.
