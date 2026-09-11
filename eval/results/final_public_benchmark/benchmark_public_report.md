# Final public benchmark report

Official 100-question run of RAG, GraphRAG, and Agentic GraphRAG.
Old partial 86-record results were not mixed into this file.

## 1. Overall comparison

| Pipeline | n | Accuracy | Retrieval success | Completeness | Avg latency (s) | Median latency (s) | Avg context tokens | Avg LLM total tokens | Token efficiency /1k | Avg steps | Avg tools | Failures | Timeouts | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag | 100 | 0.1000 | 0.4400 | 0.2557 | 250.00 | 241.70 | 987.4 | 1972.9 | 0.1013 | 0.00 | 0.00 | 90 | 0 | 0 |
| graphrag | 100 | 0.6000 | 0.9500 | 0.8524 | 389.80 | 426.71 | 1282.7 | 2782.4 | 0.4678 | 0.00 | 0.00 | 40 | 3 | 3 |
| agentic | 100 | 0.9200 | 0.9700 | 0.8650 | 18.43 | 0.42 | 1510.2 | 521.6 | 0.6092 | 7.93 | 7.93 | 8 | 1 | 1 |

## 2. Qtype breakdown

### lookup (n=19)

| Pipeline | Accuracy | Retrieval success | Completeness | Avg latency | Avg context tokens | Avg steps |
|---|---:|---:|---:|---:|---:|---:|
| rag | 0.2632 | 0.5263 | 0.5263 | 247.40 | 1030.9 | 0.00 |
| graphrag | 0.9474 | 1.0000 | 1.0000 | 417.73 | 1515.6 | 0.00 |
| agentic | 0.9474 | 0.9474 | 0.9474 | 91.93 | 228.6 | 2.42 |

### aggregation (n=21)

| Pipeline | Accuracy | Retrieval success | Completeness | Avg latency | Avg context tokens | Avg steps |
|---|---:|---:|---:|---:|---:|---:|
| rag | 0.0000 | 0.6190 | 0.0779 | 237.14 | 920.6 | 0.00 |
| graphrag | 0.0476 | 0.9048 | 0.5706 | 457.72 | 1511.1 | 0.00 |
| agentic | 0.9524 | 1.0000 | 1.0000 | 1.27 | 4341.2 | 18.10 |

### temporal (n=22)

| Pipeline | Accuracy | Retrieval success | Completeness | Avg latency | Avg context tokens | Avg steps |
|---|---:|---:|---:|---:|---:|---:|
| rag | 0.0909 | 0.4091 | 0.2500 | 258.06 | 1024.0 | 0.00 |
| graphrag | 0.9545 | 0.9545 | 0.9545 | 529.52 | 1460.8 | 0.00 |
| agentic | 0.9545 | 0.9545 | 0.4773 | 0.24 | 243.2 | 3.00 |

### multi_hop (n=28)

| Pipeline | Accuracy | Retrieval success | Completeness | Avg latency | Avg context tokens | Avg steps |
|---|---:|---:|---:|---:|---:|---:|
| rag | 0.1071 | 0.2857 | 0.2857 | 262.33 | 995.4 | 0.00 |
| graphrag | 0.7143 | 0.9286 | 0.9286 | 192.19 | 676.3 | 0.00 |
| agentic | 0.8214 | 0.9643 | 0.9643 | 1.93 | 534.4 | 5.46 |

### superlative (n=10)

| Pipeline | Accuracy | Retrieval success | Completeness | Avg latency | Avg context tokens | Avg steps |
|---|---:|---:|---:|---:|---:|---:|
| rag | 0.0000 | 0.4000 | 0.0438 | 229.65 | 941.9 | 0.00 |
| graphrag | 0.0000 | 1.0000 | 0.7256 | 440.01 | 1666.4 | 0.00 |
| agentic | 1.0000 | 1.0000 | 1.0000 | 1.05 | 3520.0 | 14.80 |

## 3. Failure analysis

### rag (90 failures)

| qid | qtype | gold | predicted | gold doc retrieved | error |
|---|---|---|---|---|---|
| pub-001 | aggregation | 5 | Insufficient evidence in the retrieved context. | True |  |
| pub-002 | temporal | Chen Ding | Insufficient evidence in the retrieved context. | False |  |
| pub-003 | aggregation | 8 | Insufficient evidence in the retrieved context. | True |  |
| pub-004 | superlative | Athletics at the 2008 Summer Olympics – Men's marathon | Insufficient evidence in the retrieved context. | True |  |
| pub-005 | multi_hop | Naim Süleymanoğlu | Miloslav Mečíř | False |  |
| pub-007 | temporal | Renaud Lavillenie | Sergey Bubka | True |  |
| pub-008 | superlative | Sailing at the 2000 Summer Olympics – Soling | Insufficient evidence in the retrieved context. | False |  |
| pub-010 | aggregation | 4 | Insufficient evidence in the retrieved context. | False |  |
| pub-011 | multi_hop | Martina Sáblíková | Lee Seung-hoon | True |  |
| pub-012 | aggregation | 6 | Insufficient evidence in the retrieved context. | True |  |
| pub-013 | temporal | Allison Schmitt | Insufficient evidence in the retrieved context. | False |  |
| pub-014 | multi_hop | Carolina Marín | Insufficient evidence in the retrieved context. | False |  |
| pub-015 | multi_hop | Dani KingLaura TrottJoanna Rowsell | The gold medal in the event held at London Velopark on 3 to 4 August at the 2012 | True |  |
| pub-016 | temporal | Arnd Peiffer | Ole Einar Bjørndalen | False |  |
| pub-017 | multi_hop | Yi Siling | Insufficient evidence in the retrieved context. | False |  |
| pub-018 | temporal | Kevin Jackson | Insufficient evidence in the retrieved context. | False |  |
| pub-019 | aggregation | 3 | Insufficient evidence in the retrieved context. | True |  |
| pub-020 | aggregation | 4 | Insufficient evidence in the retrieved context. | True |  |
| pub-021 | superlative | Alpine skiing at the 1988 Winter Olympics – Men's giant slalom | Insufficient evidence in the retrieved context. | False |  |
| pub-022 | multi_hop | Ayumi Tanimoto | Wu Jingyu | False |  |
| pub-023 | multi_hop | Hwang Young-Cho | Kevin Young | True |  |
| pub-024 | aggregation | 3 | Insufficient evidence in the retrieved context. | False |  |
| pub-026 | temporal | Jaroslav Kulhavý | Greg Van Avermaet | False |  |
| pub-027 | aggregation | 4 | Insufficient evidence in the retrieved context. | True |  |
| pub-028 | multi_hop | Michael Phelps | Answer: Michael Phelps | False |  |
| pub-029 | lookup | 28 | 41 | True |  |
| pub-030 | multi_hop | Emese Szász | Insufficient evidence in the retrieved context. | False |  |
| pub-031 | multi_hop | Dmitry Berestov | Liu Chunhong | False |  |
| pub-033 | aggregation | 4 | Insufficient evidence in the retrieved context. | True |  |
| pub-034 | lookup | 23 | 20 | False |  |
| pub-035 | lookup | 30 | Insufficient evidence in the retrieved context. | False |  |
| pub-036 | temporal | Zou Shiming | Hasanboy Dusmatov from Uzbekistan won the gold medal. | True |  |
| pub-037 | superlative | Shooting at the 2008 Summer Olympics – Men's 50 metre rifle prone | Insufficient evidence in the retrieved context. | True |  |
| pub-038 | multi_hop | Pyrros Dimas | Audley Harrison | False |  |
| pub-039 | temporal | Sandra Perković | Insufficient evidence in the retrieved context. | False |  |
| pub-041 | multi_hop | Nino Schurter | Tony Yoka | False |  |
| pub-042 | lookup | 47 | Insufficient evidence in the retrieved context. | False |  |
| pub-044 | superlative | Fencing at the 2008 Summer Olympics – Men's épée | The men's épée event at the 2008 Summer Olympics had the highest number of compe | True |  |
| pub-045 | aggregation | 20 | Insufficient evidence in the retrieved context. | True |  |
| pub-047 | lookup | 24 | 32 | True |  |
| pub-048 | temporal | Servet Tazegül | Hadi Saei | True |  |
| pub-049 | temporal | Martin Fourcade | Insufficient evidence in the retrieved context. | False |  |
| pub-050 | multi_hop | Lee Sang-hwa | Insufficient evidence in the retrieved context. | False |  |
| pub-051 | aggregation | 6 | Insufficient evidence in the retrieved context. | False |  |
| pub-052 | lookup | 20 | Insufficient evidence in the retrieved context. | False |  |
| pub-053 | superlative | Shooting at the 2016 Summer Olympics – Women's 10 metre air rifle | Insufficient evidence in the retrieved context. | True |  |
| pub-054 | lookup | 24 | 26 | True |  |
| pub-055 | temporal | Sebastián Crismanich | Steven López | True |  |
| pub-056 | temporal | Yevgeny Dementyev | Insufficient evidence in the retrieved context. | True |  |
| pub-057 | temporal | Allyson Felix | Insufficient evidence in the retrieved context. | False |  |
| pub-058 | aggregation | 5 | Insufficient evidence in the retrieved context. | False |  |
| pub-059 | temporal | Laura Dahlmeier | Insufficient evidence in the retrieved context. | False |  |
| pub-060 | multi_hop | Yana Shemyakina | Insufficient evidence in the retrieved context. | False |  |
| pub-061 | lookup | 34 | Insufficient evidence in the retrieved context. | False |  |
| pub-062 | temporal | Tirunesh Dibaba | Insufficient evidence in the retrieved context. | False |  |
| pub-064 | multi_hop | Gabriela Szabo | Szymon Ziółkowski | False |  |
| pub-065 | aggregation | 5 | Insufficient evidence in the retrieved context. | True |  |
| pub-066 | superlative | Weightlifting at the 1992 Summer Olympics – Men's 75 kg | Insufficient evidence in the retrieved context. | False |  |
| pub-067 | multi_hop | Rosannagh MacLennan | Insufficient evidence in the retrieved context. | False |  |
| pub-068 | lookup | 24 | 30 | True |  |
| pub-069 | aggregation | 3 | Insufficient evidence in the retrieved context. | True |  |
| pub-070 | aggregation | 3 | Insufficient evidence in the retrieved context. | False |  |
| pub-071 | lookup | 24 | Insufficient evidence in the retrieved context. | False |  |
| pub-072 | temporal | Lee Sang-hwa | Lee Sang-hwa from South Korea won the gold medal in the women's 500 metres speed | True |  |
| pub-073 | multi_hop | Yang Ling | Insufficient evidence in the retrieved context. | False |  |
| pub-074 | lookup | 17 | Insufficient evidence in the retrieved context. | False |  |
| pub-075 | lookup | 32 | Insufficient evidence in the retrieved context. | False |  |
| pub-076 | multi_hop | Julia Mancuso | Kjetil André Aamodt | False |  |
| pub-077 | multi_hop | Kaillie Humphries | Kaillie Humphries of the United States won the gold medal in the women's monobob | True |  |
| pub-078 | aggregation | 7 | Insufficient evidence in the retrieved context. | False |  |
| pub-079 | multi_hop | Leontien Zijlaard | Jan Ullrich | False |  |
| pub-080 | lookup | 17 | 48 | True |  |
| pub-081 | multi_hop | Guo Wenjun | Insufficient evidence in the retrieved context. | False |  |
| pub-082 | aggregation | 12 | Insufficient evidence in the retrieved context. | True |  |
| pub-084 | superlative | Fencing at the 1992 Summer Olympics – Men's épée | Insufficient evidence in the retrieved context. | False |  |
| pub-085 | superlative | Sailing at the 2008 Summer Olympics – Men's 470 | Insufficient evidence in the retrieved context. | False |  |
| pub-087 | aggregation | 17 | Insufficient evidence in the retrieved context. | True |  |
| pub-088 | superlative | Alpine skiing at the 1992 Winter Olympics – Men's giant slalom | The men's 15 kilometre freestyle pursuit in cross-country skiing at the 1992 Win | False |  |
| pub-089 | aggregation | 5 | Insufficient evidence in the retrieved context. | False |  |
| pub-090 | temporal | Aslanbek Khushtov | Ghasem Rezaei | True |  |
| pub-091 | lookup | 18 | Insufficient evidence in the retrieved context. | False |  |
| pub-092 | aggregation | 6 | Insufficient evidence in the retrieved context. | False |  |
| pub-093 | aggregation | 4 | Insufficient evidence in the retrieved context. | True |  |
| pub-094 | temporal | Oleg Kucherenko | Insufficient evidence in the retrieved context. | False |  |
| pub-095 | multi_hop | Valerie Vili | Irving Saladino of Panama won the gold medal in the men's long jump event held a | False |  |
| pub-096 | multi_hop | Fazliddin Gaibnazarov | Answer: Fazliddin Gaibnazarov | True |  |
| pub-097 | temporal | Yana Shemyakina | Insufficient evidence in the retrieved context. | False |  |
| pub-098 | multi_hop | Bekzat Sattarkhanov | Audley Harrison | False |  |
| pub-099 | multi_hop | Erik LesserDaniel BöhmArnd PeifferSimon Schempp | Insufficient evidence in the retrieved context. | False |  |
| pub-100 | temporal | Missy Franklin | Insufficient evidence in the retrieved context. | False |  |

### graphrag (40 failures)

| qid | qtype | gold | predicted | gold doc retrieved | error |
|---|---|---|---|---|---|
| pub-001 | aggregation | 5 |  | False | timeout: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read |
| pub-003 | aggregation | 8 | Insufficient evidence in the retrieved context. | True |  |
| pub-004 | superlative | Athletics at the 2008 Summer Olympics – Men's marathon | Insufficient evidence in the retrieved context. | True |  |
| pub-008 | superlative | Sailing at the 2000 Summer Olympics – Soling | Laser | True |  |
| pub-010 | aggregation | 4 | Insufficient evidence in the retrieved context. | True |  |
| pub-012 | aggregation | 6 | 3 | True |  |
| pub-014 | multi_hop | Carolina Marín | Chen Long | True |  |
| pub-015 | multi_hop | Dani KingLaura TrottJoanna Rowsell | Insufficient evidence in the retrieved context. | True |  |
| pub-016 | temporal | Arnd Peiffer |  | False | timeout: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read |
| pub-020 | aggregation | 4 |  | False | timeout: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read |
| pub-021 | superlative | Alpine skiing at the 1988 Winter Olympics – Men's giant slalom | The Women's combined event at the 1988 Winter Olympics was held at Nakiska, with | True |  |
| pub-022 | multi_hop | Ayumi Tanimoto | Insufficient evidence in the retrieved context. | False |  |
| pub-024 | aggregation | 3 | Insufficient evidence in the retrieved context. | True |  |
| pub-027 | aggregation | 4 | Insufficient evidence in the retrieved context. | True |  |
| pub-033 | aggregation | 4 | Insufficient evidence in the retrieved context. | True |  |
| pub-037 | superlative | Shooting at the 2008 Summer Olympics – Men's 50 metre rifle prone | The Men's 10 metre air rifle event had the highest number of competitors with 51 | True |  |
| pub-038 | multi_hop | Pyrros Dimas | Audley Harrison | True |  |
| pub-042 | lookup | 47 | **Answer:**

Kjetil André Aamodt won the **Men's Super-G** event at the **Albert | True |  |
| pub-044 | superlative | Fencing at the 2008 Summer Olympics – Men's épée | Insufficient evidence in the retrieved context. | True |  |
| pub-045 | aggregation | 20 | 3 | True |  |
| pub-051 | aggregation | 6 | Insufficient evidence in the retrieved context. | True |  |
| pub-053 | superlative | Shooting at the 2016 Summer Olympics – Women's 10 metre air rifle | Insufficient evidence in the retrieved context. | True |  |
| pub-058 | aggregation | 5 | 3 | True |  |
| pub-060 | multi_hop | Yana Shemyakina | Mansur Isaev | False |  |
| pub-065 | aggregation | 5 | Insufficient evidence in the retrieved context. | True |  |
| pub-066 | superlative | Weightlifting at the 1992 Summer Olympics – Men's 75 kg | The Men's 75 kg event had the highest number of competitors with 34. | True |  |
| pub-069 | aggregation | 3 | Insufficient evidence in the retrieved context. | True |  |
| pub-070 | aggregation | 3 | Answer: No new world or Olympic records were set during the 2010 Winter Olympics | True |  |
| pub-077 | multi_hop | Kaillie Humphries | Francesco Friedrich and Thorsten Margis | True |  |
| pub-078 | aggregation | 7 | Insufficient evidence in the retrieved context. | True |  |
| pub-082 | aggregation | 12 | The **women's 100 metre freestyle** event at the **1996 Summer Olympics** took p | True |  |
| pub-084 | superlative | Fencing at the 1992 Summer Olympics – Men's épée | The men's team épée event had the highest number of competitors with 60. | True |  |
| pub-085 | superlative | Sailing at the 2008 Summer Olympics – Men's 470 | Men's 470 | True |  |
| pub-087 | aggregation | 17 | Insufficient evidence in the retrieved context. | True |  |
| pub-088 | superlative | Alpine skiing at the 1992 Winter Olympics – Men's giant slalom | **Answer:**

The Women's combined competition of the 1992 Winter Olympics in Alb | True |  |
| pub-089 | aggregation | 5 | Insufficient evidence in the retrieved context. | True |  |
| pub-092 | aggregation | 6 | Insufficient evidence in the retrieved context. | True |  |
| pub-093 | aggregation | 4 | Insufficient evidence in the retrieved context. | True |  |
| pub-095 | multi_hop | Valerie Vili | Pamela Jelimo | True |  |
| pub-096 | multi_hop | Fazliddin Gaibnazarov | Shakhobidin Zoirov | True |  |

### agentic (8 failures)

| qid | qtype | gold | predicted | gold doc retrieved | error |
|---|---|---|---|---|---|
| pub-022 | multi_hop | Ayumi Tanimoto | Insufficient evidence in the retrieved context. | False |  |
| pub-038 | multi_hop | Pyrros Dimas | Audley Harrison | True |  |
| pub-051 | aggregation | 6 | Insufficient evidence in the retrieved context. | True |  |
| pub-055 | temporal | Sebastián Crismanich | Carlo Molfetta | False |  |
| pub-060 | multi_hop | Yana Shemyakina | Mansur Isaev | True |  |
| pub-075 | lookup | 32 |  | False | timeout: HTTP Error 408: Timeout |
| pub-095 | multi_hop | Valerie Vili | Pamela Jelimo | True |  |
| pub-098 | multi_hop | Bekzat Sattarkhanov | Isabel Fernández | True |  |

Agentic failure categories:

- success: 92
- wrong_answer_with_gold_docs: 4
- insufficient_evidence: 1
- iteration_limit_no_answer: 1
- timeout: 1
- wrong_answer_retrieval_miss: 1

## 4. Agentic effectiveness

- Accuracy vs RAG: +0.8200
- Accuracy vs GraphRAG: +0.3200
- Avg context-token delta vs RAG: +522.8
- Avg context-token delta vs GraphRAG: +227.5
- Avg LLM-total-token delta vs RAG: -1451.3
- Avg LLM-total-token delta vs GraphRAG: -2260.8
- Avg latency delta vs RAG: -231.57s
- Avg latency delta vs GraphRAG: -371.37s
- Average investigation steps: 7.93
- Multiple investigative actions: 99.0%
- Minimal investigation (≤2 tools): 10.0%
- Agentic correct and RAG wrong: 82
- Agentic correct and GraphRAG wrong: 35
- All three correct (agent unnecessary for correctness): 10
- Agentic still failed: 8

## 5. Which questions need an agent?

Measured from this 100-question public set, not a generic claim.

### lookup (n=19)

- All three correct: 5 (26.3%)
- Agentic only correct: 1 (5.3%)
- Agentic needed vs RAG: 13
- Agentic needed vs GraphRAG: 1
- GraphRAG correct and Agentic wrong: 1
- All three failed: 0

### aggregation (n=21)

- All three correct: 0 (0.0%)
- Agentic only correct: 19 (90.5%)
- Agentic needed vs RAG: 20
- Agentic needed vs GraphRAG: 19
- GraphRAG correct and Agentic wrong: 0
- All three failed: 1

### temporal (n=22)

- All three correct: 2 (9.1%)
- Agentic only correct: 1 (4.5%)
- Agentic needed vs RAG: 19
- Agentic needed vs GraphRAG: 1
- GraphRAG correct and Agentic wrong: 1
- All three failed: 0

### multi_hop (n=28)

- All three correct: 3 (10.7%)
- Agentic only correct: 4 (14.3%)
- Agentic needed vs RAG: 20
- Agentic needed vs GraphRAG: 4
- GraphRAG correct and Agentic wrong: 1
- All three failed: 4

### superlative (n=10)

- All three correct: 0 (0.0%)
- Agentic only correct: 10 (100.0%)
- Agentic needed vs RAG: 10
- Agentic needed vs GraphRAG: 10
- GraphRAG correct and Agentic wrong: 0
- All three failed: 0

Interpretation:

- **lookup: mixed.** All-correct 26.3%; Agentic-only 5.3%; Agentic beats GraphRAG on 1/19.
- **aggregation: agent is worth it.** Agentic-only correctness is 90.5% and Agentic beats GraphRAG on 19/21 questions.
- **temporal: mixed.** All-correct 9.1%; Agentic-only 4.5%; Agentic beats GraphRAG on 1/22.
- **multi_hop: mixed.** All-correct 10.7%; Agentic-only 14.3%; Agentic beats GraphRAG on 4/28.
- **superlative: agent is worth it.** Agentic-only correctness is 100.0% and Agentic beats GraphRAG on 10/10 questions.

## 6. Artifacts

- `eval/results/final_public_benchmark/benchmark_public_final.json`
- `eval/results/final_public_benchmark/benchmark_public_summary.json`
- `eval/results/final_public_benchmark/benchmark_public_report.md`
- `eval/results/final_public_benchmark/benchmark_public_failures.json`
- `eval/results/final_public_benchmark/benchmark_public_failures.csv`

