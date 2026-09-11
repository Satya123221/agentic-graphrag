# Hidden evaluation report

Answerless execution report. No accuracy is computed because hidden questions have no gold answers.

## Counts

- Questions: 50
- Records: 150 / 150
- Successful executions: 150
- Failed executions: 0
- Timeouts: 0
- Complete: True

Records per pipeline:

- rag: 50
- graphrag: 50
- agentic: 50

## Latency (seconds)

| Pipeline | n | success | failed | timeouts | avg | median | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rag | 50 | 50 | 0 | 0 | 194.48 | 192.34 | 133.06 | 289.5 |
| graphrag | 50 | 50 | 0 | 0 | 303.64 | 334.15 | 0.07 | 492.26 |
| agentic | 50 | 50 | 0 | 0 | 4.9 | 0.43 | 0.03 | 57.61 |

## Agentic steps

- Average steps: 8.58
- Median steps: 7.0
- Min / max steps: 2 / 44
- Average tool calls: 8.58
- Records with investigation trace: 50

## Ollama usage

- Records with token metadata: 103
- Average total LLM tokens (where available): 2341.0

## Question types in the hidden file

- multi_hop: 10
- lookup: 7
- aggregation: 15
- superlative: 10
- temporal: 8

## Artifacts

- `eval/results/hidden_benchmark/hidden_results.jsonl`
- `eval/results/hidden_benchmark/hidden_results.json`
- `eval/results/hidden_benchmark/hidden_summary.json`
- `eval/results/hidden_benchmark/hidden_report.md`

Resume by rerunning `python -m eval.run_hidden`. Completed qid/pipeline pairs are skipped.
