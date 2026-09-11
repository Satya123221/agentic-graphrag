"""Assemble dashboard sections from frozen artifacts. Does not invent metrics."""

from collections import Counter, defaultdict

from eval.dashboard.artifacts import PIPELINES, QTYPES, load_run


def _pct(value):
    return round(100.0 * float(value or 0.0), 2)


def _used_ollama_generate(record):
    return (
        record.get("llm_prompt_eval_count") is not None
        or record.get("llm_eval_count") is not None
        or record.get("llm_total_tokens") is not None
    )


def _pipeline_rows(records, pipeline):
    return [row for row in records if row.get("pipeline") == pipeline]


def _by_qid(records):
    grouped = defaultdict(dict)
    for row in records:
        grouped[row["qid"]][row["pipeline"]] = row
    return grouped


def _counts(records, questions):
    pipelines = Counter(row.get("pipeline") for row in records)
    return {
        "n_questions": len(questions) or len({row.get("qid") for row in records}),
        "n_records": len(records),
        "n_rag": int(pipelines.get("rag") or 0),
        "n_graphrag": int(pipelines.get("graphrag") or 0),
        "n_agentic": int(pipelines.get("agentic") or 0),
    }


def _overall(summary):
    overall = summary.get("overall") or {}
    rows = []
    for name in PIPELINES:
        stats = overall.get(name) or {}
        rows.append({
            "pipeline": name,
            "n": stats.get("n"),
            "accuracy": stats.get("overall_accuracy"),
            "accuracy_pct": _pct(stats.get("overall_accuracy")),
            "retrieval_success": stats.get("retrieval_success"),
            "retrieval_success_pct": _pct(stats.get("retrieval_success")),
            "completeness": stats.get("retrieval_completeness"),
            "avg_latency_sec": stats.get("avg_latency_sec"),
            "median_latency_sec": stats.get("median_latency_sec"),
            "avg_context_tokens": stats.get("avg_context_tokens"),
            "avg_llm_prompt_tokens": stats.get("avg_llm_prompt_tokens"),
            "avg_llm_output_tokens": stats.get("avg_llm_output_tokens"),
            "avg_llm_total_tokens": stats.get("avg_llm_total_tokens"),
            "avg_agent_steps": stats.get("avg_agent_steps"),
            "n_failures": stats.get("n_failures"),
            "n_timeouts": stats.get("n_timeouts"),
            "n_errors": stats.get("n_errors"),
        })
    return rows


def _qtype_accuracy(summary):
    by_qtype = summary.get("by_qtype") or {}
    counts = summary.get("qtype_counts") or {}
    rows = []
    for qtype in QTYPES:
        row = {"qtype": qtype, "n": counts.get(qtype)}
        for name in PIPELINES:
            stats = (by_qtype.get(name) or {}).get(qtype) or {}
            row[name + "_accuracy"] = stats.get("overall_accuracy")
            row[name + "_accuracy_pct"] = _pct(stats.get("overall_accuracy"))
        rows.append(row)
    return rows


def _advantage(records, summary):
    grouped = _by_qid(records)
    agent_beats_rag = []
    agent_beats_graph = []
    all_correct = []
    graph_ok_agent_fail = []
    rag_ok_agent_fail = []
    for qid, pipes in grouped.items():
        agent = pipes.get("agentic") or {}
        rag = pipes.get("rag") or {}
        graph = pipes.get("graphrag") or {}
        a_ok = bool(agent.get("normalized_match"))
        r_ok = bool(rag.get("normalized_match"))
        g_ok = bool(graph.get("normalized_match"))
        rec = {
            "qid": qid,
            "qtype": agent.get("qtype") or rag.get("qtype"),
        }
        if a_ok and not r_ok:
            agent_beats_rag.append(rec)
        if a_ok and not g_ok:
            agent_beats_graph.append(rec)
        if a_ok and r_ok and g_ok:
            all_correct.append(rec)
        if g_ok and not a_ok:
            graph_ok_agent_fail.append(rec)
        if r_ok and not a_ok:
            rag_ok_agent_fail.append(rec)
    effect = summary.get("agentic_effectiveness") or {}
    return {
        "agentic_correct_rag_incorrect": {
            "n": len(agent_beats_rag),
            "official_n": effect.get("n_agent_beats_rag"),
            "qids": [row["qid"] for row in agent_beats_rag],
        },
        "agentic_correct_graphrag_incorrect": {
            "n": len(agent_beats_graph),
            "official_n": effect.get("n_agent_beats_graphrag"),
            "qids": [row["qid"] for row in agent_beats_graph],
        },
        "all_three_correct": {
            "n": len(all_correct),
            "official_n": effect.get("n_agent_unnecessary_all_correct"),
            "qids": [row["qid"] for row in all_correct],
        },
        "graphrag_correct_agentic_incorrect": {
            "n": len(graph_ok_agent_fail),
            "qids": [row["qid"] for row in graph_ok_agent_fail],
        },
        "rag_correct_agentic_incorrect": {
            "n": len(rag_ok_agent_fail),
            "qids": [row["qid"] for row in rag_ok_agent_fail],
        },
    }


def _agentic_behavior(records, summary):
    agentic = _pipeline_rows(records, "agentic")
    n = len(agentic)
    with_generate = [row for row in agentic if _used_ollama_generate(row)]
    without_generate = [row for row in agentic if not _used_ollama_generate(row)]
    effect = summary.get("agentic_effectiveness") or {}
    overall = (summary.get("overall") or {}).get("agentic") or {}
    n_ge2 = sum(1 for row in agentic if len(row.get("agent_tool_calls") or []) >= 2)
    n_le2 = sum(1 for row in agentic if len(row.get("agent_tool_calls") or []) <= 2)
    return {
        "n": n,
        "avg_agent_steps": overall.get("avg_agent_steps"),
        "official_avg_investigation_steps": effect.get("avg_investigation_steps"),
        "n_using_ge2_tools": n_ge2,
        "pct_using_ge2_tools": effect.get("pct_multiple_investigative_actions"),
        "n_stopping_le2_tools": n_le2,
        "pct_stopping_le2_tools": effect.get("pct_minimal_investigation"),
        "n_with_ollama_generate": len(with_generate),
        "n_without_ollama_generate": len(without_generate),
        "token_notes": [
            "context_token_count is a whitespace word count of concatenated evidence text, not an Ollama tokenizer count.",
            "avg_llm_prompt_tokens / avg_llm_output_tokens / avg_llm_total_tokens are Ollama generate-call counts, averaged only over records that made a generate call.",
            "90 of 100 official Agentic questions had no Ollama generate call.",
            "token_efficiency_per_1k_tokens in the official summary is accuracy per 1k context words. It is not LLM-cost efficiency and must not be presented as such.",
        ],
    }


def _failures(summary, failures):
    categories = summary.get("agentic_failure_categories") or {}
    agentic_failures = list((failures or {}).get("agentic") or [])
    return {
        "categories": categories,
        "n_agentic_failures": len(agentic_failures),
        "agentic": [
            {
                "qid": row.get("qid"),
                "qtype": row.get("qtype"),
                "predicted_answer": row.get("predicted_answer"),
                "gold_answer": row.get("gold_answer"),
                "gold_doc_retrieved": row.get("gold_doc_retrieved"),
                "error": row.get("error"),
                "latency_sec": row.get("latency_sec"),
            }
            for row in agentic_failures
        ],
    }


def dashboard_view(run_id="public"):
    if run_id == "hidden":
        return hidden_dashboard_view()
    bundle = load_run(run_id)
    summary = bundle["summary"]
    records = bundle["records"]
    questions = bundle["questions"]
    counts = _counts(records, questions)
    return {
        "run_id": run_id,
        "run": bundle["final"].get("run"),
        "model": bundle["final"].get("model"),
        "directory": bundle["directory"],
        "counts": counts,
        "overall": _overall(summary),
        "by_qtype": _qtype_accuracy(summary),
        "advantage": _advantage(records, summary),
        "agentic_behavior": _agentic_behavior(records, summary),
        "failures": _failures(summary, bundle["failures"]),
        "token_warning": (
            "Do not treat Agentic avg_llm_total_tokens as an n=100 LLM-cost figure. "
            "Official Agentic LLM-token averages cover the generate-call subset only."
        ),
    }


def _mean(values):
    return round(sum(values) / len(values), 2) if values else 0.0


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 2)


def _has_mapping_values(value):
    if isinstance(value, dict):
        return any(bool(item) for item in value.values())
    return bool(value)


def _nonempty_answer(record):
    return bool((record.get("answer") or "").strip())


def _hidden_pipeline_completeness(records, pipeline):
    rows = _pipeline_rows(records, pipeline)
    n = len(rows)
    n_answer = sum(1 for row in rows if _nonempty_answer(row))
    n_docs = sum(1 for row in rows if row.get("retrieved_document_ids"))
    n_evidence = sum(1 for row in rows if row.get("evidence"))
    n_llm = sum(1 for row in rows if _used_ollama_generate(row))
    n_errors = sum(1 for row in rows if row.get("error"))
    latencies = [float(row.get("latency_seconds") or 0.0) for row in rows]
    out = {
        "pipeline": pipeline,
        "n": n,
        "n_with_answer": n_answer,
        "n_empty_answer": n - n_answer,
        "n_errors": n_errors,
        "n_with_documents": n_docs,
        "n_with_evidence": n_evidence,
        "n_with_llm_tokens": n_llm,
        "avg_latency_seconds": _mean(latencies),
        "median_latency_seconds": _median(latencies),
        "min_latency_seconds": round(min(latencies), 2) if latencies else 0.0,
        "max_latency_seconds": round(max(latencies), 2) if latencies else 0.0,
        "avg_llm_prompt_tokens": _mean([
            float(row["llm_prompt_eval_count"])
            for row in rows
            if row.get("llm_prompt_eval_count") is not None
        ]),
        "avg_llm_output_tokens": _mean([
            float(row["llm_eval_count"])
            for row in rows
            if row.get("llm_eval_count") is not None
        ]),
        "avg_llm_total_tokens": _mean([
            float(row["llm_total_tokens"])
            for row in rows
            if row.get("llm_total_tokens") is not None
        ]),
    }
    if pipeline == "graphrag":
        out["n_with_graph_paths"] = sum(1 for row in rows if row.get("graph_paths"))
        out["n_with_entities"] = sum(
            1 for row in rows if _has_mapping_values(row.get("matched_entities"))
        )
    if pipeline == "agentic":
        steps = [int(row.get("agent_iterations") or 0) for row in rows]
        tools = [len(row.get("agent_tool_calls") or row.get("tools_used") or []) for row in rows]
        out.update({
            "avg_agent_steps": _mean(steps),
            "median_agent_steps": _median(steps),
            "avg_tool_calls": _mean(tools),
            "n_with_trace": sum(1 for row in rows if row.get("agent_trace")),
            "n_with_entities": sum(
                1 for row in rows if _has_mapping_values(row.get("identified_entities"))
            ),
            "n_with_graph_paths": sum(1 for row in rows if row.get("graph_paths")),
            "n_with_stop_reason": sum(1 for row in rows if row.get("agent_stop_reason")),
            "n_with_ollama_generate": n_llm,
            "n_without_ollama_generate": n - n_llm,
        })
    return out


def _hidden_agentic_behavior(records):
    rows = _pipeline_rows(records, "agentic")
    n = len(rows)
    reasons = Counter(row.get("agent_stop_reason") or "unspecified" for row in rows)
    return {
        "n": n,
        "avg_agent_steps": _mean([int(row.get("agent_iterations") or 0) for row in rows]),
        "median_agent_steps": _median([int(row.get("agent_iterations") or 0) for row in rows]),
        "avg_tool_calls": _mean([
            len(row.get("agent_tool_calls") or row.get("tools_used") or [])
            for row in rows
        ]),
        "n_using_ge2_tools": sum(
            1 for row in rows
            if len(row.get("agent_tool_calls") or row.get("tools_used") or []) >= 2
        ),
        "n_with_entities": sum(
            1 for row in rows if _has_mapping_values(row.get("identified_entities"))
        ),
        "n_with_evidence": sum(1 for row in rows if row.get("evidence")),
        "n_with_graph_paths": sum(1 for row in rows if row.get("graph_paths")),
        "n_with_documents": sum(1 for row in rows if row.get("retrieved_document_ids")),
        "n_with_trace": sum(1 for row in rows if row.get("agent_trace")),
        "n_with_stop_reason": sum(1 for row in rows if row.get("agent_stop_reason")),
        "n_with_ollama_generate": sum(1 for row in rows if _used_ollama_generate(row)),
        "n_without_ollama_generate": sum(1 for row in rows if not _used_ollama_generate(row)),
        "stop_reasons": [
            {"reason": reason, "n": count}
            for reason, count in reasons.most_common()
        ],
    }


def _hidden_questions(records, questions):
    grouped = _by_qid(records)
    order = questions or [{"qid": qid} for qid in sorted(grouped)]
    rows = []
    for item in order:
        qid = item.get("qid")
        pipes = grouped.get(qid) or {}
        rag = pipes.get("rag") or {}
        graph = pipes.get("graphrag") or {}
        agent = pipes.get("agentic") or {}
        rows.append({
            "qid": qid,
            "qtype": item.get("qtype") or rag.get("qtype") or graph.get("qtype") or agent.get("qtype"),
            "question": item.get("question") or rag.get("question") or "",
            "rag_answer": rag.get("answer") if rag.get("answer") is not None else "",
            "graphrag_answer": graph.get("answer") if graph.get("answer") is not None else "",
            "agentic_answer": agent.get("answer") if agent.get("answer") is not None else "",
            "rag_latency_seconds": rag.get("latency_seconds"),
            "graphrag_latency_seconds": graph.get("latency_seconds"),
            "agentic_latency_seconds": agent.get("latency_seconds"),
            "rag_error": rag.get("error"),
            "graphrag_error": graph.get("error"),
            "agentic_error": agent.get("error"),
        })
    return rows


def hidden_dashboard_view():
    """Answerless hidden-run view. Does not calculate or expose accuracy."""
    bundle = load_run("hidden")
    records = bundle["records"]
    questions = bundle["questions"]
    summary = bundle["summary"] or {}
    by_pipeline = [
        _hidden_pipeline_completeness(records, name) for name in PIPELINES
    ]
    return {
        "run_id": "hidden",
        "run": bundle["final"].get("run") or "hidden_benchmark",
        "model": bundle["final"].get("model"),
        "directory": bundle["directory"],
        "counts": _counts(records, questions),
        "n_unique_pairs": len({(row.get("qid"), row.get("pipeline")) for row in records}),
        "complete": bool(summary.get("complete")),
        "no_gold_note": (
            "Hidden accuracy cannot be calculated because there are no gold answers."
        ),
        "completeness": by_pipeline,
        "latency": [
            {
                "pipeline": row["pipeline"],
                "avg_latency_seconds": row["avg_latency_seconds"],
                "median_latency_seconds": row["median_latency_seconds"],
                "min_latency_seconds": row["min_latency_seconds"],
                "max_latency_seconds": row["max_latency_seconds"],
            }
            for row in by_pipeline
        ],
        "retrieval": [
            {
                "pipeline": row["pipeline"],
                "n_with_documents": row["n_with_documents"],
                "n_with_evidence": row["n_with_evidence"],
                **(
                    {
                        "n_with_graph_paths": row.get("n_with_graph_paths"),
                        "n_with_entities": row.get("n_with_entities"),
                    }
                    if row["pipeline"] in ("graphrag", "agentic")
                    else {}
                ),
            }
            for row in by_pipeline
        ],
        "agentic_behavior": _hidden_agentic_behavior(records),
        "llm_tokens": [
            {
                "pipeline": row["pipeline"],
                "n_with_llm_tokens": row["n_with_llm_tokens"],
                "avg_llm_prompt_tokens": row["avg_llm_prompt_tokens"],
                "avg_llm_output_tokens": row["avg_llm_output_tokens"],
                "avg_llm_total_tokens": row["avg_llm_total_tokens"],
            }
            for row in by_pipeline
        ],
        "questions": _hidden_questions(records, questions),
        "token_warning": (
            "Ollama token columns are generate-call metadata averaged only over records "
            "that include those fields. Missing token metadata is not treated as an error."
        ),
    }

