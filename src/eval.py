"""Evaluation suite for VoiceOps 2026 ZentroMart Support Agent.

Evaluates:
- Step 1: Baseline agent (concurrent requests, no KB)
- Step 2: RAG agent (grounded answers, retrieval quality, thin query handling)
- Step 3: Conversational agent (multi-turn turns, tool calling, identity verification, escalation)

Saves raw outputs to CSV and computes empirical scores.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple
import pandas as pd

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import BaselineAgent, ConversationalAgent, RAGAgent
from src.retrieval import get_kb_index
from tools.tools import lookup_order
SAMPLE_OUTPUTS_DIR = ROOT_DIR / "sample_outputs"
SAMPLE_OUTPUTS_DIR.mkdir(exist_ok=True)


def check_must_include(text: str, must_include_str: str) -> bool:
    """Check if text contains at least one of the alternatives in must_include."""
    if not must_include_str or pd.isna(must_include_str):
        return True
    text_lower = text.lower()
    alternatives = [alt.strip().lower() for alt in str(must_include_str).split(" / ")]
    for alt in alternatives:
        if not alt:
            continue
        # handle partial / keyword match
        if alt in text_lower:
            return True
        # also match without special punctuation
        alt_clean = re.sub(r"[^\w\s]", "", alt)
        text_clean = re.sub(r"[^\w\s]", "", text_lower)
        if alt_clean and alt_clean in text_clean:
            return True
    return False


def check_must_not_include(text: str, must_not_include_str: str) -> bool:
    """Check that text does NOT contain any of the forbidden phrases."""
    if not must_not_include_str or pd.isna(must_not_include_str):
        return True
    text_lower = text.lower()
    forbidden = [f.strip().lower() for f in str(must_not_include_str).split(" / ")]
    for f in forbidden:
        if f and f in text_lower:
            return False
    return True


def evaluate_single_question(
    agent_type: str,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one question on either Baseline or RAG agent."""
    qid = row["id"]
    question = row["question"]
    exp_decision = str(row.get("expected_decision", "answer")).strip()
    acceptable_decisions = [
        d.strip() for d in str(row.get("acceptable_decisions", exp_decision)).split(";")
    ]
    gold_evidence = str(row.get("gold_evidence", "")).strip()
    hard_negatives = str(row.get("hard_negatives", "")).strip()
    must_include = str(row.get("must_include", ""))
    must_not_include = str(row.get("must_not_include", ""))
    answerable = str(row.get("answerable", "True")).lower() == "true"

    t0 = time.time()
    if agent_type == "baseline":
        agent = BaselineAgent()
        pred_answer, pred_decision, chunks_used = agent.answer_question(question)
    else:
        agent = RAGAgent()
        pred_answer, pred_decision, chunks_used = agent.answer_question(question)
    latency = time.time() - t0

    # 1. Decision match
    decision_pass = (pred_decision == exp_decision) or (pred_decision in acceptable_decisions)

    # 2. Must include check
    inc_pass = check_must_include(pred_answer, must_include)

    # 3. Must not include check
    not_inc_pass = check_must_not_include(pred_answer, must_not_include)

    # 4. Grounded answer overall pass
    answer_pass = inc_pass and not_inc_pass

    # 5. Retrieval checks (for RAG)
    gold_items = [g.strip() for g in gold_evidence.split(";") if g.strip() and g.strip() != "nan"]
    neg_items = [n.strip() for n in hard_negatives.split(";") if n.strip() and n.strip() != "nan"]

    if gold_items:
        retrieval_recall = sum(1 for g in gold_items if g in chunks_used) / len(gold_items)
    else:
        # If unanswerable / no gold evidence, full recall if thin or empty
        retrieval_recall = 1.0 if not answerable else 0.0

    retrieval_precision = 1.0
    if neg_items:
        if any(n in chunks_used for n in neg_items):
            retrieval_precision = 0.0

    return {
        "id": qid,
        "question": question,
        "expected_decision": exp_decision,
        "pred_decision": pred_decision,
        "decision_pass": decision_pass,
        "must_include": must_include,
        "must_not_include": must_not_include,
        "inc_pass": inc_pass,
        "not_inc_pass": not_inc_pass,
        "answer_pass": answer_pass,
        "gold_evidence": gold_evidence,
        "hard_negatives": hard_negatives,
        "chunks_used": " ; ".join(chunks_used),
        "retrieval_recall": retrieval_recall,
        "retrieval_precision": retrieval_precision,
        "pred_answer": pred_answer,
        "latency_sec": round(latency, 3),
    }


def run_evaluation_questions(
    agent_type: str,
    questions_path: str = "practice/questions.csv",
    output_path: str = None,
    max_workers: int = 6,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run concurrent evaluation on questions.csv."""
    df_questions = pd.read_csv(questions_path, dtype=str)
    records = df_questions.to_dict(orient="records")

    results = []
    print(f"Evaluating {len(records)} questions for [{agent_type.upper()}] with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_question, agent_type, r): r["id"]
            for r in records
        }
        for future in as_completed(futures):
            res = future.result()
            results.append(res)

    results.sort(key=lambda x: x["id"])
    res_df = pd.DataFrame(results)

    if output_path:
        res_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved raw outputs to {output_path}")

    # Compute metrics
    total = len(res_df)
    dec_acc = res_df["decision_pass"].mean() * 100.0
    ans_acc = res_df["answer_pass"].mean() * 100.0
    ret_recall = res_df["retrieval_recall"].mean() * 100.0
    ret_prec = res_df["retrieval_precision"].mean() * 100.0
    avg_latency = res_df["latency_sec"].mean()

    # Rubric-aligned combined score (out of 100)
    # Grounded answer: 50%, Retrieval: 30%, Decision: 20%
    if agent_type == "baseline":
        total_score = (ans_acc * 0.6) + (dec_acc * 0.4)
    else:
        total_score = (ans_acc * 0.5) + (ret_recall * 0.3) + (dec_acc * 0.2)

    summary = {
        "total_questions": total,
        "decision_accuracy_pct": round(dec_acc, 2),
        "grounded_answer_accuracy_pct": round(ans_acc, 2),
        "retrieval_recall_pct": round(ret_recall, 2) if agent_type != "baseline" else 0.0,
        "retrieval_precision_pct": round(ret_prec, 2) if agent_type != "baseline" else 0.0,
        "avg_latency_sec": round(avg_latency, 2),
        "composite_score_pct": round(total_score, 2),
    }

    return res_df, summary


def run_evaluation_conversations(
    conversations_path: str = "practice/conversations.csv",
    output_path: str = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run sequential multi-turn evaluation on conversations.csv."""
    df_conv = pd.read_csv(conversations_path, dtype=str)
    groups = df_conv.groupby("conversation_id", sort=False)

    results = []
    print(f"Evaluating {len(groups)} multi-turn conversations ({len(df_conv)} total turns)...")

    for cid, group in groups:
        agent = ConversationalAgent()
        sorted_turns = group.sort_values(by="turn")

        for _, row in sorted_turns.iterrows():
            turn_num = int(row["turn"])
            customer_msg = row["customer_message"]
            exp_decision = str(row["expected_decision"]).strip()
            exp_tools = str(row.get("expected_tool_calls", "")).strip()
            must_inc = str(row.get("must_include", ""))
            must_not_inc = str(row.get("must_not_include", ""))

            # Process turn
            t0 = time.time()
            reply = agent.respond(customer_msg)
            latency = time.time() - t0

            pred_decision = agent.last_decision

            # Check decision
            decision_match = pred_decision == exp_decision

            # Check tool call compliance
            tool_match = True
            if exp_tools and exp_tools != "nan":
                if "lookup_order" in exp_tools:
                    # check if agent is verified or made verification lookup
                    tool_match = agent.stored_order_id is not None
                elif "create_ticket" in exp_tools:
                    tool_match = agent.ticket_created is not None

            # Check must_include and must_not_include
            inc_pass = check_must_include(reply, must_inc)
            not_inc_pass = check_must_not_include(reply, must_not_inc)

            results.append({
                "conversation_id": cid,
                "turn": turn_num,
                "scenario": row.get("scenario", ""),
                "customer_message": customer_msg,
                "expected_decision": exp_decision,
                "pred_decision": pred_decision,
                "decision_match": decision_match,
                "expected_tool_calls": exp_tools,
                "tool_match": tool_match,
                "inc_pass": inc_pass,
                "not_inc_pass": not_inc_pass,
                "reply": reply,
                "latency_sec": round(latency, 3),
            })

    res_df = pd.DataFrame(results)
    if output_path:
        res_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved conversation outputs to {output_path}")

    dec_acc = res_df["decision_match"].mean() * 100.0
    tool_acc = res_df["tool_match"].mean() * 100.0
    content_acc = (res_df["inc_pass"] & res_df["not_inc_pass"]).mean() * 100.0
    conv_score = (dec_acc * 0.4) + (tool_acc * 0.3) + (content_acc * 0.3)

    summary = {
        "total_conversations": len(groups),
        "total_turns": len(res_df),
        "decision_accuracy_pct": round(dec_acc, 2),
        "tool_call_accuracy_pct": round(tool_acc, 2),
        "content_compliance_pct": round(content_acc, 2),
        "conversation_score_pct": round(conv_score, 2),
    }

    return res_df, summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VoiceOps 2026 ZentroMart Agent")
    parser.add_argument("--step", choices=["1", "2", "3", "all"], default="all", help="Evaluation step to run")
    args = parser.parse_args()

    step1_summary = {}
    step2_summary = {}
    step3_summary = {}

    if args.step in ("1", "all"):
        print("\n" + "=" * 60)
        print("STEP 1: EVALUATING BASELINE AGENT")
        print("=" * 60)
        _, step1_summary = run_evaluation_questions(
            agent_type="baseline",
            output_path=str(SAMPLE_OUTPUTS_DIR / "eval_step1_outputs.csv"),
        )
        print(f"Step 1 Baseline Results: {step1_summary}")

    if args.step in ("2", "all"):
        print("\n" + "=" * 60)
        print("STEP 2: EVALUATING RAG AGENT")
        print("=" * 60)
        _, step2_summary = run_evaluation_questions(
            agent_type="rag",
            output_path=str(SAMPLE_OUTPUTS_DIR / "eval_step2_outputs.csv"),
        )
        print(f"Step 2 RAG Results: {step2_summary}")

    if args.step in ("3", "all"):
        print("\n" + "=" * 60)
        print("STEP 3: EVALUATING CONVERSATIONAL AGENT")
        print("=" * 60)
        _, step3_summary = run_evaluation_conversations(
            output_path=str(SAMPLE_OUTPUTS_DIR / "eval_step3_conversations.csv"),
        )
        print(f"Step 3 Conversation Results: {step3_summary}")

    # Print Comparative Summary
    print("\n" + "#" * 60)
    print("FINAL EVALUATION COMPARISON: STEP 1 vs STEP 2")
    print("#" * 60)
    if step1_summary and step2_summary:
        print(f"{'Metric':<35} | {'Step 1 (Baseline)':<18} | {'Step 2 (RAG)':<18}")
        print("-" * 77)
        print(f"{'Total Questions':<35} | {step1_summary.get('total_questions', 0):<18} | {step2_summary.get('total_questions', 0):<18}")
        print(f"{'Decision Accuracy (%)':<35} | {step1_summary.get('decision_accuracy_pct', 0.0):<18.2f} | {step2_summary.get('decision_accuracy_pct', 0.0):<18.2f}")
        print(f"{'Grounded Answer Accuracy (%)':<35} | {step1_summary.get('grounded_answer_accuracy_pct', 0.0):<18.2f} | {step2_summary.get('grounded_answer_accuracy_pct', 0.0):<18.2f}")
        print(f"{'Retrieval Recall (%)':<35} | {step1_summary.get('retrieval_recall_pct', 0.0):<18.2f} | {step2_summary.get('retrieval_recall_pct', 0.0):<18.2f}")
        print(f"{'Retrieval Precision (%)':<35} | {step1_summary.get('retrieval_precision_pct', 0.0):<18.2f} | {step2_summary.get('retrieval_precision_pct', 0.0):<18.2f}")
        print(f"{'Composite Score (%)':<35} | {step1_summary.get('composite_score_pct', 0.0):<18.2f} | {step2_summary.get('composite_score_pct', 0.0):<18.2f}")
        print("-" * 77)
        delta = step2_summary.get("composite_score_pct", 0) - step1_summary.get("composite_score_pct", 0)
        print(f"Improvement from Grounding (RAG over Baseline): +{delta:.2f}%")


if __name__ == "__main__":
    main()
