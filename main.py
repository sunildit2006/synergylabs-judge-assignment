import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dotenv import load_dotenv
from judge import GeminiJudge
from mitigations import evaluate_pairwise_unbiased, compute_flip_rate
from aggregator import build_suite_report
from validator import compute_gold_agreement, run_adversarial_probes

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.6-flash")


def main():
    judge = GeminiJudge(model_name=JUDGE_MODEL)

    with open(os.path.join(BASE_DIR, "data", "test_suites", "general_qa.json")) as f:
        test_cases = json.load(f)

    with open(os.path.join(BASE_DIR, "data", "gold_labels", "human_annotated.json")) as f:
        gold_labels = {item["id"]: item["human_winner"] for item in json.load(f)}

    with open(os.path.join(BASE_DIR, "data", "test_suites", "adversarial_probes.json")) as f:
        probes = json.load(f)

    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Running {len(test_cases)} pairwise cases (each run in BOTH orders for position-bias check)...\n")

    position_results = []
    for case in test_cases:
        print(f"  Judging {case['id']}...")
        result = evaluate_pairwise_unbiased(
            judge, case["input"], case["system_prompt"], case["output_a"], case["output_b"],
            case_id=case["id"],
        )
        position_results.append(result)
        time.sleep(10)

    flip_rate = compute_flip_rate(position_results)
    suite_report = build_suite_report(position_results, flip_rate)

    print("\n=== SUITE REPORT (A vs B) ===")
    print(json.dumps(suite_report, indent=2))

    human_winners = [gold_labels[r["case_id"]] for r in position_results if r["case_id"] in gold_labels]
    judge_winners = [r["final_winner"] for r in position_results if r["case_id"] in gold_labels]
    judge_winners = ["Tie" if w.startswith("Tie") else w for w in judge_winners]
    gold_agreement = compute_gold_agreement(human_winners, judge_winners)

    print("\n=== JUDGE VALIDATION: gold-label agreement ===")
    print(json.dumps(gold_agreement, indent=2))

    time.sleep(10)

    print(f"\nRunning {len(probes)} adversarial probes...")
    probe_results = run_adversarial_probes(judge, probes)

    print("\n=== JUDGE VALIDATION: adversarial probes ===")
    print(json.dumps(probe_results, indent=2))

    print(f"\nTotal judge API calls: {judge.call_count}")

    full_output = {
        "judge_model": JUDGE_MODEL,
        "suite_report": suite_report,
        "gold_agreement": gold_agreement,
        "adversarial_probes": probe_results,
        "per_case_position_results": position_results,
        "total_judge_calls": judge.call_count,
    }
    out_path = os.path.join(BASE_DIR, "logs", "suite_report.json")
    with open(out_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
