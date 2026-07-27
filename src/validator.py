from sklearn.metrics import cohen_kappa_score
import numpy as np


def compute_gold_agreement(human_winners: list, judge_winners: list) -> dict:
    y_human = np.array(human_winners)
    y_judge = np.array(judge_winners)
    exact_match = float(np.mean(y_human == y_judge))

    labels = sorted(set(human_winners) | set(judge_winners))
    kappa = float(cohen_kappa_score(y_human, y_judge, labels=labels)) if len(labels) > 1 else None

    return {
        "exact_agreement_rate": round(exact_match, 3),
        "cohens_kappa": round(kappa, 3) if kappa is not None else None,
        "sample_size": len(human_winners),
    }


def compute_test_retest_consistency(judge, cases: list) -> dict:
    identical = 0
    details = []
    for case in cases:
        v1 = judge.judge_pair(case["input"], case["system_prompt"], case["output_a"], case["output_b"],
                               case_id=f"{case.get('id', 'x')}_retest1")
        v2 = judge.judge_pair(case["input"], case["system_prompt"], case["output_a"], case["output_b"],
                               case_id=f"{case.get('id', 'x')}_retest2")
        same = v1.winner == v2.winner
        identical += int(same)
        details.append({"case_id": case.get("id"), "run1": v1.winner, "run2": v2.winner, "consistent": same})

    n = len(cases)
    return {
        "consistency_rate": round(identical / n, 3) if n else None,
        "num_cases": n,
        "details": details,
    }


def run_adversarial_probes(judge, probes: list) -> dict:
    results = []
    passed = 0
    for probe in probes:
        v = judge.judge_pair(probe["input"], probe["system_prompt"], probe["output_a"], probe["output_b"],
                              case_id=f"probe_{probe['probe_type']}")
        fooled = v.winner != probe["expected_safe_winner"] and v.winner != "Tie"
        results.append({
            "probe_type": probe["probe_type"],
            "expected_safe_winner": probe["expected_safe_winner"],
            "actual_winner": v.winner,
            "fooled": fooled,
        })
        passed += int(not fooled)

    n = len(probes)
    return {
        "num_probes": n,
        "pass_rate": round(passed / n, 3) if n else None,
        "details": results,
    }
