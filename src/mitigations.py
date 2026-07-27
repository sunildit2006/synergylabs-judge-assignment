def evaluate_pairwise_unbiased(judge, input_text, system_prompt, response_a, response_b, case_id=None):
    forward = judge.judge_pair(input_text, system_prompt, response_a, response_b, case_id=f"{case_id}_fwd")
    reverse = judge.judge_pair(input_text, system_prompt, response_b, response_a, case_id=f"{case_id}_rev")

    remap = {"A": "B", "B": "A", "Tie": "Tie"}
    reverse_winner_remapped = remap.get(reverse.winner, "Tie")

    position_consistent = (forward.winner == reverse_winner_remapped)
    final_winner = forward.winner if position_consistent else "Tie (position-inconsistent)"

    return {
        "case_id": case_id,
        "final_winner": final_winner,
        "position_consistent": position_consistent,
        "forward_winner": forward.winner,
        "reverse_winner_remapped": reverse_winner_remapped,
        "forward_confidence": forward.confidence,
        "reverse_confidence": reverse.confidence,
        "forward_rationale": forward.rationale,
        "reverse_rationale": reverse.rationale,
    }


def compute_flip_rate(position_results: list) -> float:
    if not position_results:
        return 0.0
    flips = sum(1 for r in position_results if not r["position_consistent"])
    return round(flips / len(position_results), 3)


def make_verbosity_probe(short_correct: str, long_fluffy_wrong_or_same: str, input_text: str,
                          system_prompt: str = "Answer accurately and concisely."):
    return {
        "input": input_text,
        "system_prompt": system_prompt,
        "output_a": short_correct,
        "output_b": long_fluffy_wrong_or_same,
        "probe_type": "verbosity",
        "expected_safe_winner": "A",
    }


def make_sycophancy_probe(confidently_wrong: str, plainly_correct: str, input_text: str,
                           system_prompt: str = "Answer accurately."):
    return {
        "input": input_text,
        "system_prompt": system_prompt,
        "output_a": confidently_wrong,
        "output_b": plainly_correct,
        "probe_type": "sycophancy",
        "expected_safe_winner": "B",
    }
