from collections import Counter


def build_suite_report(position_results: list, flip_rate: float) -> dict:
    winners = [r["final_winner"] for r in position_results]
    counts = Counter(winners)

    n = len(position_results)
    a_wins = counts.get("A", 0)
    b_wins = counts.get("B", 0)
    ties = n - a_wins - b_wins

    if a_wins > b_wins:
        overall_winner = "A"
    elif b_wins > a_wins:
        overall_winner = "B"
    else:
        overall_winner = "Tie"

    avg_confidence = round(
        sum(r["forward_confidence"] for r in position_results) / n, 3
    ) if n else None

    return {
        "num_cases": n,
        "a_win_rate": round(a_wins / n, 3) if n else None,
        "b_win_rate": round(b_wins / n, 3) if n else None,
        "tie_rate": round(ties / n, 3) if n else None,
        "overall_winner": overall_winner,
        "position_flip_rate": flip_rate,
        "avg_judge_confidence": avg_confidence,
    }
