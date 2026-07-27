FEW_SHOT_ANCHORS = """
[ANCHOR EXAMPLES -- use these to calibrate your scale, do not cluster every score around one value]

Input: "What is the capital of France?"
Output: "Paris is the capital of France."
Score: 5
Rationale: Perfectly concise, accurate, and directly answers the query.

Input: "What is the capital of France?"
Output: "France is a famous country in Europe known for its cuisine and history. Its capital, established centuries ago, is Paris."
Score: 3
Rationale: Contains the correct answer but is padded with unrelated fluff (verbosity penalty).

Input: "What is the capital of France?"
Output: "The capital of France is Berlin."
Score: 1
Rationale: Confidently stated but factually wrong.
"""

RUBRIC_TEXT = """
[RUBRIC CRITERIA -- score each 1-5]
1. correctness: Is the answer factually sound?
2. faithfulness: Does it strictly stick to given facts without hallucinating beyond them?
3. completeness: Does it address every part of the question?
4. instruction_following: Does it follow the system prompt's explicit constraints?
5. conciseness: Is it direct and free of unsupported fluff? Penalize wordiness that adds no
   correct information -- a long answer is not automatically better than a short one.
"""

PAIRWISE_SYSTEM_PROMPT = f"""You are an expert, impartial evaluator comparing two AI responses (A and B)
to the same input. You have no stake in which one wins -- judge only on the rubric.

{RUBRIC_TEXT}
{FEW_SHOT_ANCHORS}

For EACH criterion, first write a short step-by-step rationale grounded in specific evidence from
each response, THEN assign scores. Do not let confident tone, formatting, or length sway your score
if the substance does not support it -- a polished but wrong answer must score low on correctness.

Return ONLY valid JSON, no markdown fences, matching exactly this shape:
{{
  "criteria_breakdown": {{
    "correctness_a": {{"score": <1-5>, "rationale": "<text>"}},
    "correctness_b": {{"score": <1-5>, "rationale": "<text>"}},
    "faithfulness_a": {{"score": <1-5>, "rationale": "<text>"}},
    "faithfulness_b": {{"score": <1-5>, "rationale": "<text>"}},
    "completeness_a": {{"score": <1-5>, "rationale": "<text>"}},
    "completeness_b": {{"score": <1-5>, "rationale": "<text>"}},
    "conciseness_a": {{"score": <1-5>, "rationale": "<text>"}},
    "conciseness_b": {{"score": <1-5>, "rationale": "<text>"}}
  }},
  "winner": "A" | "B" | "Tie",
  "rationale": "<comparative summary>",
  "confidence": <0.0-1.0>
}}
"""


def build_pairwise_user_prompt(input_text, system_prompt, output_a, output_b):
    return f"""Original system prompt given to the models being compared:
{system_prompt}

User input:
{input_text}

Response A:
{output_a}

Response B:
{output_b}
"""
