import os
import json
import time
import uuid
from datetime import datetime, timezone

import google.generativeai as genai
from pydantic import ValidationError

from schema import PairwiseVerdict
from prompts import PAIRWISE_SYSTEM_PROMPT, build_pairwise_user_prompt

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "judge_log.jsonl")


class JudgeError(Exception):
    pass


class GeminiJudge:
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name, system_instruction=PAIRWISE_SYSTEM_PROMPT)
        self.call_count = 0
        self.total_tokens = 0

    def _log(self, record: dict):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _extract_json(self, raw: str) -> dict:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(cleaned[start:end + 1])
            raise

    def judge_pair(self, input_text: str, system_prompt: str, output_a: str, output_b: str,
                    case_id: str = None, max_attempts: int = 3) -> PairwiseVerdict:
        user_prompt = build_pairwise_user_prompt(input_text, system_prompt, output_a, output_b)
        case_id = case_id or str(uuid.uuid4())[:8]

        last_error = None
        for attempt in range(max_attempts):
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                response = self.model.generate_content(user_prompt)
                raw_text = response.text
                self.call_count += 1
                usage = getattr(response, "usage_metadata", None)
                tokens = usage.total_token_count if usage else None
                if tokens:
                    self.total_tokens += tokens

                parsed_json = self._extract_json(raw_text)
                verdict = PairwiseVerdict(**parsed_json)

                self._log({
                    "case_id": case_id, "timestamp": timestamp, "attempt": attempt + 1,
                    "model": self.model_name, "prompt": user_prompt, "raw_response": raw_text,
                    "parsed_verdict": verdict.model_dump(), "tokens": tokens, "status": "ok",
                })
                return verdict

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = f"parse_error: {e}"
                self._log({
                    "case_id": case_id, "timestamp": timestamp, "attempt": attempt + 1,
                    "model": self.model_name, "prompt": user_prompt,
                    "raw_response": raw_text if "raw_text" in dir() else None,
                    "status": "parse_failed", "error": str(e),
                })
                time.sleep(2)
            except Exception as e:
                last_error = f"api_error: {e}"
                self._log({
                    "case_id": case_id, "timestamp": timestamp, "attempt": attempt + 1,
                    "model": self.model_name, "status": "api_failed", "error": str(e),
                })
                wait = 15 * (attempt + 1)
                time.sleep(wait)

        raise JudgeError(f"Gave up after {max_attempts} attempts for case {case_id}: {last_error}")
