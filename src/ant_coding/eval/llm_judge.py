"""
LLM-as-Judge for scoring patches on PRD+ dimensions.

Evaluates generated patches on 4 dimensions: correctness, minimality,
code_quality, completeness. Uses a separate model from the one being tested.
"""

import json
import logging
from typing import Any, Dict, Optional

from ant_coding.core.config import get_env

logger = logging.getLogger(__name__)

# Default judge model — should differ from the model being tested
DEFAULT_JUDGE_MODEL = "gemini/gemini-2.5-flash"

JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer evaluating a generated patch for a software engineering task.

Score the patch on these 4 dimensions (1-5 scale):

1. **correctness** (1-5):
   - 5: Root cause identified and properly fixed
   - 3: Symptom patched but root cause not addressed
   - 1: Wrong fix, introduces new issues

2. **minimality** (1-5):
   - 5: Minimal diff, changes only what's necessary
   - 3: Some extra changes but mostly focused
   - 1: Rewrote everything, massive unnecessary changes

3. **code_quality** (1-5):
   - 5: Production-grade, follows conventions, well-structured
   - 3: Acceptable, functional but could be cleaner
   - 1: Hacky, poor style, unclear intent

4. **completeness** (1-5):
   - 5: Comprehensive fix handling all edge cases
   - 3: Main case handled, edge cases missed
   - 1: Incomplete, partial fix

Respond ONLY with a JSON object (no markdown, no code fences):
{
  "correctness": <int 1-5>,
  "minimality": <int 1-5>,
  "code_quality": <int 1-5>,
  "completeness": <int 1-5>,
  "reasoning": "<brief explanation>"
}"""

JUDGE_USER_TEMPLATE = """## Task Description
{task_description}

## Generated Patch
{patch}

## Test Output
{test_output}

Evaluate this patch and respond with the JSON scoring."""


def _default_scores(error_note: str = "") -> Dict[str, Any]:
    """Return default scores when judge evaluation fails."""
    return {
        "correctness": 1,
        "minimality": 1,
        "code_quality": 1,
        "completeness": 1,
        "overall": 1.0,
        "reasoning": f"Judge evaluation failed: {error_note}" if error_note else "No evaluation performed",
    }


class LLMJudge:
    """
    Evaluates generated patches using an LLM judge model.

    Uses a separate model from the one being tested to avoid bias.
    Scores on 4 PRD+ dimensions and computes a weighted overall score.
    """

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        api_key_env: str = "JUDGE_API_KEY",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the LLM Judge.

        Args:
            judge_model: LiteLLM model identifier for the judge.
            api_key_env: Environment variable name for the judge API key.
            weights: Optional dimension weights for overall score.
                Defaults to equal weights.
        """
        self.judge_model = judge_model
        self.api_key_env = api_key_env
        self.weights = weights or {
            "correctness": 0.4,
            "minimality": 0.2,
            "code_quality": 0.2,
            "completeness": 0.2,
        }

    async def evaluate(
        self,
        task_description: str,
        patch: str,
        test_output: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate a generated patch on PRD+ dimensions.

        Args:
            task_description: Description of the task that was solved.
            patch: The generated code patch.
            test_output: Output from running tests (pass/fail info).

        Returns:
            Dict with keys: correctness, minimality, code_quality,
            completeness (each 1-5), overall (weighted float), reasoning (str).
        """
        from litellm import acompletion

        user_content = JUDGE_USER_TEMPLATE.format(
            task_description=task_description,
            patch=patch or "(no patch generated)",
            test_output=test_output or "(no test output)",
        )

        try:
            api_key = get_env(self.api_key_env)
        except Exception:
            api_key = None

        try:
            response = await acompletion(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                api_key=api_key,
                max_tokens=500,
                temperature=0.0,
            )

            return self._parse_response(response.choices[0].message.content)

        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}")
            return _default_scores(str(e))

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """
        Parse the judge's JSON response.

        Args:
            content: Raw response string from the judge model.

        Returns:
            Parsed scores dict with overall score added.
        """
        try:
            # Strip markdown code fences if present
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)

            # Validate and clamp scores
            dimensions = ["correctness", "minimality", "code_quality", "completeness"]
            for dim in dimensions:
                if dim not in data or not isinstance(data[dim], (int, float)):
                    data[dim] = 1
                data[dim] = max(1, min(5, int(data[dim])))

            # Compute weighted overall score
            overall = sum(
                data[dim] * self.weights.get(dim, 0.25) for dim in dimensions
            )
            data["overall"] = round(overall, 2)

            # Ensure reasoning exists
            if "reasoning" not in data or not isinstance(data["reasoning"], str):
                data["reasoning"] = "No reasoning provided"

            return data

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse judge response: {e}")
            return _default_scores(f"Parse error: {e}")
