"""
Failure classification for failed tasks.

Categorizes WHY a task failed into one of 6 categories using either
deterministic shortcuts (timeout, tool_failure) or an LLM classifier
for ambiguous failures.

Reference: docs/prd-plus.md Section 4
"""

import json
import logging
from typing import List, Optional

from ant_coding.core.config import get_env
from ant_coding.observability.event_logger import Event, EventType
from ant_coding.tasks.types import VALID_FAILURE_CATEGORIES, TaskResult

logger = logging.getLogger(__name__)

# Default classifier model — cheap and fast for scale
DEFAULT_CLASSIFIER_MODEL = "gemini/gemini-2.5-flash"

CLASSIFIER_SYSTEM_PROMPT = """You are a failure analysis expert. Given information about a failed software engineering task, classify the root cause into exactly ONE of these categories:

1. **planning** — Agent made a wrong plan or misunderstood the task
2. **implementation** — Code logic errors, wrong API usage, incorrect fix
3. **integration** — Components don't work together, import errors, wrong file paths
4. **hallucination_cascade** — Agent generated non-existent APIs, functions, or files
5. **timeout** — Task ran out of time
6. **tool_failure** — Tool execution failed (command errors, permission issues)

Respond with ONLY a JSON object:
{"category": "<one of the 6 categories>", "reasoning": "<brief explanation>"}"""

CLASSIFIER_USER_TEMPLATE = """## Task Description
{task_description}

## Generated Patch
{patch}

## Test Output
{test_output}

## Recent Events (last 20)
{events_summary}

## Memory Access Summary
{memory_summary}

Classify the failure root cause."""


class FailureClassifier:
    """
    Classifies failed tasks into one of 6 failure categories.

    Uses deterministic shortcuts for clear cases (timeout, tool_failure)
    and falls back to an LLM classifier for ambiguous failures.
    """

    def __init__(
        self,
        classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
        api_key_env: str = "CLASSIFIER_API_KEY",
    ):
        """
        Initialize the FailureClassifier.

        Args:
            classifier_model: LiteLLM model identifier for classification.
            api_key_env: Environment variable name for the classifier API key.
        """
        self.classifier_model = classifier_model
        self.api_key_env = api_key_env

    async def classify(
        self,
        task_description: str,
        result: TaskResult,
        events: Optional[List[Event]] = None,
    ) -> str:
        """
        Classify the failure category for a failed task.

        Args:
            task_description: Description of the task.
            result: The failed TaskResult.
            events: Optional list of events from the task execution.

        Returns:
            One of the 6 valid failure categories.
        """
        # Deterministic shortcuts
        shortcut = self._check_shortcuts(result, events)
        if shortcut:
            return shortcut

        # LLM-based classification
        return await self._llm_classify(task_description, result, events)

    def _check_shortcuts(
        self,
        result: TaskResult,
        events: Optional[List[Event]] = None,
    ) -> Optional[str]:
        """
        Check for deterministic failure categories that don't need LLM.

        Args:
            result: The failed TaskResult.
            events: Optional event list.

        Returns:
            Category string if a shortcut matches, None otherwise.
        """
        # Timeout shortcut
        error = result.error or ""
        if "timeout" in error.lower() or "timed out" in error.lower():
            return "timeout"

        # Tool failure shortcut — check event log for TOOL_CALL failures
        if events:
            tool_failures = [
                e for e in events
                if e.type == EventType.TOOL_CALL
                and e.task_id == result.task_id
                and not e.payload.get("success", True)
            ]
            if tool_failures:
                return "tool_failure"

        return None

    async def _llm_classify(
        self,
        task_description: str,
        result: TaskResult,
        events: Optional[List[Event]] = None,
    ) -> str:
        """
        Use an LLM to classify the failure.

        Args:
            task_description: Description of the task.
            result: The failed TaskResult.
            events: Optional event list.

        Returns:
            One of the 6 valid failure categories.
        """
        from litellm import acompletion

        events_summary = self._format_events(events, result.task_id)
        memory_summary = self._format_memory_summary(events, result.task_id)

        # Extract patch from traces if available
        patch = ""
        for trace in result.agent_traces:
            if trace.get("action") in ("solve", "implement", "merge"):
                patch = str(trace.get("output", ""))
                break

        user_content = CLASSIFIER_USER_TEMPLATE.format(
            task_description=task_description,
            patch=patch or "(no patch generated)",
            test_output=result.error or "(no test output)",
            events_summary=events_summary,
            memory_summary=memory_summary,
        )

        try:
            api_key = get_env(self.api_key_env)
        except Exception:
            api_key = None

        try:
            response = await acompletion(
                model=self.classifier_model,
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                api_key=api_key,
                max_tokens=200,
                temperature=0.0,
            )

            return self._parse_response(response.choices[0].message.content)

        except Exception as e:
            logger.warning(f"Failure classification failed: {e}. Defaulting to 'implementation'.")
            return "implementation"

    def _parse_response(self, content: str) -> str:
        """
        Parse the classifier's JSON response.

        Args:
            content: Raw response string.

        Returns:
            A valid failure category string.
        """
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)
            category = data.get("category", "").lower().strip()

            if category in VALID_FAILURE_CATEGORIES:
                return category

            logger.warning(f"Invalid category '{category}'. Defaulting to 'implementation'.")
            return "implementation"

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse classifier response: {e}. Defaulting to 'implementation'.")
            return "implementation"

    def _format_events(
        self, events: Optional[List[Event]], task_id: str
    ) -> str:
        """Format the last 20 events for the classifier prompt."""
        if not events:
            return "(no events available)"

        task_events = [e for e in events if e.task_id == task_id]
        last_20 = task_events[-20:]

        if not last_20:
            return "(no events for this task)"

        lines = []
        for e in last_20:
            agent = e.agent_id or "system"
            lines.append(f"- [{e.type.value}] agent={agent} payload={e.payload}")
        return "\n".join(lines)

    def _format_memory_summary(
        self, events: Optional[List[Event]], task_id: str
    ) -> str:
        """Summarize memory reads, highlighting those that returned None."""
        if not events:
            return "(no events available)"

        reads = [
            e for e in events
            if e.type == EventType.MEMORY_READ and e.task_id == task_id
        ]

        if not reads:
            return "(no memory reads)"

        lines = []
        for r in reads:
            agent = r.payload.get("agent", "unknown")
            key = r.payload.get("key", "?")
            found = r.payload.get("found", True)
            status = "found" if found else "NOT FOUND (information gap)"
            lines.append(f"- agent={agent} key={key} → {status}")
        return "\n".join(lines)
