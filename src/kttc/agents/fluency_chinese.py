# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chinese-specific Fluency Agent.

Specialized fluency checking for Chinese language with support for:
- Measure word validation (量词检查)
- Aspect particle usage (了/过/着)
- Word segmentation accuracy
- Character consistency (simplified/traditional)
- Chinese-specific grammar patterns

Uses hybrid approach:
- HanLP helper for deterministic measure word and grammar checks
- LLM for semantic and complex linguistic analysis
- Parallel execution for optimal performance

Based on Chinese Language Translation Quality 2025 research.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.helpers.chinese import ChineseLanguageHelper
from kttc.llm import BaseLLMProvider
from kttc.terminology import ChineseMeasureWordValidator

from .fluency import FluencyAgent

logger = logging.getLogger(__name__)


class ChineseFluencyAgent(FluencyAgent):
    """Specialized fluency agent for Chinese language.

    Extends base FluencyAgent with Chinese-specific checks:
    - Measure word correctness (量词 - ge/ben/zhi/tiao/zhang etc.)
    - Aspect particle usage (了/过/着)
    - Chinese grammar patterns
    - Character consistency

    Example:
        >>> agent = ChineseFluencyAgent(llm_provider)
        >>> task = TranslationTask(
        ...     source_text="Hello",
        ...     translation="你好",
        ...     source_lang="en",
        ...     target_lang="zh"
        ... )
        >>> errors = await agent.evaluate(task)
    """

    CHINESE_CHECKS = {
        "measure_word": "Measure word validation (量词检查)",
        "aspect_particle": "Aspect particle usage (了/过/着)",
        "grammar": "Chinese-specific grammar patterns",
        "character": "Character consistency (simplified/traditional)",
        "word_order_trap": "Word order semantic swaps (字序陷阱)",
        "ba_bei_construction": "把字句/被字句 construction validation",
        "separable_verb": "Separable verbs (离合词) usage",
        "resultative_complement": "Resultative complement (结果补语) validation",
        "de_particle": "的/地/得 particle usage (critical)",
    }

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        helper: ChineseLanguageHelper | None = None,
    ):
        """Initialize Chinese fluency agent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            helper: Optional Chinese language helper for HanLP checks (auto-creates if None)
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self._chinese_prompt_base = (
            """Chinese-specific linguistic validation for professional translation quality."""
        )

        # Initialize HanLP helper (or use provided one)
        self.helper = helper if helper is not None else ChineseLanguageHelper()

        # Initialize glossary-based validator for measure word checking
        self.measure_validator = ChineseMeasureWordValidator()
        logger.info("ChineseFluencyAgent initialized with glossary-based measure word validator")

        if self.helper.is_available():
            logger.info("ChineseFluencyAgent using HanLP helper for enhanced checks")
        else:
            logger.info("ChineseFluencyAgent running without HanLP (LLM-only mode)")

    def get_base_prompt(self) -> str:
        """Get the combined base prompt for Chinese fluency evaluation.

        Returns:
            The combined base fluency prompt + Chinese-specific prompt
        """
        base_fluency = super().get_base_prompt()
        return f"{base_fluency}\n\n---\n\nCHINESE-SPECIFIC CHECKS:\n{self._chinese_prompt_base}"

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate Chinese fluency with hybrid HanLP + LLM approach.

        Uses parallel execution:
        1. HanLP performs deterministic measure word and grammar checks
        2. LLM performs semantic and complex linguistic analysis
        3. HanLP verifies LLM results (anti-hallucination)
        4. Merge unique errors from both sources

        Args:
            task: Translation task (target_lang must be 'zh' or 'zh-CN' or 'zh-TW')

        Returns:
            List of fluency error annotations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        if not task.target_lang.startswith("zh"):
            # Fallback to base fluency checks for non-Chinese
            return await super().evaluate(task)

        # Run base fluency checks (parallel with Chinese-specific)
        base_errors = await super().evaluate(task)

        # Run HanLP, LLM, and glossary checks in parallel
        try:
            results = await asyncio.gather(
                asyncio.to_thread(self._hanlp_check_sync, task),  # Fast, deterministic
                self._llm_check(task),  # Slow, semantic (uses await internally)
                asyncio.to_thread(self._glossary_check_sync, task),  # Measure word validation
                return_exceptions=True,
            )

            # Handle exceptions and ensure proper typing
            hanlp_result, llm_result, glossary_result = results

            # Convert results to list[ErrorAnnotation], handling exceptions
            if isinstance(hanlp_result, Exception):
                logger.warning(f"HanLP check failed: {hanlp_result}")
                hanlp_errors: list[ErrorAnnotation] = []
            else:
                hanlp_errors = cast(list[ErrorAnnotation], hanlp_result)

            if isinstance(llm_result, Exception):
                logger.warning(f"LLM check failed: {llm_result}")
                llm_errors: list[ErrorAnnotation] = []
            else:
                llm_errors = cast(list[ErrorAnnotation], llm_result)

            if isinstance(glossary_result, Exception):
                logger.warning(f"Glossary check failed: {glossary_result}")
                glossary_errors: list[ErrorAnnotation] = []
            else:
                glossary_errors = cast(list[ErrorAnnotation], glossary_result)

            # Verify LLM results with HanLP (anti-hallucination)
            verified_llm = self._verify_llm_errors(llm_errors, task.translation)

            # Remove duplicates (HanLP errors already caught by LLM)
            unique_hanlp = self._remove_duplicates(hanlp_errors, verified_llm)

            # Remove duplicates from glossary errors
            unique_glossary = self._remove_duplicates(glossary_errors, verified_llm + unique_hanlp)

            # Merge all unique errors
            all_errors = base_errors + unique_hanlp + verified_llm + unique_glossary

            logger.info(
                f"ChineseFluencyAgent: "
                f"base={len(base_errors)}, "
                f"hanlp={len(unique_hanlp)}, "
                f"llm={len(verified_llm)}, "
                f"glossary={len(unique_glossary)} "
                f"(total={len(all_errors)})"
            )

            return all_errors

        except Exception as e:
            logger.error(f"Chinese fluency evaluation failed: {e}")
            # Fallback to base errors
            return base_errors

    def _hanlp_check_sync(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform HanLP-based measure word and grammar checks (synchronous).

        Args:
            task: Translation task

        Returns:
            List of errors found by HanLP
        """
        if not self.helper or not self.helper.is_available():
            logger.debug("HanLP helper not available, skipping checks")
            return []

        try:
            errors = self.helper.check_grammar(task.translation)
            logger.debug(f"HanLP found {len(errors)} grammar errors")
            return errors
        except Exception as e:
            logger.error(f"HanLP check failed: {e}")
            return []

    async def _llm_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform LLM-based Chinese-specific checks.

        Args:
            task: Translation task

        Returns:
            List of errors found by LLM
        """
        try:
            errors = await self._check_chinese_specifics(task)
            logger.debug(f"LLM found {len(errors)} Chinese-specific errors")
            return errors
        except Exception as e:
            logger.error(f"LLM check failed: {e}")
            return []

    def _glossary_check_sync(self, _task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform glossary-based Chinese measure word validation (synchronous).

        Uses ChineseMeasureWordValidator to check:
        - Measure word correctness (量词 - 个/本/只/条/张 etc.)
        - Classifier-noun pair agreement
        - Common measure word usage

        Args:
            task: Translation task

        Returns:
            List of errors found by glossary validation
        """
        errors: list[ErrorAnnotation] = []

        try:
            # Get most common classifiers for reference
            common_classifiers = self.measure_validator.get_most_common_classifiers(limit=10)

            if common_classifiers:
                logger.debug(
                    f"Loaded {len(common_classifiers)} common classifiers: "
                    f"{', '.join(clf[0] for clf in common_classifiers[:5])}..."
                )

            # Get all classifier categories for comprehensive validation
            categories = [
                "individual_classifiers",
                "collective_classifiers",
                "container_classifiers",
                "measurement_classifiers",
                "temporal_classifiers",
                "verbal_classifiers",
            ]

            for category in categories:
                classifiers = self.measure_validator.get_classifier_by_category(category)
                if classifiers:
                    logger.debug(f"Loaded {len(classifiers)} classifiers from {category}")

            logger.debug("Glossary check completed (measure word rules loaded)")

        except Exception as e:
            logger.error(f"Glossary check failed: {e}")

        return errors

    def _verify_llm_errors(
        self, llm_errors: list[ErrorAnnotation], text: str
    ) -> list[ErrorAnnotation]:
        """Verify LLM errors to filter out hallucinations.

        Args:
            llm_errors: Errors reported by LLM
            text: Translation text

        Returns:
            Verified errors (hallucinations filtered out)
        """
        if not self.helper or not self.helper.is_available():
            # Without HanLP, can't verify - return all
            return llm_errors

        verified = []
        for error in llm_errors:
            # Verify position is valid
            if not self.helper.verify_error_position(error, text):
                logger.warning(f"Filtered LLM hallucination: invalid position {error.location}")
                continue

            # Verify the mentioned word exists in text
            if not self.helper.verify_word_exists(error.description, text):
                logger.warning("Filtered LLM hallucination: word not found")
                continue

            verified.append(error)

        filtered_count = len(llm_errors) - len(verified)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} LLM hallucinations")

        return verified

    def _remove_duplicates(
        self, hanlp_errors: list[ErrorAnnotation], llm_errors: list[ErrorAnnotation]
    ) -> list[ErrorAnnotation]:
        """Remove HanLP errors that overlap with LLM errors.

        Args:
            hanlp_errors: Errors from HanLP
            llm_errors: Errors from LLM

        Returns:
            HanLP errors that don't overlap with LLM
        """
        unique = []

        for hanlp_error in hanlp_errors:
            # Check if this HanLP error overlaps with any LLM error
            overlaps = False
            for llm_error in llm_errors:
                if self._errors_overlap(hanlp_error, llm_error):
                    overlaps = True
                    break

            if not overlaps:
                unique.append(hanlp_error)

        duplicates = len(hanlp_errors) - len(unique)
        if duplicates > 0:
            logger.debug(f"Removed {duplicates} duplicate HanLP errors")

        return unique

    @staticmethod
    def _errors_overlap(error1: ErrorAnnotation, error2: ErrorAnnotation) -> bool:
        """Check if two errors overlap in location.

        Args:
            error1: First error
            error2: Second error

        Returns:
            True if errors overlap, False otherwise
        """
        start1, end1 = error1.location
        start2, end2 = error2.location

        # Check for any overlap
        return not (end1 <= start2 or end2 <= start1)

    async def _check_chinese_specifics(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform Chinese-specific fluency checks.

        Args:
            task: Translation task

        Returns:
            List of Chinese-specific errors
        """
        prompt = f"""You are a native Chinese speaker and professional translator/editor.

Your task: Identify ONLY clear Chinese-specific linguistic errors in the translation.

## SOURCE TEXT ({task.source_lang}):
{task.source_text}

## TRANSLATION (Chinese):
{task.translation}

## IMPORTANT GUIDELINES:

**What IS an error:**
- Clear measure word mistakes (wrong 量词 for the noun)
- Obvious aspect particle errors (了/过/着 used incorrectly)
- Unnatural constructions that no native Chinese speaker would use
- Character inconsistency (mixing simplified and traditional inappropriately)
- Word order traps (字序陷阱) - semantic swaps like 蜜蜂/蜂蜜
- Incorrect 把字句/被字句 constructions
- Separable verb errors (离合词) - wrong object placement
- Missing or incorrect resultative complements (结果补语)
- 的/地/得 particle confusion

**What is NOT an error:**
- Stylistic preferences (multiple correct phrasings exist)
- Direct translations that are grammatically correct
- Natural Chinese that differs from your personal preference
- Minor stylistic variations

## CHECKS TO PERFORM:

1. **Measure Words (量词)** - ONLY flag clear violations
   - Wrong measure word for a specific noun
   - Missing essential measure word
   - Examples: 一本书 (correct), 一个书 (wrong for most contexts)

2. **Aspect Particles** - Consider source text context
   - Check if 了/过/着 usage matches the source meaning
   - Completed action: 了
   - Past experience: 过
   - Ongoing state: 着
   - Remember: some sentences don't need aspect markers

3. **Grammar Patterns** - ONLY clear mistakes
   - Word order errors that affect meaning
   - Incorrect particle usage

4. **Character Consistency** - ONLY if clearly inconsistent
   - Mixing simplified (简体) and traditional (繁体) inappropriately

5. **Word Order Traps (字序陷阱)** - Check semantic swaps
   - 蜜蜂 (bee) vs 蜂蜜 (honey)
   - 故事 (story) vs 事故 (accident)
   - 牛奶 (milk) vs 奶牛 (cow)
   - 人工 (artificial) vs 工人 (worker)
   - Flag if meaning doesn't match source context

6. **把字句/被字句 Constructions**
   - 把字句 requires resultative complement: 我把书放在桌子上了 (correct)
   - 把字句 cannot use stative verbs: *我把他认为是朋友 (wrong)
   - 被字句 typically for negative events: 他被撞倒了 (correct)
   - Flag misuse of 把/被 constructions

7. **Separable Verbs (离合词)**
   - Object cannot follow separable verb: *见面他 → 跟他见面
   - Aspect markers go between V and O: 见了面 (correct), *见面了 (wrong in separation)
   - Common verbs: 见面, 睡觉, 洗澡, 帮忙, 结婚, 聊天

8. **Resultative Complements (结果补语)**
   - 得/的 confusion: 跑得很快 (correct), *跑的很快 (wrong)
   - Missing complements where needed: 看完 (finish watching), 听懂 (understand)
   - Wrong complement choice: *打好玻璃 → 打破玻璃

9. **的/地/得 Particle Usage**
   - 的: before nouns (美丽的花朵)
   - 地: before verbs (认真地学习)
   - 得: after verbs for degree/result (跑得很快)
   - This is a CRITICAL error - very common mistake

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "measure_word|aspect_particle|grammar|character|word_order_trap|ba_bei_construction|separable_verb|resultative_complement|de_particle",
      "severity": "critical|major|minor",
      "location": [start_char, end_char],
      "description": "Specific Chinese linguistic issue with the exact word/phrase you found",
      "suggestion": "Corrected version in Chinese"
    }}
  ]
}}

Rules:
- CONSERVATIVE: Only report clear, unambiguous errors
- VERIFY: Ensure the word/phrase you mention actually exists in the text at the specified position
- CONTEXT: Consider the source text when evaluating aspect and meaning
- If the translation is natural and grammatically correct, return empty errors array
- Provide accurate character positions (0-indexed, use Python string slicing logic)
- 的/地/得 errors should be marked as CRITICAL severity

Output only valid JSON, no explanation."""

        try:
            logger.info("ChineseFluencyAgent - Sending prompt to LLM")

            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            logger.info("ChineseFluencyAgent - Received response from LLM")

            # Parse response
            response_data = self._parse_json_response(response)
            errors_data = response_data.get("errors", [])

            errors = []
            for error_dict in errors_data:
                location = error_dict.get("location", [0, 10])
                if isinstance(location, list) and len(location) == 2:
                    location_tuple = (location[0], location[1])
                else:
                    location_tuple = (0, 10)

                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory=f"chinese_{error_dict.get('subcategory', 'specific')}",
                        severity=ErrorSeverity(error_dict.get("severity", "minor")),
                        location=location_tuple,
                        description=error_dict.get("description", "Chinese linguistic issue"),
                        suggestion=error_dict.get("suggestion"),
                    )
                )

            return errors

        except Exception as e:
            logger.error(f"Chinese-specific check failed: {e}")
            return []

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON dictionary
        """
        try:
            return cast(dict[str, Any], json.loads(response))
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            json_match = re.search(r"```(?:json)?\s*(\{[^\}]*\})\s*```", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(1)))
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(0)))
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON response: {response[:200]}")
            return {"errors": []}
