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

"""Russian language helper with pymorphy3 and razdel integration."""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation, ErrorSeverity

from .base import LanguageHelper, MorphologyInfo

logger = logging.getLogger(__name__)

# Try to import MAWO core (for morphology, tokenization, NER)
try:
    from mawo import Russian  # mawo-core package

    MAWO_AVAILABLE = True
    logger.info("Using MAWO core for Russian NLP (morphology, tokenization, NER)")
except ImportError as e:
    MAWO_AVAILABLE = False
    logger.warning(
        f"MAWO core not installed: {e}. "
        "RussianLanguageHelper will run in limited mode. "
        "Install with: pip install 'mawo-core[all]>=0.1.1'"
    )

# Try to import LanguageTool (for grammar checking)
try:
    import language_tool_python

    LANGUAGETOOL_AVAILABLE = True
    logger.info("LanguageTool available for Russian grammar checking (930+ rules)")
except ImportError:
    LANGUAGETOOL_AVAILABLE = False
    logger.warning(
        "LanguageTool not installed. "
        "RussianLanguageHelper will run without grammar checking. "
        "Install with: pip install language-tool-python"
    )


class RussianLanguageHelper(LanguageHelper):
    """Language helper for Russian with MAWO core + LanguageTool.

    Uses MAWO core for NLP features:
    - Morphological analysis with rich Document/Token objects
    - Russian tokenization and NER
    - Verb aspect detection
    - Adjective-noun agreement checking
    - Entity preservation validation
    - Anti-hallucination verification

    Uses LanguageTool for grammar checking:
    - 930+ Russian grammar rules
    - Case agreement validation
    - Spelling and punctuation checks
    - Style and typography checks

    Example:
        >>> helper = RussianLanguageHelper()
        >>> if helper.is_available():
        ...     errors = helper.check_grammar("Я пошел в магазин")
        ...     print(errors[0].description if errors else "No errors")
    """

    # IT terminology whitelist - words that should not be flagged as spelling errors
    IT_TERMS_WHITELIST: set[str] = {
        # CLI and shell terms
        "демо",
        "демо-режим",
        "воркер",
        "воркеров",
        "воркеры",
        "бенчмарк",
        "бенчмаркинг",
        "хотфикс",
        "фикс",
        "коммит",
        "коммитить",
        "закоммитить",
        "пуш",
        "пушить",
        "пулл",
        "пуллить",
        "мёрдж",
        "мёрджить",
        "ребейз",
        "чекаут",
        "стеш",
        "стешить",
        "клон",
        "клонировать",
        # Development process
        "спринт",
        "скрам",
        "аджайл",
        "канбан",
        "бэклог",
        "стендап",
        "ретро",
        "ретроспектива",
        "дейлик",
        "деплой",
        "деплоить",
        "релиз",
        "релизить",
        "прод",
        "продакшн",
        "стейджинг",
        "девелопмент",
        "дебаг",
        "дебажить",
        "дебаггер",
        "логгирование",
        "рефакторинг",
        "рефакторить",
        "код-ревью",
        # Architecture
        "микросервис",
        "микросервисы",
        "монолит",
        "контейнер",
        "контейнеризация",
        "докер",
        "докеризация",
        "оркестрация",
        "кубернетес",
        "инстанс",
        "инстансы",
        "нода",
        "ноды",
        "кластер",
        "кластеры",
        "реплика",
        "шардинг",
        "партиционирование",
        "балансировщик",
        # Data and storage
        "постгрес",
        "монго",
        "редис",
        "мемкеш",
        "кэш",
        "кэширование",
        "бэкап",
        "бэкапить",
        "дамп",
        "дампить",
        "миграция",
        "сид",
        # API and integration
        "эндпоинт",
        "эндпоинты",
        "рест",
        "веб-хук",
        "вебхук",
        "колбэк",
        "токен",
        "токены",
        "пейлоад",
        "хедер",
        "хедеры",
        # Testing
        "тест",
        "тесты",
        "юнит-тест",
        "юнит-тесты",
        "мок",
        "моки",
        "мокать",
        "стаб",
        "стабы",
        "фикстура",
        "фикстуры",
        "ассерт",
        "ассерты",
        # ML/AI
        "нейросеть",
        "нейросети",
        "промпт",
        "промпты",
        "токенизация",
        "токенизатор",
        "эмбеддинг",
        "эмбеддинги",
        "файнтюнинг",
        "инференс",
        "батч",
        "батчи",
        "батчинг",
        # Code quality
        "линтер",
        "линтинг",
        "форматтер",
        "типизация",
        "аннотация",
        "код-стайл",
        "конвенция",
    }

    def __init__(self) -> None:
        """Initialize Russian language helper."""
        self._nlp: Any = None
        self._language_tool: Any = None
        self._initialized = False
        self._lt_available = False

        # Initialize MAWO core for NLP features
        if MAWO_AVAILABLE:
            try:
                self._nlp = Russian()
                self._initialized = True
                logger.info("RussianLanguageHelper initialized with MAWO core (NLP features)")
            except Exception as e:
                logger.error(f"Failed to initialize MAWO core: {e}")
                self._initialized = False
        else:
            logger.info("RussianLanguageHelper running in limited mode (no MAWO core)")

        # Initialize LanguageTool for grammar checking
        if LANGUAGETOOL_AVAILABLE:
            try:
                self._language_tool = language_tool_python.LanguageTool("ru")
                self._lt_available = True
                logger.info("LanguageTool initialized successfully (930+ Russian grammar rules)")
            except Exception as e:
                logger.warning(f"LanguageTool initialization failed: {e}")
                self._lt_available = False

    @property
    def language_code(self) -> str:
        """Get language code."""
        return "ru"

    def is_available(self) -> bool:
        """Check if NLP dependencies are available."""
        return self._initialized and MAWO_AVAILABLE

    def verify_word_exists(self, word: str, text: str) -> bool:
        """Verify word exists in text (anti-hallucination).

        Args:
            word: Word to search for
            text: Text to search in

        Returns:
            True if word found, False if not (LLM hallucination)
        """
        if not self.is_available():
            # Fallback: simple case-insensitive search
            return word.lower() in text.lower()

        # Use proper tokenization
        tokens = self.tokenize(text)
        word_lower = word.lower()
        return any(token[0].lower() == word_lower for token in tokens)

    def verify_error_position(self, error: ErrorAnnotation, text: str) -> bool:
        """Verify error position is valid.

        Args:
            error: Error with location field
            text: Full text

        Returns:
            True if position valid, False otherwise
        """
        start, end = error.location

        # Check bounds
        if start < 0 or end > len(text) or start >= end:
            return False

        # Extract text at position
        substring = text[start:end]

        # Check if it's not empty
        if not substring.strip():
            return False

        # If error mentions specific word, verify it exists in substring
        if hasattr(error, "description"):
            # Try to extract quoted words from description
            import re

            quoted_words = re.findall(r"'([^']+)'|\"([^\"]+)\"", error.description)
            if quoted_words:
                for word_tuple in quoted_words:
                    word = word_tuple[0] or word_tuple[1]
                    if word and word not in substring:
                        logger.warning(
                            f"Error mentions '{word}' but it's not at position {start}:{end}"
                        )
                        return False

        return True

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Tokenize Russian text with accurate positions using MAWO core.

        Args:
            text: Text to tokenize

        Returns:
            List of (word, start, end) tuples
        """
        if not self.is_available():
            # Fallback: simple split
            tokens = []
            start = 0
            for word in text.split():
                idx = text.find(word, start)
                if idx != -1:
                    tokens.append((word, idx, idx + len(word)))
                    start = idx + len(word)
            return tokens

        # Use MAWO core for proper Russian tokenization
        doc = self._nlp(text)
        return [(token.text, token.start, token.end) for token in doc.tokens]

    def analyze_morphology(self, text: str) -> list[MorphologyInfo]:
        """Analyze morphology of all words using MAWO core.

        MAWO core provides automatic context-aware disambiguation with:
        - POS disambiguation for function words
        - Adjective-noun agreement matching
        - Preposition-driven case selection
        - Custom vocabulary support

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo objects with disambiguated tags
        """
        if not self.is_available():
            return []

        # Use MAWO core for morphological analysis
        doc = self._nlp(text)
        results: list[MorphologyInfo] = []

        for token in doc.tokens:
            results.append(
                MorphologyInfo(
                    word=token.text,
                    pos=token.pos,
                    gender=token.gender,
                    case=token.case,
                    number=token.number,
                    aspect=token.aspect,
                    start=token.start,
                    stop=token.end,
                )
            )

        return results

    def _should_skip_lt_match(self, text: str, match: Any) -> bool:
        """Check if a LanguageTool match should be skipped.

        Args:
            text: Original text
            match: LanguageTool match object

        Returns:
            True if match should be skipped
        """
        if not self._is_translation_relevant(match):
            return True

        if self._is_csv_or_code_context(text, match):
            logger.debug(f"Skipping CSV/code context: {match.ruleId}")
            return True

        error_word = text[match.offset : match.offset + match.errorLength].lower()
        if self._is_it_term(error_word):
            logger.debug(f"Skipping IT term: {error_word}")
            return True

        return False

    def _match_to_error(self, match: Any) -> ErrorAnnotation:
        """Convert LanguageTool match to ErrorAnnotation.

        Args:
            match: LanguageTool match object

        Returns:
            ErrorAnnotation instance
        """
        return ErrorAnnotation(
            category="fluency",
            subcategory=f"russian_{match.ruleId}",
            severity=self._map_severity(match),
            location=(match.offset, match.offset + match.errorLength),
            description=match.message,
            suggestion=match.replacements[0] if match.replacements else None,
        )

    def _run_languagetool_checks(self, text: str) -> list[ErrorAnnotation]:
        """Run LanguageTool grammar checks.

        Args:
            text: Text to check

        Returns:
            List of errors found
        """
        errors = []
        try:
            matches = self._language_tool.check(text)
            for match in matches:
                if not self._should_skip_lt_match(text, match):
                    errors.append(self._match_to_error(match))
            logger.debug(f"LanguageTool found {len(errors)} grammar errors")
        except Exception as e:
            logger.error(f"LanguageTool check failed: {e}")
        return errors

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check Russian grammar using LanguageTool + custom rules.

        Hybrid approach:
        1. LanguageTool - 930+ rules for spelling, punctuation, style
        2. Custom rules - Adjective-noun agreement that LanguageTool misses

        The grammar checker validates:
        - Grammar (грамматика)
        - Spelling (орфография)
        - Punctuation (пунктуация)
        - Style (стиль)
        - Typography (типографика)
        - Adjective-noun gender/case/number agreement (custom)
        - And 925+ more rules

        Args:
            text: Russian text to check

        Returns:
            List of detected grammar errors
        """
        errors = []

        if self._lt_available:
            errors.extend(self._run_languagetool_checks(text))

        if self.is_available():
            try:
                custom_errors = self._check_adjective_noun_agreement(text)
                errors.extend(custom_errors)
                logger.debug(f"Custom rules found {len(custom_errors)} additional errors")
            except Exception as e:
                logger.error(f"Custom agreement checks failed: {e}")

        return self._deduplicate_errors(errors)

    def _deduplicate_errors(self, errors: list[ErrorAnnotation]) -> list[ErrorAnnotation]:
        """Remove duplicate errors at the same location.

        Keeps the error with higher severity, or the first one if severity is equal.

        Args:
            errors: List of errors to deduplicate

        Returns:
            Deduplicated list of errors
        """
        if not errors:
            return errors

        # Group by location
        by_location: dict[tuple[int, int], list[ErrorAnnotation]] = {}
        for error in errors:
            key = error.location
            if key not in by_location:
                by_location[key] = []
            by_location[key].append(error)

        # Keep best error per location
        severity_order = {
            ErrorSeverity.CRITICAL: 3,
            ErrorSeverity.MAJOR: 2,
            ErrorSeverity.MINOR: 1,
        }

        deduplicated = []
        for _location, group in by_location.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Sort by severity (highest first)
                group.sort(key=lambda e: severity_order.get(e.severity, 0), reverse=True)
                deduplicated.append(group[0])

        return deduplicated

    # Words to skip in agreement checking (numerals, demonstrative pronouns)
    _AGREEMENT_SKIP_WORDS = frozenset(
        [
            "один",
            "одна",
            "одно",
            "этот",
            "эта",
            "это",
            "тот",
            "та",
            "то",
            "весь",
            "вся",
            "всё",
        ]
    )

    @staticmethod
    def _find_next_noun(tokens: list[Any], start_idx: int) -> Any | None:
        """Find the next noun token after an adjective.

        Args:
            tokens: List of tokens
            start_idx: Index to start searching from

        Returns:
            Noun token or None if not found
        """
        j = start_idx
        while j < len(tokens):
            if tokens[j].pos in ["ADJF", "ADJS"]:
                j += 1
                continue
            if tokens[j].pos == "NOUN":
                return tokens[j]
            break
        return None

    def _is_agreement_false_positive(
        self, adj_token: Any, noun_token: Any, adj_gender: str, noun_gender: str
    ) -> bool:
        """Check if a gender mismatch is a false positive.

        Args:
            adj_token: Adjective token
            noun_token: Noun token
            adj_gender: Adjective gender
            noun_gender: Noun gender

        Returns:
            True if this should be skipped as false positive
        """
        adj_text = adj_token.text.lower()
        if adj_text in self._AGREEMENT_SKIP_WORDS:
            return True

        # Skip genitive case - masculine and neuter have identical endings
        adj_case = getattr(adj_token, "case", None)
        noun_case = getattr(noun_token, "case", None)
        if adj_case in ["gent", "gen"] or noun_case in ["gent", "gen"]:
            if {adj_gender, noun_gender} == {"masc", "neut"}:
                return True

        return False

    def _check_adjective_noun_agreement(self, text: str) -> list[ErrorAnnotation]:
        """Check adjective-noun agreement using MAWO core morphology.

        This catches errors that LanguageTool misses, like:
        - "Быстрый лиса" (masc adjective + fem noun)
        - "Красивая дом" (fem adjective + masc noun)

        Args:
            text: Russian text to check

        Returns:
            List of agreement errors
        """
        if not self.is_available():
            return []

        errors = []
        doc = self._nlp(text)
        tokens = list(doc.tokens)

        for i, token in enumerate(tokens):
            if token.pos not in ["ADJF", "ADJS"] or not token.gender:
                continue

            next_noun = self._find_next_noun(tokens, i + 1)
            if not next_noun or not next_noun.gender:
                continue

            if token.gender == next_noun.gender:
                continue

            if self._is_agreement_false_positive(token, next_noun, token.gender, next_noun.gender):
                continue

            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="russian_adj_noun_gender_agreement",
                    severity=ErrorSeverity.MAJOR,
                    location=(token.start, next_noun.end),
                    description=(
                        f"Adjective-noun gender mismatch: '{token.text}' is {token.gender}, "
                        f"but '{next_noun.text}' is {next_noun.gender}"
                    ),
                    suggestion=None,
                )
            )

        return errors

    def _map_severity(self, match: Any) -> ErrorSeverity:
        """Map LanguageTool match to ErrorSeverity.

        Args:
            match: LanguageTool Match object

        Returns:
            ErrorSeverity enum value
        """
        rule_id = match.ruleId.lower()

        # Critical errors (spelling, clear grammar mistakes)
        if any(pattern in rule_id for pattern in ["spelling", "typo", "misspell", "орфография"]):
            return ErrorSeverity.CRITICAL

        # Major errors (agreement, verb form, tense)
        if any(
            pattern in rule_id
            for pattern in [
                "grammar",
                "agreement",
                "verb",
                "case",
                "грамматика",
                "согласование",
            ]
        ):
            return ErrorSeverity.MAJOR

        # Minor errors (everything else)
        return ErrorSeverity.MINOR

    def _is_translation_relevant(self, match: Any) -> bool:
        """Filter out style-only suggestions not relevant for translation QA.

        Args:
            match: LanguageTool Match object

        Returns:
            True if error is relevant for translation, False otherwise
        """
        rule_id = match.ruleId.lower()

        # Exclude pure style suggestions and known false positives
        exclude_patterns = [
            "style",
            "redundancy",
            "collocation",
            "cliche",
            "wordiness",
            "comma_before_kak",  # Often false positive in conversational phrases
        ]

        if any(pattern in rule_id for pattern in exclude_patterns):
            return False

        return True

    def _is_csv_or_code_context(self, text: str, match: Any) -> bool:
        """Check if match is in CSV or code context (to avoid false positives).

        CSV format (RFC 4180) does not use spaces after commas.
        Code blocks also have different punctuation rules.

        Rules filtered:
        - COMMA_PARENTHESIS_WHITESPACE: "Put a space after comma"

        Args:
            text: Full text being checked
            match: LanguageTool Match object

        Returns:
            True if match should be skipped (is in CSV/code context)
        """
        import re

        rule_id = match.ruleId.upper()

        # Only filter specific punctuation rules
        punctuation_rules = [
            "COMMA_PARENTHESIS_WHITESPACE",
            "WHITESPACE_RULE",
        ]

        if rule_id not in punctuation_rules:
            return False

        # Get context around the match (100 chars before and after)
        start = max(0, match.offset - 100)
        end = min(len(text), match.offset + match.errorLength + 100)
        context = text[start:end]

        # CSV patterns (comma-separated values without spaces)
        csv_patterns = [
            r"\w+,\w+,\w+",  # word,word,word (3+ comma-separated)
            r"\.csv|\.tsv",  # file extensions
            r"source,|target,|lang,|translation,",  # common CSV headers
            r"\w+_\w+,\w+_\w+",  # snake_case,snake_case
        ]

        # Code block patterns
        code_patterns = [
            r"```",  # Markdown code fence
            r"^\s{4,}\S",  # Indented code (4+ spaces)
            r"--\w+",  # CLI options
            r"\$\w+|%\w+",  # Variables
            r"import\s|from\s|def\s|class\s",  # Python keywords
        ]

        # Check CSV patterns
        for pattern in csv_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True

        # Check code patterns
        for pattern in code_patterns:
            if re.search(pattern, context, re.MULTILINE):
                return True

        return False

    def _is_it_term(self, word: str) -> bool:
        """Check if a word is a known IT term that should not be flagged.

        Args:
            word: Word to check (lowercase)

        Returns:
            True if word is in IT terminology whitelist
        """
        # Clean word from punctuation
        clean_word = word.strip(".,;:!?()[]{}\"'«»—-").lower()

        # Direct match
        if clean_word in self.IT_TERMS_WHITELIST:
            return True

        # Check if word starts with known IT prefix
        it_prefixes = [
            "демо-",
            "код-",
            "веб-",
            "юнит-",
            "бэк-",
            "фронт-",
            "микро-",
            "мульти-",
            "авто-",
        ]
        for prefix in it_prefixes:
            if clean_word.startswith(prefix):
                return True

        # Check if word is likely an English loanword (ends with common suffixes)
        if any(clean_word.endswith(suffix) for suffix in ["инг", "ить", "ер", "мент"]):
            # Could be IT term, but also check if it looks like technical jargon
            if any(root in clean_word for root in ["дебаг", "рефактор", "деплой", "релиз"]):
                return True

        return False

    def get_enrichment_data(self, text: str) -> dict[str, Any]:
        """Get comprehensive morphological data using MAWO core.

        Provides detailed linguistic context to help LLM make better decisions:
        - Verb aspects (perfective/imperfective) with aspect pairs
        - Adjective-noun pairs with automatic agreement checking
        - Gender, case, number information
        - POS distribution

        Args:
            text: Text to analyze

        Returns:
            Dictionary with morphological insights for LLM
        """
        if not self.is_available():
            return {"has_morphology": False}

        # Use MAWO core document for rich linguistic features
        doc = self._nlp(text)

        # Extract verb aspects using doc.verbs
        verb_aspects = {}
        verbs_list = []
        for verb in doc.verbs:
            verb_aspects[verb.text] = {
                "aspect": verb.aspect,
                "aspect_name": "perfective" if verb.aspect == "perf" else "imperfective",
                "position": f"{verb.start}-{verb.end}",
            }
            verbs_list.append(verb.text)

        # Extract adjective-noun pairs using doc.adjective_noun_pairs
        adj_noun_pairs = []
        for pair in doc.adjective_noun_pairs:
            pair_info = {
                "adjective": {
                    "word": pair.adjective.text,
                    "gender": pair.adjective.gender,
                    "case": pair.adjective.case,
                    "number": pair.adjective.number,
                },
                "noun": {
                    "word": pair.noun.text,
                    "gender": pair.noun.gender,
                    "case": pair.noun.case,
                    "number": pair.noun.number,
                },
                "agreement": pair.agreement,  # Returns "correct" or "mismatch"
            }
            adj_noun_pairs.append(pair_info)

        # Count parts of speech
        pos_counts: dict[str, int] = {}
        for token in doc.tokens:
            if token.pos:
                pos_counts[token.pos] = pos_counts.get(token.pos, 0) + 1

        return {
            "has_morphology": True,
            "word_count": len(doc.tokens),
            "verb_aspects": verb_aspects,
            "verbs_found": verbs_list,
            "adjective_noun_pairs": adj_noun_pairs,
            "pos_distribution": pos_counts,
        }

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from Russian text using MAWO core.

        Args:
            text: Text to extract entities from

        Returns:
            List of entities with type, text, and position
        """
        if not self.is_available():
            logger.debug("MAWO core not available, returning empty list")
            return []

        try:
            # Use MAWO core for entity extraction
            doc = self._nlp(text)

            # Convert entities to our format
            entities = []
            for entity in doc.entities:
                entities.append(
                    {
                        "text": entity.text,
                        "type": entity.label,
                        "start": entity.start,
                        "stop": entity.end,
                    }
                )

            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

    def check_entity_preservation(
        self, source_text: str, translation_text: str
    ) -> list[ErrorAnnotation]:
        """Check if named entities from source are preserved in translation using MAWO core.

        Args:
            source_text: Original text (may be in any language)
            translation_text: Russian translation

        Returns:
            List of errors for missing/mismatched entities
        """
        if not self.is_available():
            logger.debug("MAWO core not available, skipping entity preservation check")
            return []

        try:
            # Use MAWO core for cross-document entity matching
            source_doc = self._nlp(source_text)
            translation_doc = self._nlp(translation_text)

            # Check entity preservation using match_entities
            matches = self._nlp.match_entities(source_doc, translation_doc)

            errors = []

            # Check for missing entities
            source_entities = list(source_doc.entities)
            matched_source_entities = {match.source for match in matches}

            missing_entities = [
                entity for entity in source_entities if entity not in matched_source_entities
            ]

            if missing_entities:
                for entity in missing_entities[:3]:  # Limit to first 3 to avoid spam
                    errors.append(
                        ErrorAnnotation(
                            category="accuracy",
                            subcategory="entity_omission",
                            severity=ErrorSeverity.MAJOR,
                            location=(0, min(50, len(translation_text))),
                            description=(
                                f"Entity '{entity.text}' ({entity.label}) from source "
                                f"appears to be missing in translation"
                            ),
                            suggestion="Verify that all proper nouns are correctly translated",
                        )
                    )

            logger.debug(
                f"Entity preservation check: "
                f"source_entities={len(source_entities)}, "
                f"translation_entities={len(list(translation_doc.entities))}, "
                f"matches={len(matches)}, "
                f"missing={len(missing_entities)}"
            )

            return errors

        except Exception as e:
            logger.error(f"Entity preservation check failed: {e}")
            # Fallback to basic check
            return self._basic_entity_check(source_text, translation_text)

    def _basic_entity_check(self, source_text: str, translation_text: str) -> list[ErrorAnnotation]:
        """Basic entity preservation check using regex (fallback method).

        Args:
            source_text: Original text
            translation_text: Russian translation

        Returns:
            List of errors for potential entity issues
        """
        import re

        errors = []

        # Find capitalized sequences in source (potential entity names)
        source_caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", source_text)
        translation_entities = self.extract_entities(translation_text)

        if len(source_caps) > 0 and len(translation_entities) == 0:
            errors.append(
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="entity_omission",
                    severity=ErrorSeverity.MAJOR,
                    location=(0, min(50, len(translation_text))),
                    description=(
                        f"Source text contains {len(source_caps)} potential entities "
                        f"but translation has no named entities detected"
                    ),
                    suggestion="Verify that proper nouns are correctly translated",
                )
            )

        return errors
