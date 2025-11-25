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

"""StyleFingerprint analyzer for automatic style detection.

Analyzes source texts to detect stylistic features and deviations,
enabling style-aware translation quality evaluation without manual
author profiles.

Based on:
- Delta method (Burrows, 2002) for stylometry
- Russian stylometry research (Lagutina et al.)
- LiTransProQA (Zhang et al., 2025) for literary evaluation
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

from .models import (
    StyleDeviation,
    StyleDeviationType,
    StylePattern,
    StyleProfile,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StyleFingerprint:
    """Automatic stylistic fingerprint analyzer.

    Analyzes text to create a style profile without requiring
    knowledge of the author. Works on any text in supported languages.

    Example:
        >>> fingerprint = StyleFingerprint()
        >>> profile = fingerprint.analyze(
        ...     "Душа его желала жить жизнью...",
        ...     lang="ru"
        ... )
        >>> print(f"Deviation score: {profile.deviation_score}")
        >>> if profile.has_significant_deviations:
        ...     print("Literary text with stylistic deviations detected")
    """

    # Pleonasm patterns (redundant phrases) - Platanov style
    PLEONASM_PATTERNS_RU = [
        r"\b(\w+)\s+\1\b",  # Word repetition
        r"жить\s+жизн",  # "live life"
        r"думать\s+дум",  # "think thought"
        r"петь\s+песн",  # "sing song"
        r"видеть\s+вид",  # "see sight"
        r"душ[аеуой]+\s+душ",  # "soul of soul"
        r"своими?\s+собственн",  # "own own"
        r"впервые\s+первый",  # "first time first"
        r"полностью\s+весь",  # "completely all"
        # Additional Platanov-style patterns
        r"горе\s+горе",  # "grief grief"
        r"слова[мих]*\s+слов",  # "words of words"
        r"знать\s+не\s+знаю.*ведать\s+не\s+ведаю",  # Folk formula
        r"путь\s+путин",  # "path of path"
        r"дело\s+дела",  # "deed of deed"
        r"слышать\s+слух",  # "hear hearing"
        r"говорить\s+говор",  # "speak speech"
    ]

    PLEONASM_PATTERNS_EN = [
        r"\b(\w+)\s+\1\b",  # Word repetition
        r"free\s+gift",
        r"true\s+fact",
        r"past\s+history",
        r"future\s+plans",
        r"completely\s+finished",
        r"basic\s+fundamentals",
        # Additional English pleonasms
        r"end\s+result",
        r"advance\s+warning",
        r"each\s+and\s+every",
        r"first\s+and\s+foremost",
        r"one\s+and\s+only",
        r"if\s+and\s+when",
        r"null\s+and\s+void",
        r"any\s+and\s+all",
    ]

    # Inversion patterns (non-standard word order)
    INVERSION_PATTERNS_RU = [
        r"[а-яё]+(?:ого|его|ую|юю)\s+[а-яё]+(?:ом|ем|ой|ей)\s+(?:был[аои]?|стал[аои]?)",
        # Adjective after noun patterns
        r"\b(?:человек|люди|мир|жизнь|душа)\s+(?:этот|эта|это|этой|этому)",
    ]

    # Folk speech markers (Russian) - Leskov-style skaz
    FOLK_SPEECH_RU = [
        r"\bбатюшк",
        r"\bматушк",
        r"\bголубчик",
        r"\bродимы[йя]",
        r"\bкормил",
        r"\bпоил",
        r"\bавось",
        r"\bнебось",
        r"\bчай\b",  # in meaning "probably"
        r"\bвишь\b",
        r"\bглядь\b",
        r"\bбишь\b",
        r"\bвона\b",
        r"\bтутось",
        r"\bтамось",
        r"\bниже\s+спин",
        # Leskov-style errative (folk etymology)
        r"\bбуреметр",  # from "barometr"
        r"\bажидац",  # folk form of "agitation"
        r"\bклеветон",  # folk form of "feuilleton"
        r"\bмелкоскоп",  # folk form of "microscope"
        r"\bнимфозор",  # folk form
        r"\bдолбиц",  # folk form
        r"\bпубель",  # folk form of "pudel"
        # Additional skaz markers
        r"\bстало\s+быть",
        r"\bдескать",
        r"\bмол\b",
        r"\bде\b",
        r"\bишь\s+ты",
        r"\bвот\s+те\s+на",
        r"\bах\s+ты",
        r"\bох\s+ты",
        r"\bну\s+и\s+ну",
        r"\bгосподи\s+помилуй",
        r"\bцарство\s+небесное",
        # Dialectal forms
        r"\bничаво",
        r"\bчаво",
        r"\bнынче",
        r"\bнонче",
        r"\bдавеча",
        r"\bпрежде\s+всего",
    ]

    # Errative patterns (Leskov-style intentional word distortion)
    ERRATIVE_PATTERNS_RU = [
        r"\w+метр\b",  # buremeter, termometer
        r"\w+скоп\b",  # melkoskop, teleskop
        r"\w+фон\b",  # telefon variants
        r"[а-я]+озор\b",  # nimfozor
    ]

    # Stream of consciousness markers
    STREAM_MARKERS = [
        r"\.{3,}",  # Multiple dots
        r"—\s*—",  # Multiple dashes
        r"\?\s*\?",  # Multiple question marks
        r"[,;]\s*[,;]",  # Repeated punctuation
    ]

    # Literary keywords indicating creative writing
    LITERARY_KEYWORDS_RU = {
        "душа",
        "сердце",
        "любовь",
        "судьба",
        "мечта",
        "тоска",
        "печаль",
        "радость",
        "страдание",
        "смерть",
        "жизнь",
        "вечность",
        "мгновение",
        "тишина",
        "молчание",
        # Silver Age poetry vocabulary
        "бездна",  # abyss
        "мрак",  # darkness
        "свет",  # light
        "тень",  # shadow
        "заря",  # dawn
        "закат",  # sunset
        "луна",  # moon
        "звезда",  # star
        "небо",  # sky
        "ночь",  # night
        "туман",  # fog
        "роса",  # dew
        "слеза",  # tear
        # Dostoevsky psychological vocabulary
        "бунт",  # rebellion
        "грех",  # sin
        "искупление",  # redemption
        "унижение",  # humiliation
        "страсть",  # passion
        "безумие",  # madness
        "отчаяние",  # despair
        "раскаяние",  # repentance
        "совесть",  # conscience
        # Tolstoy spiritual vocabulary
        "истина",  # truth
        "добро",  # good
        "красота",  # beauty
        "смысл",  # meaning
        "вера",  # faith
        "надежда",  # hope
        # Chekhov mood words
        "скука",  # boredom
        "усталость",  # fatigue
        "одиночество",  # loneliness
        "меланхолия",  # melancholy
        "ностальгия",  # nostalgia
    }

    # Russian function words for stylometric analysis
    FUNCTION_WORDS_RU = {
        "и",
        "в",
        "не",
        "на",
        "с",
        "что",
        "а",
        "как",
        "это",
        "он",
        "к",
        "по",
        "но",
        "из",
        "у",
        "за",
        "от",
        "о",
        "до",
        "же",
        "то",
        "все",
        "так",
        "его",
        "только",
        "она",
        "еще",
        "бы",
        "мне",
        "уже",
        "вот",
        "или",
        "ни",
        "быть",
        "был",
        "даже",
        "для",
        "без",
        "вы",
        "себя",
        "когда",
        "чтобы",
        "там",
        "потом",
        "теперь",
        "где",
        "здесь",
        "тут",
        "очень",
        "под",
    }

    LITERARY_KEYWORDS_EN = {
        "soul",
        "heart",
        "love",
        "fate",
        "destiny",
        "dream",
        "longing",
        "sorrow",
        "joy",
        "suffering",
        "death",
        "life",
        "eternity",
        "moment",
        "silence",
        # Romantic/Victorian literary vocabulary
        "melancholy",
        "despair",
        "anguish",
        "rapture",
        "sublime",
        "abyss",
        "twilight",
        "dawn",
        "moonlight",
        "shadows",
        "tempest",
        "passion",
        "devotion",
        "yearning",
        "solitude",
        "reverie",
        "elegy",
        "lament",
        "woe",
        # Modernist vocabulary (Joyce, Woolf, Faulkner)
        "consciousness",
        "stream",
        "epiphany",
        "flux",
        "void",
        "absurd",
        "alienation",
        "fragmentation",
        "memory",
        "perception",
        "interiority",
    }

    # English function words for Burrows Delta analysis
    FUNCTION_WORDS_EN = {
        "the",
        "and",
        "to",
        "of",
        "a",
        "in",
        "that",
        "it",
        "is",
        "was",
        "for",
        "on",
        "are",
        "as",
        "with",
        "his",
        "they",
        "be",
        "at",
        "one",
        "have",
        "this",
        "from",
        "by",
        "not",
        "but",
        "what",
        "all",
        "were",
        "we",
        "when",
        "your",
        "can",
        "said",
        "there",
        "use",
        "an",
        "each",
        "which",
        "she",
        "do",
        "how",
        "their",
        "if",
        "will",
        "up",
        "other",
        "about",
        "out",
        "many",
        "then",
        "them",
        "these",
        "so",
        "some",
        "her",
        "would",
        "make",
        "like",
        "him",
        "into",
        "time",
        "has",
        "look",
        "two",
        "more",
        "go",
        "see",
        "no",
        "way",
        "could",
        "my",
        "than",
        "been",
        "call",
        "who",
        "its",
        "now",
        "find",
        "long",
    }

    # English stream of consciousness patterns (Joyce, Woolf style)
    STREAM_PATTERNS_EN = [
        r"and\s+yes.*yes",  # Joyce "yes and yes"
        r"\.\.\.\s*\.\.\.",  # multiple ellipses
        r"I\s+thought\s+.*\s+I\s+thought",  # thought repetition
        r"—\s*.*\s*—",  # em-dash interruptions
    ]

    # English Victorian/formal register markers
    FORMAL_MARKERS_EN = [
        r"\b(?:hitherto|heretofore|wherefore|whereupon|henceforth)\b",
        r"\b(?:notwithstanding|nevertheless|furthermore|moreover)\b",
        r"\b(?:whilst|amongst|amidst)\b",
        r"\b(?:shall|ought|must needs)\b",
    ]

    # Chinese literary keywords (Lu Xun style, classical poetry)
    LITERARY_KEYWORDS_ZH = {
        "灵魂",  # soul
        "心",  # heart
        "爱",  # love
        "命运",  # fate
        "梦",  # dream
        "悲",  # sorrow
        "喜",  # joy
        "生",  # life
        "死",  # death
        "永恒",  # eternity
        "月",  # moon (classical poetry)
        "风",  # wind
        "花",  # flower
        "雪",  # snow
        "酒",  # wine (Li Bai)
        "山",  # mountain
        "水",  # water
        "天",  # sky/heaven
        "情",  # emotion/love
        "思",  # thought/longing
        # Additional classical imagery (意象)
        "霜",  # frost
        "露",  # dew
        "云",  # cloud
        "烟",  # mist/smoke
        "柳",  # willow
        "梅",  # plum blossom
        "松",  # pine
        "竹",  # bamboo
        "菊",  # chrysanthemum
        "兰",  # orchid
        "鹤",  # crane
        "燕",  # swallow
        "鸿雁",  # wild goose (longing)
        "杜鹃",  # cuckoo (sorrow)
        "夜",  # night
        "秋",  # autumn
        "春",  # spring
        "残",  # withered/remaining
        "落",  # falling
        "断肠",  # heartbroken
        "相思",  # longing
        "愁",  # melancholy
        "泪",  # tears
        "魂",  # spirit
        # Lu Xun modernist vocabulary
        "铁屋子",  # iron house
        "呐喊",  # outcry
        "彷徨",  # wandering
        "狂人",  # madman
        "阿Q",  # Ah Q
    }

    # Chinese classical poetry structural patterns
    CLASSICAL_PATTERNS_ZH = [
        r"[\u4e00-\u9fff]{5}[，。][\u4e00-\u9fff]{5}[。]",  # 5-char regulated verse
        r"[\u4e00-\u9fff]{7}[，。][\u4e00-\u9fff]{7}[。]",  # 7-char regulated verse
        r"[\u4e00-\u9fff]{4}[，][\u4e00-\u9fff]{4}[。]",  # 4-char classical
    ]

    # Chinese four-character idioms (成语) - literary marker
    CHENGYU_PATTERNS = [
        r"[\u4e00-\u9fff]{4}",  # Four-character compounds
    ]

    # Hindi literary keywords (Premchand style, Chhayavad poetry)
    LITERARY_KEYWORDS_HI = {
        "आत्मा",  # soul
        "हृदय",  # heart
        "प्रेम",  # love
        "भाग्य",  # fate
        "सपना",  # dream
        "दुख",  # sorrow
        "सुख",  # joy
        "जीवन",  # life
        "मृत्यु",  # death
        "अमर",  # immortal
        "चाँद",  # moon
        "प्रकृति",  # nature
        "किसान",  # farmer (Premchand)
        "गरीब",  # poor
        "समाज",  # society
        "न्याय",  # justice
        # Chhayavad poetry additions
        "छाया",  # shadow
        "नीरव",  # silent
        "विरह",  # separation
        "मिलन",  # union
        "यौवन",  # youth
        "सौंदर्य",  # beauty
        "रहस्य",  # mystery
        "स्वप्न",  # dream (Sanskrit)
        "वेदना",  # anguish
        "पीड़ा",  # pain
        "आँसू",  # tears
        "उदासी",  # sadness
        "तड़प",  # longing
        "बादल",  # cloud
        "बरसात",  # rain
        "सांझ",  # evening
        "प्रभात",  # dawn
        # Premchand social realism
        "जमींदार",  # landlord
        "मजदूर",  # laborer
        "दलित",  # oppressed
        "शोषण",  # exploitation
        "क्रांति",  # revolution
        "स्वतंत्रता",  # freedom
        "आजादी",  # independence
        "गाँव",  # village
        "खेत",  # field
        "झोपड़ी",  # hut
    }

    # Hindi Chhayavad style patterns
    CHHAYAVAD_PATTERNS_HI = [
        r"[\u0900-\u097F]+\s+की\s+छाया",  # shadow of X
        r"मन\s+[\u0900-\u097F]+",  # mind + noun
        r"[\u0900-\u097F]+\s+में\s+खो",  # lost in X
    ]

    # Hindi idioms and proverbs (Premchand style)
    HINDI_IDIOMS = [
        r"आँखों\s+का\s+तारा",  # apple of one's eye
        r"दिल\s+का\s+दर्द",  # heartache
        r"पत्थर\s+का\s+दिल",  # heart of stone
    ]

    # Persian literary keywords (Hedayat style, classical poetry)
    LITERARY_KEYWORDS_FA = {
        "روح",  # soul
        "دل",  # heart
        "عشق",  # love
        "سرنوشت",  # fate
        "خواب",  # dream
        "غم",  # sorrow
        "شادی",  # joy
        "زندگی",  # life
        "مرگ",  # death
        "ابدیت",  # eternity
        "ماه",  # moon
        "گل",  # flower
        "شراب",  # wine (Hafez)
        "عارف",  # mystic
        "تنهایی",  # loneliness (Hedayat)
        "یأس",  # despair
        "سایه",  # shadow
        # Classical Persian poetry (Hafez, Rumi, Saadi)
        "معشوق",  # beloved
        "عاشق",  # lover
        "ساقی",  # cupbearer
        "میخانه",  # tavern
        "مستی",  # intoxication
        "بلبل",  # nightingale
        "سرو",  # cypress
        "نرگس",  # narcissus
        "لاله",  # tulip
        "باغ",  # garden
        "بهار",  # spring
        "شب",  # night
        "سحر",  # dawn
        "فراق",  # separation
        "وصال",  # union
        "حیرت",  # bewilderment
        "فنا",  # annihilation (Sufi)
        "بقا",  # permanence (Sufi)
        "وحدت",  # unity (Sufi)
        "سکوت",  # silence
        "آینه",  # mirror
        "نور",  # light
        "ظلمت",  # darkness
        # Hedayat modernist vocabulary
        "بوف",  # owl (Blind Owl)
        "کور",  # blind
        "راغه",  # lakhak (in Hedayat)
        "لکاته",  # ethereal woman
        "پیرمرد",  # old man
        "سردابه",  # cellar
        "افیون",  # opium
        "هذیان",  # delirium
        "کابوس",  # nightmare
        "توهم",  # illusion
    }

    # Persian Ghazal structural patterns (Hafez/Rumi style)
    GHAZAL_PATTERNS_FA = [
        r"[\u0600-\u06FF]+\s+[\u0600-\u06FF]+\s+که\s+",  # common ghazal structure
        r"[\u0600-\u06FF]+ی\s+من",  # my X (possessive)
        r"ای\s+[\u0600-\u06FF]+",  # O [beloved]! vocative
    ]

    # Persian Sufi terminology
    SUFI_MARKERS_FA = [
        r"\bفنا\b",  # annihilation
        r"\bبقا\b",  # permanence
        r"\bوحدت\b",  # unity
        r"\bکشف\b",  # unveiling
        r"\bسلوک\b",  # spiritual journey
        r"\bطریقت\b",  # path
        r"\bحقیقت\b",  # truth
        r"\bمعرفت\b",  # gnosis
    ]

    # Chinese classical poetry patterns
    CLASSICAL_CHINESE_PATTERNS = [
        r"[，。！？、；：" + "'']+",  # Chinese punctuation density
        r"[\u4e00-\u9fff]{4,7}[，。]",  # 4-7 character lines (regulated verse)
    ]

    # Persian/Farsi poetic patterns (Hafez, Rumi style)
    PERSIAN_POETIC_PATTERNS = [
        r"[\u0600-\u06FF]+\s+[\u0600-\u06FF]+\s+که\s+",  # Common Persian conjunctions
    ]

    # Hindi literary patterns
    HINDI_LITERARY_PATTERNS = [
        r"[\u0900-\u097F]+\s+है[।]",  # Hindi sentence endings
    ]

    def __init__(self) -> None:
        """Initialize StyleFingerprint analyzer."""
        self._compiled_patterns: dict[str, list[re.Pattern[str]]] = {}

    def analyze(self, text: str, lang: str = "en") -> StyleProfile:
        """Analyze text and create a style profile.

        Args:
            text: Text to analyze
            lang: Language code (e.g., 'en', 'ru')

        Returns:
            StyleProfile with detected features and deviations

        Example:
            >>> fp = StyleFingerprint()
            >>> profile = fp.analyze("The quick brown fox...", lang="en")
            >>> print(profile.deviation_score)
        """
        if not text or not text.strip():
            return StyleProfile()

        # Basic text preprocessing
        text_clean = text.strip()

        # Check for technical documentation first (skip literary analysis)
        is_technical = self._is_technical_documentation(text_clean)
        if is_technical:
            return StyleProfile(
                deviation_score=0.0,
                detected_pattern=StylePattern.TECHNICAL,
                detected_deviations=[],
                lexical_diversity=0.0,
                avg_sentence_length=0.0,
                sentence_length_variance=0.0,
                punctuation_density=0.0,
                is_literary=False,
                is_technical=True,
                recommended_fluency_tolerance=0.0,
                metadata={
                    "language": lang,
                    "text_length": len(text_clean),
                    "word_count": len(text_clean.split()),
                    "document_type": "technical_documentation",
                },
            )

        # Compute stylometric features
        lexical_diversity = self._compute_lexical_diversity(text_clean)
        avg_sent_len, sent_variance = self._compute_sentence_stats(text_clean)
        punct_density = self._compute_punctuation_density(text_clean)

        # Detect specific deviations
        deviations = self._detect_deviations(text_clean, lang)

        # Calculate overall deviation score
        deviation_score = self._calculate_deviation_score(
            deviations=deviations,
            lexical_diversity=lexical_diversity,
            sentence_variance=sent_variance,
            punct_density=punct_density,
            text=text_clean,
            lang=lang,
        )

        # Determine style pattern
        pattern = self._determine_pattern(deviations, deviation_score, lang)

        # Check if literary
        is_literary = self._is_literary_text(text_clean, lang, deviation_score)

        # Calculate fluency tolerance
        fluency_tolerance = self._calculate_fluency_tolerance(deviation_score, pattern)

        profile = StyleProfile(
            deviation_score=deviation_score,
            detected_pattern=pattern,
            detected_deviations=deviations,
            lexical_diversity=lexical_diversity,
            avg_sentence_length=avg_sent_len,
            sentence_length_variance=sent_variance,
            punctuation_density=punct_density,
            is_literary=is_literary,
            recommended_fluency_tolerance=fluency_tolerance,
            metadata={
                "language": lang,
                "text_length": len(text_clean),
                "word_count": len(text_clean.split()),
            },
        )

        logger.info(
            f"Style analysis complete: deviation_score={deviation_score:.2f}, "
            f"pattern={pattern.value}, deviations={len(deviations)}"
        )

        return profile

    def _compute_lexical_diversity(self, text: str) -> float:
        """Compute Type-Token Ratio (TTR) as lexical diversity measure."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        unique_words = set(words)
        # Use root TTR for longer texts to reduce length bias
        if len(words) > 100:
            return len(unique_words) / math.sqrt(len(words))
        return len(unique_words) / len(words) if words else 0.0

    def _compute_sentence_stats(self, text: str) -> tuple[float, float]:
        """Compute average sentence length and variance."""
        # Split by sentence-ending punctuation
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0, 0.0

        lengths = [len(s.split()) for s in sentences]

        avg_len = sum(lengths) / len(lengths)

        # Variance
        if len(lengths) > 1:
            variance = sum((ln - avg_len) ** 2 for ln in lengths) / len(lengths)
        else:
            variance = 0.0

        return avg_len, variance

    def _compute_punctuation_density(self, text: str) -> float:
        """Compute punctuation marks per 100 words."""
        words = text.split()
        if not words:
            return 0.0

        punct_count = len(re.findall(r"[.,;:!?—\-\"'()[\]{}]", text))
        return (punct_count / len(words)) * 100

    def _is_technical_documentation(self, text: str) -> bool:
        """Detect if text is technical documentation (CLI, API docs, Markdown).

        Technical documentation markers:
        - Code blocks (```bash, ```python, etc.)
        - CLI options (--option, -flag)
        - Markdown headers (##, ###)
        - Tables (| Column |)
        - Environment variables (KTTC_*, API_KEY)
        - File extensions (.py, .json, .md)
        - Command examples (kttc check, npm install)

        Returns:
            True if text appears to be technical documentation
        """
        # Technical documentation markers
        technical_markers = [
            r"```\w*",  # Code blocks
            r"--\w+[-\w]*",  # CLI long options
            r"\s-[a-zA-Z]\s",  # CLI short options
            r"^#{1,4}\s",  # Markdown headers
            r"\|\s*[-:]+\s*\|",  # Markdown tables
            r"\bAPI\b|\bCLI\b|\bSDK\b|\bHTTP\b",  # Technical acronyms
            r"\b\w+_\w+_\w+\b",  # SNAKE_CASE variables
            r"\.\w{2,4}\b",  # File extensions
            r"kttc\s+\w+|npm\s+\w+|pip\s+\w+",  # CLI commands
            r"--source-lang|--target-lang|--provider",  # KTTC specific
            r"bash\n|python\n|json\n",  # Code block languages
            r"\bdef\s+\w+\(|\bclass\s+\w+:",  # Python code
            r"import\s+\w+|from\s+\w+\s+import",  # Python imports
        ]

        marker_count = 0
        for pattern in technical_markers:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            marker_count += len(matches)

        # Calculate marker density (markers per 1000 characters)
        text_len = len(text)
        if text_len == 0:
            return False

        marker_density = (marker_count / text_len) * 1000

        # Threshold: if more than 5 markers per 1000 chars, it's technical
        # Or if absolute count > 10 for shorter texts
        is_technical = marker_density > 5 or (marker_count > 10 and text_len < 5000)

        if is_technical:
            logger.info(
                f"Technical documentation detected: {marker_count} markers, "
                f"density={marker_density:.2f}/1000 chars"
            )

        return is_technical

    def _detect_deviations(self, text: str, lang: str) -> list[StyleDeviation]:
        """Detect stylistic deviations in text."""
        deviations: list[StyleDeviation] = []

        # Pleonasms
        pleonasm_dev = self._detect_pleonasms(text, lang)
        if pleonasm_dev:
            deviations.append(pleonasm_dev)

        # Syntactic inversions
        inversion_dev = self._detect_inversions(text, lang)
        if inversion_dev:
            deviations.append(inversion_dev)

        # Folk speech (Russian)
        if lang == "ru":
            folk_dev = self._detect_folk_speech(text)
            if folk_dev:
                deviations.append(folk_dev)

        # Stream of consciousness markers
        stream_dev = self._detect_stream_markers(text)
        if stream_dev:
            deviations.append(stream_dev)

        # Register mixing
        register_dev = self._detect_register_mixing(text, lang)
        if register_dev:
            deviations.append(register_dev)

        # Fragmentation
        frag_dev = self._detect_fragmentation(text)
        if frag_dev:
            deviations.append(frag_dev)

        return deviations

    def _detect_pleonasms(self, text: str, lang: str) -> StyleDeviation | None:
        """Detect pleonastic (redundant) expressions."""
        patterns = self.PLEONASM_PATTERNS_RU if lang == "ru" else self.PLEONASM_PATTERNS_EN

        examples: list[str] = []
        locations: list[tuple[int, int]] = []

        text_lower = text.lower()
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                examples.append(match.group())
                locations.append((match.start(), match.end()))

        if examples:
            return StyleDeviation(
                type=StyleDeviationType.PLEONASM,
                examples=examples[:5],  # Limit examples
                locations=locations[:5],
                confidence=min(0.6 + len(examples) * 0.1, 0.95),
                interpretation="Deliberate redundancy - possible Platanov-style writing",
                is_intentional=True,
            )
        return None

    def _detect_inversions(self, text: str, lang: str) -> StyleDeviation | None:
        """Detect syntactic inversions (non-standard word order)."""
        if lang != "ru":
            return None  # Currently only Russian patterns

        examples: list[str] = []
        locations: list[tuple[int, int]] = []

        for pattern in self.INVERSION_PATTERNS_RU:
            for match in re.finditer(pattern, text.lower()):
                examples.append(match.group())
                locations.append((match.start(), match.end()))

        if examples:
            return StyleDeviation(
                type=StyleDeviationType.INVERSION,
                examples=examples[:5],
                locations=locations[:5],
                confidence=0.7,
                interpretation="Non-standard word order - may be stylistic choice",
                is_intentional=True,
            )
        return None

    def _detect_folk_speech(self, text: str) -> StyleDeviation | None:
        """Detect folk speech patterns (Russian skaz markers)."""
        examples: list[str] = []
        locations: list[tuple[int, int]] = []

        for pattern in self.FOLK_SPEECH_RU:
            for match in re.finditer(pattern, text.lower()):
                examples.append(match.group())
                locations.append((match.start(), match.end()))

        if len(examples) >= 2:  # Need multiple markers
            return StyleDeviation(
                type=StyleDeviationType.SKAZ,
                examples=examples[:5],
                locations=locations[:5],
                confidence=min(0.5 + len(examples) * 0.1, 0.9),
                interpretation="Folk speech patterns - possible skaz narrative style (Leskov)",
                is_intentional=True,
            )
        return None

    def _detect_stream_markers(self, text: str) -> StyleDeviation | None:
        """Detect stream of consciousness markers."""
        examples: list[str] = []
        locations: list[tuple[int, int]] = []

        for pattern in self.STREAM_MARKERS:
            for match in re.finditer(pattern, text):
                examples.append(match.group())
                locations.append((match.start(), match.end()))

        # Also check for very long sentences (stream of consciousness)
        sentences = re.split(r"[.!?]+", text)
        long_sentences = [s for s in sentences if len(s.split()) > 50]

        if long_sentences:
            examples.extend([f"Long sentence ({len(s.split())} words)" for s in long_sentences[:2]])

        if len(examples) >= 3:
            return StyleDeviation(
                type=StyleDeviationType.STREAM_OF_CONSCIOUSNESS,
                examples=examples[:5],
                locations=locations[:5],
                confidence=min(0.5 + len(examples) * 0.1, 0.85),
                interpretation="Stream of consciousness markers detected",
                is_intentional=True,
            )
        return None

    def _detect_register_mixing(self, text: str, lang: str) -> StyleDeviation | None:
        """Detect mixing of formal and informal registers."""
        # Simple heuristic: check for mix of formal and informal markers
        if lang == "ru":
            formal_markers = len(
                re.findall(r"\b(?:вследствие|посему|таким образом|надлежит)\b", text.lower())
            )
            informal_markers = len(
                re.findall(r"\b(?:блин|типа|короче|прикинь|чё|ваще)\b", text.lower())
            )
        else:
            formal_markers = len(
                re.findall(r"\b(?:therefore|hence|wherein|thereof|pursuant)\b", text.lower())
            )
            informal_markers = len(
                re.findall(r"\b(?:gonna|wanna|kinda|sorta|yeah|nope)\b", text.lower())
            )

        if formal_markers >= 2 and informal_markers >= 2:
            return StyleDeviation(
                type=StyleDeviationType.REGISTER_MIXING,
                examples=[
                    f"Formal markers: {formal_markers}, Informal markers: {informal_markers}"
                ],
                locations=[],
                confidence=0.7,
                interpretation="Mixing of formal and informal registers - may be intentional",
                is_intentional=True,
            )
        return None

    def _detect_fragmentation(self, text: str) -> StyleDeviation | None:
        """Detect sentence fragmentation."""
        sentences = re.split(r"[.!?]+", text)
        short_sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) <= 3]

        if len(short_sentences) >= 5:
            return StyleDeviation(
                type=StyleDeviationType.FRAGMENTATION,
                examples=short_sentences[:5],
                locations=[],
                confidence=0.6,
                interpretation="Many short/fragmented sentences - may be stylistic choice",
                is_intentional=True,
            )
        return None

    def _calculate_deviation_score(
        self,
        deviations: list[StyleDeviation],
        lexical_diversity: float,
        sentence_variance: float,
        punct_density: float,
        text: str,
        lang: str,
    ) -> float:
        """Calculate overall deviation score (0.0-1.0)."""
        score = 0.0

        # Deviation count contribution (max 0.4)
        deviation_contribution = min(len(deviations) * 0.1, 0.4)
        score += deviation_contribution

        # High sentence variance suggests unusual style (max 0.2)
        if sentence_variance > 100:
            score += min(sentence_variance / 1000, 0.2)

        # Unusual punctuation density (max 0.15)
        if punct_density > 15:  # Very high
            score += 0.15
        elif punct_density > 10:
            score += 0.08

        # Literary keywords density (max 0.15)
        keywords = self._get_literary_keywords(lang)

        # For Chinese, don't lowercase
        if lang == "zh":
            text_check = text
            word_count = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
        else:
            text_check = text.lower()
            word_count = len(text.split())

        keyword_count = sum(1 for kw in keywords if kw in text_check)
        if word_count > 0:
            keyword_density = keyword_count / word_count
            score += min(keyword_density * 10, 0.15)

        # Unusual lexical diversity (max 0.1)
        if lexical_diversity > 0.8 or lexical_diversity < 0.3:
            score += 0.1

        return min(score, 1.0)

    def _determine_pattern(
        self, deviations: list[StyleDeviation], deviation_score: float, lang: str
    ) -> StylePattern:
        """Determine the primary style pattern."""
        deviation_types = {d.type for d in deviations}

        # Check for specific patterns
        if StyleDeviationType.SKAZ in deviation_types:
            return StylePattern.SKAZ_NARRATIVE

        if StyleDeviationType.STREAM_OF_CONSCIOUSNESS in deviation_types:
            return StylePattern.STREAM

        if (
            StyleDeviationType.PLEONASM in deviation_types
            and StyleDeviationType.INVERSION in deviation_types
        ):
            return StylePattern.MODERNIST

        if (
            StyleDeviationType.COLLOQUIALISM in deviation_types
            or StyleDeviationType.DIALECTISM in deviation_types
        ):
            return StylePattern.COLLOQUIAL

        if StyleDeviationType.ARCHAISM in deviation_types:
            return StylePattern.ARCHAIC

        if len(deviation_types) >= 3:
            return StylePattern.MIXED

        if deviation_score > 0.3:
            return StylePattern.POETIC

        return StylePattern.STANDARD

    def _get_literary_keywords(self, lang: str) -> set[str]:
        """Get literary keywords for the specified language."""
        keywords_map = {
            "ru": self.LITERARY_KEYWORDS_RU,
            "en": self.LITERARY_KEYWORDS_EN,
            "zh": self.LITERARY_KEYWORDS_ZH,
            "hi": self.LITERARY_KEYWORDS_HI,
            "fa": self.LITERARY_KEYWORDS_FA,
        }
        return keywords_map.get(lang, self.LITERARY_KEYWORDS_EN)

    def _is_literary_text(self, text: str, lang: str, deviation_score: float) -> bool:
        """Determine if text appears to be literary."""
        # High deviation score suggests literary
        if deviation_score > 0.4:
            return True

        # Check literary keyword density
        keywords = self._get_literary_keywords(lang)

        # For Chinese, don't lowercase (no case in Chinese)
        if lang == "zh":
            text_check = text
        else:
            text_check = text.lower()

        keyword_count = sum(1 for kw in keywords if kw in text_check)

        # Word count calculation varies by language
        if lang == "zh":
            # Chinese: count characters (rough approximation)
            word_count = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
        else:
            word_count = len(text.split())

        if word_count > 0 and keyword_count / max(word_count, 1) > 0.02:
            return True

        return False

    def _calculate_fluency_tolerance(self, deviation_score: float, pattern: StylePattern) -> float:
        """Calculate recommended fluency tolerance based on style."""
        base_tolerance = deviation_score * 0.5

        # Pattern-specific adjustments
        pattern_adjustments = {
            StylePattern.STANDARD: 0.0,
            StylePattern.SKAZ_NARRATIVE: 0.3,
            StylePattern.MODERNIST: 0.4,
            StylePattern.STREAM: 0.5,
            StylePattern.POETIC: 0.2,
            StylePattern.COLLOQUIAL: 0.2,
            StylePattern.ARCHAIC: 0.15,
            StylePattern.MIXED: 0.25,
        }

        return min(base_tolerance + pattern_adjustments.get(pattern, 0.0), 0.8)
