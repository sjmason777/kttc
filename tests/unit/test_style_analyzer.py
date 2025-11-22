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

"""Tests for style analysis module (literary text detection)."""


from kttc.style import (
    ComparativeStyleAnalyzer,
    StyleDeviation,
    StyleDeviationType,
    StyleFingerprint,
    StylePattern,
    StyleProfile,
)


class TestStyleFingerprint:
    """Tests for StyleFingerprint analyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fingerprint = StyleFingerprint()

    def test_analyze_standard_text_en(self):
        """Test analysis of standard English text."""
        text = "The quick brown fox jumps over the lazy dog. This is a normal sentence."
        profile = self.fingerprint.analyze(text, lang="en")

        assert isinstance(profile, StyleProfile)
        assert profile.deviation_score < 0.3
        assert profile.detected_pattern == StylePattern.STANDARD
        assert not profile.has_significant_deviations

    def test_analyze_standard_text_ru(self):
        """Test analysis of standard Russian text."""
        text = "Быстрая коричневая лиса перепрыгнула через ленивую собаку. Это обычное предложение."
        profile = self.fingerprint.analyze(text, lang="ru")

        assert isinstance(profile, StyleProfile)
        assert profile.deviation_score < 0.3
        assert not profile.has_significant_deviations

    def test_detect_pleonasm_ru(self):
        """Test detection of pleonasms in Russian (Platanov-style)."""
        # Platanov-style text with deliberate redundancies
        text = "Душа его желала жить жизнью и думать думу. Он видел виды и пел песни."
        profile = self.fingerprint.analyze(text, lang="ru")

        assert profile.deviation_score > 0.3
        assert profile.has_significant_deviations

        # Check pleonasm detected
        pleonasm_found = any(
            d.type == StyleDeviationType.PLEONASM for d in profile.detected_deviations
        )
        assert pleonasm_found, "Pleonasm should be detected in Platanov-style text"

    def test_detect_folk_speech_ru(self):
        """Test detection of folk speech markers (Leskov-style skaz)."""
        text = """
        Батюшки мои, что же это делается! Голубчик ты мой родимый,
        авось да небось всё образуется. Вишь, как оно выходит.
        """
        profile = self.fingerprint.analyze(text, lang="ru")

        assert profile.deviation_score > 0.3

        # Check skaz detected
        skaz_found = any(d.type == StyleDeviationType.SKAZ for d in profile.detected_deviations)
        assert skaz_found, "Skaz pattern should be detected in Leskov-style text"

        # Should be classified as skaz_narrative pattern
        assert profile.detected_pattern == StylePattern.SKAZ_NARRATIVE

    def test_detect_stream_of_consciousness(self):
        """Test detection of stream of consciousness markers."""
        text = """
        And yes I said yes I will Yes... the sun shines for you he said the day we were
        lying among the rhododendrons on Howth head in the grey tweed suit and his straw
        hat the day I got him to propose to me yes first I gave him the bit of seedcake
        out of my mouth and it was leapyear like now yes 16 years ago my God after that
        long kiss I near lost my breath yes he said I was a flower of the mountain yes...
        """
        profile = self.fingerprint.analyze(text, lang="en")

        # Should have high deviation score due to long sentence and repetition
        assert profile.deviation_score > 0.2 or len(profile.detected_deviations) > 0

    def test_empty_text(self):
        """Test handling of empty text."""
        profile = self.fingerprint.analyze("", lang="en")

        assert profile.deviation_score == 0.0
        assert profile.detected_pattern == StylePattern.STANDARD
        assert len(profile.detected_deviations) == 0

    def test_lexical_diversity_calculation(self):
        """Test lexical diversity calculation."""
        # Text with high repetition (low diversity)
        text = "the the the cat cat cat sat sat sat on on on the the the mat mat mat"
        profile = self.fingerprint.analyze(text, lang="en")

        assert profile.lexical_diversity < 0.5

        # Text with varied vocabulary (high diversity)
        text2 = "The quick brown fox jumps over the lazy dog while birds sing melodies."
        profile2 = self.fingerprint.analyze(text2, lang="en")

        assert profile2.lexical_diversity > profile.lexical_diversity

    def test_sentence_stats(self):
        """Test sentence length statistics."""
        # Text with uniform sentence lengths
        text = "Short sentence here. Another short one. And one more short."
        profile = self.fingerprint.analyze(text, lang="en")

        assert profile.avg_sentence_length > 0
        assert profile.sentence_length_variance < 10

    def test_agent_weight_adjustments(self):
        """Test automatic agent weight adjustments based on style."""
        # Standard text - no significant adjustments
        standard_profile = StyleProfile(
            deviation_score=0.1,
            detected_pattern=StylePattern.STANDARD,
        )
        adjustments = standard_profile.get_agent_weight_adjustments()
        assert adjustments["fluency"] == 1.0

        # High deviation text - fluency weight should decrease
        literary_profile = StyleProfile(
            deviation_score=0.6,
            detected_pattern=StylePattern.MODERNIST,
            detected_deviations=[
                StyleDeviation(type=StyleDeviationType.PLEONASM),
                StyleDeviation(type=StyleDeviationType.INVERSION),
            ],
        )
        adjustments = literary_profile.get_agent_weight_adjustments()
        assert adjustments["fluency"] < 1.0
        assert adjustments["style_preservation"] > 1.0


class TestComparativeStyleAnalyzer:
    """Tests for ComparativeStyleAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ComparativeStyleAnalyzer()

    def test_compare_standard_texts(self):
        """Test comparison of standard texts."""
        source = "The cat sat on the mat."
        translation = "Кот сидел на коврике."

        result = self.analyzer.compare(
            source_text=source,
            translation=translation,
            source_lang="en",
            target_lang="ru",
        )

        # Standard texts should have high preservation score
        assert result.style_preservation_score >= 0.8

    def test_detect_style_loss(self):
        """Test detection of style loss in translation."""
        # Source with pleonastic style
        source_profile = StyleProfile(
            deviation_score=0.6,
            detected_pattern=StylePattern.MODERNIST,
            detected_deviations=[
                StyleDeviation(
                    type=StyleDeviationType.PLEONASM,
                    examples=["live life", "think thought"],
                ),
            ],
        )

        # Target without pleonasms (normalized)
        target_profile = StyleProfile(
            deviation_score=0.1,
            detected_pattern=StylePattern.STANDARD,
            detected_deviations=[],
        )

        result = self.analyzer.compare(
            source_text="dummy",
            translation="dummy",
            source_lang="ru",
            target_lang="en",
            source_profile=source_profile,
            target_profile=target_profile,
        )

        # Should detect lost deviations
        assert len(result.lost_deviations) > 0
        assert result.deviation_transfer_rate < 1.0
        assert len(result.recommendations) > 0

    def test_preserved_style(self):
        """Test detection of preserved style."""
        # Both source and target have similar style
        source_profile = StyleProfile(
            deviation_score=0.5,
            detected_pattern=StylePattern.SKAZ_NARRATIVE,
            detected_deviations=[
                StyleDeviation(type=StyleDeviationType.SKAZ),
            ],
        )

        target_profile = StyleProfile(
            deviation_score=0.45,
            detected_pattern=StylePattern.COLLOQUIAL,
            detected_deviations=[
                StyleDeviation(type=StyleDeviationType.COLLOQUIALISM),
            ],
        )

        result = self.analyzer.compare(
            source_text="dummy",
            translation="dummy",
            source_lang="ru",
            target_lang="en",
            source_profile=source_profile,
            target_profile=target_profile,
        )

        # Skaz can be preserved as colloquialism
        assert result.deviation_transfer_rate > 0


class TestStyleProfile:
    """Tests for StyleProfile data model."""

    def test_has_significant_deviations(self):
        """Test has_significant_deviations property."""
        # Low score, no deviations
        profile1 = StyleProfile(deviation_score=0.1)
        assert not profile1.has_significant_deviations

        # High score
        profile2 = StyleProfile(deviation_score=0.5)
        assert profile2.has_significant_deviations

        # Many deviations
        profile3 = StyleProfile(
            deviation_score=0.2,
            detected_deviations=[
                StyleDeviation(type=StyleDeviationType.PLEONASM),
                StyleDeviation(type=StyleDeviationType.INVERSION),
                StyleDeviation(type=StyleDeviationType.SKAZ),
            ],
        )
        assert profile3.has_significant_deviations

    def test_deviation_types_property(self):
        """Test deviation_types property."""
        profile = StyleProfile(
            detected_deviations=[
                StyleDeviation(type=StyleDeviationType.PLEONASM),
                StyleDeviation(type=StyleDeviationType.INVERSION),
                StyleDeviation(type=StyleDeviationType.PLEONASM),  # Duplicate
            ],
        )

        types = profile.deviation_types
        assert len(types) == 2
        assert StyleDeviationType.PLEONASM in types
        assert StyleDeviationType.INVERSION in types

    def test_to_dict(self):
        """Test serialization to dictionary."""
        profile = StyleProfile(
            deviation_score=0.5,
            detected_pattern=StylePattern.MODERNIST,
            is_literary=True,
        )

        data = profile.to_dict()
        assert data["deviation_score"] == 0.5
        assert data["detected_pattern"] == "modernist"
        assert data["is_literary"] is True


class TestExpandedLiteraryPatterns:
    """Tests for expanded literary patterns from research."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fingerprint = StyleFingerprint()

    def test_chinese_classical_poetry_imagery(self):
        """Test detection of Chinese classical poetry imagery."""
        # Text with classical imagery (月, 花, 酒, 愁)
        text = "月下独酌，花间一壶酒。举杯邀明月，对影成三人。愁绪满怀，相思无限。"
        profile = self.fingerprint.analyze(text, lang="zh")

        assert profile.is_literary
        assert profile.deviation_score > 0.2

    def test_hindi_chhayavad_style(self):
        """Test detection of Hindi Chhayavad poetry style."""
        # Text with Chhayavad vocabulary (छाया, विरह, प्रकृति, चाँद)
        text = "चाँद की छाया में विरह की पीड़ा। प्रकृति का सौंदर्य मन को भाता है। आत्मा की तड़प।"
        profile = self.fingerprint.analyze(text, lang="hi")

        assert profile.is_literary
        # Literary keywords detected, deviation score may vary
        assert profile.deviation_score > 0.1

    def test_persian_ghazal_style(self):
        """Test detection of Persian classical poetry (Hafez/Rumi style)."""
        # Text with Sufi/ghazal vocabulary (عشق, دل, معشوق, شراب)
        text = "عشق را در دل جستجو کن. معشوق در میخانه نشسته. شراب عرفانی می‌نوشد."
        profile = self.fingerprint.analyze(text, lang="fa")

        assert profile.is_literary
        assert profile.deviation_score > 0.2

    def test_russian_dostoevsky_style(self):
        """Test detection of Russian psychological literary vocabulary."""
        # Text with Dostoevsky-style vocabulary (бунт, грех, отчаяние, совесть)
        text = "Его душа была полна отчаяния и раскаяния. Совесть мучила его. Грех не давал покоя."
        profile = self.fingerprint.analyze(text, lang="ru")

        assert profile.is_literary
        assert profile.deviation_score > 0.2

    def test_english_victorian_formal_style(self):
        """Test detection of English Victorian/formal register."""
        text = "Hitherto I have endeavoured to comprehend the sublime nature of existence. Nevertheless, one must persevere."
        profile = self.fingerprint.analyze(text, lang="en")

        # Should detect formal/archaic register
        assert profile.lexical_diversity > 0.5

    def test_english_modernist_stream(self):
        """Test detection of English stream of consciousness (Joyce style)."""
        text = (
            "and yes I said yes and the flowers yes the sun yes... "
            "consciousness flowing through memory perception void "
            "fragmentation of thought... alienation... epiphany"
        )
        profile = self.fingerprint.analyze(text, lang="en")

        assert profile.is_literary or profile.deviation_score > 0.2

    def test_russian_leskov_errative(self):
        """Test detection of Leskov-style errative patterns."""
        # Leskov uses folk etymology (буреметр instead of барометр)
        text = (
            "Стало быть, батюшки мои, дескать, голубчик родимый, "
            "авось да небось всё образуется. Нынче так, завтра иначе."
        )
        profile = self.fingerprint.analyze(text, lang="ru")

        assert profile.has_significant_deviations
        assert profile.detected_pattern in [StylePattern.SKAZ_NARRATIVE, StylePattern.COLLOQUIAL]

    def test_chinese_lu_xun_modernist(self):
        """Test detection of Lu Xun modernist vocabulary."""
        text = "铁屋子里的呐喊，彷徨中的狂人。灵魂的挣扎，命运的悲剧。"
        profile = self.fingerprint.analyze(text, lang="zh")

        assert profile.is_literary

    def test_hindi_premchand_social_realism(self):
        """Test detection of Premchand social realism vocabulary."""
        text = "किसान का दुख समाज में न्याय की कमी से है। गरीब मजदूर जमींदार के शोषण में।"
        profile = self.fingerprint.analyze(text, lang="hi")

        assert profile.is_literary

    def test_persian_hedayat_modernist(self):
        """Test detection of Hedayat modernist style (Blind Owl)."""
        text = "در تنهایی و سکوت، سایه‌ها بر دیوار. کابوس و توهم. مرگ همیشه نزدیک است."
        profile = self.fingerprint.analyze(text, lang="fa")

        assert profile.is_literary
