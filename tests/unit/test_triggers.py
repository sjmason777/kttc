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

"""Strict tests for QA Triggers module.

Tests trigger models, conditions, and manager.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from kttc.triggers import (
    BaseCondition,
    CompositeCondition,
    ErrorCountCondition,
    FilePatternCondition,
    LanguagePairCondition,
    ScoreThresholdCondition,
    TimeBasedCondition,
    Trigger,
    TriggerAction,
    TriggerEvent,
    TriggerManager,
    TriggerResult,
)
from kttc.triggers.models import TRIGGER_TEMPLATES


class TestTriggerEvent:
    """Tests for TriggerEvent enum."""

    def test_all_events_exist(self) -> None:
        """Test that all expected events exist."""
        assert TriggerEvent.FILE_CREATED.value == "file_created"
        assert TriggerEvent.FILE_MODIFIED.value == "file_modified"
        assert TriggerEvent.BATCH_COMPLETE.value == "batch_complete"
        assert TriggerEvent.THRESHOLD_VIOLATED.value == "threshold_violated"
        assert TriggerEvent.SCHEDULE.value == "schedule"
        assert TriggerEvent.MANUAL.value == "manual"
        assert TriggerEvent.WEBHOOK.value == "webhook"


class TestTriggerAction:
    """Tests for TriggerAction enum."""

    def test_all_actions_exist(self) -> None:
        """Test that all expected actions exist."""
        assert TriggerAction.RUN_CHECK.value == "run_check"
        assert TriggerAction.SEND_NOTIFICATION.value == "send_notification"
        assert TriggerAction.BLOCK_DELIVERY.value == "block_delivery"
        assert TriggerAction.LOG_ONLY.value == "log_only"
        assert TriggerAction.ESCALATE.value == "escalate"


class TestTrigger:
    """Comprehensive tests for Trigger model."""

    def test_basic_creation(self) -> None:
        """Test basic trigger creation."""
        trigger = Trigger(
            id="test_trigger",
            name="Test Trigger",
            event=TriggerEvent.FILE_MODIFIED,
        )
        assert trigger.id == "test_trigger"
        assert trigger.name == "Test Trigger"
        assert trigger.event == TriggerEvent.FILE_MODIFIED
        assert trigger.enabled is True
        assert trigger.actions == [TriggerAction.RUN_CHECK]

    def test_creation_with_all_fields(self) -> None:
        """Test trigger with all optional fields."""
        trigger = Trigger(
            id="full_trigger",
            name="Full Trigger",
            description="A comprehensive trigger",
            enabled=False,
            event=TriggerEvent.BATCH_COMPLETE,
            conditions=[{"type": "file_pattern", "patterns": ["*.xliff"]}],
            actions=[TriggerAction.RUN_CHECK, TriggerAction.SEND_NOTIFICATION],
            priority=90,
            cooldown_seconds=300,
            metadata={"custom_key": "custom_value"},
        )
        assert trigger.description == "A comprehensive trigger"
        assert trigger.enabled is False
        assert len(trigger.conditions) == 1
        assert len(trigger.actions) == 2
        assert trigger.priority == 90
        assert trigger.cooldown_seconds == 300
        assert trigger.metadata["custom_key"] == "custom_value"

    def test_should_fire_enabled(self) -> None:
        """Test should_fire when enabled."""
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
            enabled=True,
        )
        assert trigger.should_fire() is True

    def test_should_fire_disabled(self) -> None:
        """Test should_fire when disabled."""
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
            enabled=False,
        )
        assert trigger.should_fire() is False

    def test_should_fire_no_cooldown(self) -> None:
        """Test should_fire with no cooldown."""
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
            cooldown_seconds=0,
        )
        trigger.record_fire()
        # Should still fire even immediately after
        assert trigger.should_fire() is True

    def test_should_fire_during_cooldown(self) -> None:
        """Test should_fire during cooldown period."""
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
            cooldown_seconds=3600,  # 1 hour
        )
        trigger.record_fire()
        # Should not fire during cooldown
        assert trigger.should_fire() is False

    def test_should_fire_after_cooldown(self) -> None:
        """Test should_fire after cooldown period."""
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
            cooldown_seconds=1,
        )
        # Set last_fired to past
        trigger.last_fired = datetime.now() - timedelta(seconds=10)
        assert trigger.should_fire() is True

    def test_record_fire(self) -> None:
        """Test recording a fire."""
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
        )
        assert trigger.fire_count == 0
        assert trigger.last_fired is None

        trigger.record_fire()

        assert trigger.fire_count == 1
        assert trigger.last_fired is not None

        trigger.record_fire()
        assert trigger.fire_count == 2

    def test_priority_validation(self) -> None:
        """Test priority range validation."""
        # Valid priorities
        trigger = Trigger(id="t", name="T", event=TriggerEvent.MANUAL, priority=0)
        assert trigger.priority == 0

        trigger = Trigger(id="t", name="T", event=TriggerEvent.MANUAL, priority=100)
        assert trigger.priority == 100

        # Invalid priorities should raise
        with pytest.raises(ValueError):
            Trigger(id="t", name="T", event=TriggerEvent.MANUAL, priority=-1)

        with pytest.raises(ValueError):
            Trigger(id="t", name="T", event=TriggerEvent.MANUAL, priority=101)


class TestTriggerResult:
    """Tests for TriggerResult model."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = TriggerResult(trigger_id="trigger_001")
        assert result.trigger_id == "trigger_001"
        assert result.fired is False
        assert result.event is None
        assert result.actions_taken == []

    def test_creation_with_all_fields(self) -> None:
        """Test result with all fields."""
        result = TriggerResult(
            trigger_id="trigger_001",
            fired=True,
            event=TriggerEvent.FILE_MODIFIED,
            actions_taken=[TriggerAction.RUN_CHECK],
            details={"file": "test.xliff"},
        )
        assert result.fired is True
        assert result.event == TriggerEvent.FILE_MODIFIED
        assert TriggerAction.RUN_CHECK in result.actions_taken


class TestTriggerTemplates:
    """Tests for pre-built trigger templates."""

    def test_templates_exist(self) -> None:
        """Test that all expected templates exist."""
        assert "ci_cd_check" in TRIGGER_TEMPLATES
        assert "threshold_alert" in TRIGGER_TEMPLATES
        assert "batch_review" in TRIGGER_TEMPLATES
        assert "daily_audit" in TRIGGER_TEMPLATES

    def test_ci_cd_check_template(self) -> None:
        """Test CI/CD check template configuration."""
        template = TRIGGER_TEMPLATES["ci_cd_check"]
        assert template["event"] == TriggerEvent.FILE_MODIFIED
        assert TriggerAction.BLOCK_DELIVERY in template["actions"]
        assert template["priority"] == 90

    def test_threshold_alert_template(self) -> None:
        """Test threshold alert template configuration."""
        template = TRIGGER_TEMPLATES["threshold_alert"]
        assert template["event"] == TriggerEvent.THRESHOLD_VIOLATED
        assert TriggerAction.ESCALATE in template["actions"]


class TestFilePatternCondition:
    """Tests for FilePatternCondition."""

    def test_match_simple_pattern(self) -> None:
        """Test matching simple file patterns."""
        condition = FilePatternCondition(patterns=["*.xliff"])
        assert condition.evaluate({"file_path": "translations.xliff"}) is True
        assert condition.evaluate({"file_path": "data.json"}) is False

    def test_match_multiple_patterns(self) -> None:
        """Test matching multiple patterns."""
        condition = FilePatternCondition(patterns=["*.xliff", "*.po", "*.tmx"])
        assert condition.evaluate({"file_path": "file.xliff"}) is True
        assert condition.evaluate({"file_path": "file.po"}) is True
        assert condition.evaluate({"file_path": "file.tmx"}) is True
        assert condition.evaluate({"file_path": "file.txt"}) is False

    def test_match_full_path(self) -> None:
        """Test matching full file paths."""
        condition = FilePatternCondition(patterns=["translations/*.xliff"])
        assert condition.evaluate({"file_path": "translations/en-ru.xliff"}) is True

    def test_no_file_path(self) -> None:
        """Test with missing file_path."""
        condition = FilePatternCondition(patterns=["*.xliff"])
        assert condition.evaluate({}) is False
        assert condition.evaluate({"file_path": ""}) is False


class TestScoreThresholdCondition:
    """Tests for ScoreThresholdCondition."""

    def test_min_score_violation(self) -> None:
        """Test minimum score threshold violation."""
        condition = ScoreThresholdCondition(metric="mqm", min_score=85.0)
        # Score below minimum should trigger
        assert condition.evaluate({"mqm_score": 80.0}) is True
        assert condition.evaluate({"mqm_score": 85.0}) is False
        assert condition.evaluate({"mqm_score": 90.0}) is False

    def test_max_score_violation(self) -> None:
        """Test maximum score threshold violation."""
        condition = ScoreThresholdCondition(metric="ter", max_score=30.0)
        # TER above maximum should trigger
        assert condition.evaluate({"ter_score": 35.0}) is True
        assert condition.evaluate({"ter_score": 30.0}) is False
        assert condition.evaluate({"ter_score": 25.0}) is False

    def test_both_thresholds(self) -> None:
        """Test with both min and max thresholds."""
        condition = ScoreThresholdCondition(metric="bleu", min_score=50.0, max_score=95.0)
        # Outside range should trigger
        assert condition.evaluate({"bleu_score": 45.0}) is True  # Below min
        assert condition.evaluate({"bleu_score": 98.0}) is True  # Above max
        assert condition.evaluate({"bleu_score": 70.0}) is False  # Within range

    def test_missing_score(self) -> None:
        """Test with missing score."""
        condition = ScoreThresholdCondition(metric="mqm", min_score=85.0)
        assert condition.evaluate({}) is False
        assert condition.evaluate({"other_score": 80.0}) is False

    def test_invalid_score_value(self) -> None:
        """Test with invalid score value."""
        condition = ScoreThresholdCondition(metric="mqm", min_score=85.0)
        assert condition.evaluate({"mqm_score": "not a number"}) is False
        assert condition.evaluate({"mqm_score": None}) is False


class TestTimeBasedCondition:
    """Tests for TimeBasedCondition."""

    def test_match_specific_time(self) -> None:
        """Test matching specific time."""
        condition = TimeBasedCondition(schedule="30 9 * * *")  # 9:30 daily
        # Should match at 9:30
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 9, 30)}) is True
        # Should not match at other times
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 10, 30)}) is False

    def test_match_any_minute(self) -> None:
        """Test matching any minute."""
        condition = TimeBasedCondition(schedule="* 9 * * *")  # Any minute at 9 AM
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 9, 0)}) is True
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 9, 59)}) is True
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 10, 0)}) is False

    def test_match_weekday(self) -> None:
        """Test matching weekday.

        Note: Implementation uses Python weekday() where 0=Monday, 1=Tuesday, etc.
        This differs from cron where 0=Sunday, 1=Monday, etc.
        """
        condition = TimeBasedCondition(
            schedule="0 9 * * 0"
        )  # Monday at 9:00 (Python weekday 0=Monday)
        # Monday (weekday 0 in Python)
        monday = datetime(2025, 1, 13, 9, 0)  # This is a Monday
        tuesday = datetime(2025, 1, 14, 9, 0)
        assert condition.evaluate({"current_time": monday}) is True
        assert condition.evaluate({"current_time": tuesday}) is False

    def test_match_step(self) -> None:
        """Test step patterns (*/5)."""
        condition = TimeBasedCondition(schedule="*/15 * * * *")  # Every 15 minutes
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 10, 0)}) is True
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 10, 15)}) is True
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 10, 30)}) is True
        assert condition.evaluate({"current_time": datetime(2025, 1, 15, 10, 7)}) is False

    def test_invalid_schedule(self) -> None:
        """Test with invalid schedule format."""
        condition = TimeBasedCondition(schedule="invalid")
        # Should default to always match
        assert condition.evaluate({"current_time": datetime.now()}) is True

    def test_no_current_time(self) -> None:
        """Test without current_time in context."""
        condition = TimeBasedCondition(schedule="0 9 * * *")
        # Should use datetime.now()
        result = condition.evaluate({})
        # Result depends on current time, just check it doesn't crash
        assert isinstance(result, bool)


class TestLanguagePairCondition:
    """Tests for LanguagePairCondition."""

    def test_match_source_language(self) -> None:
        """Test matching source language."""
        condition = LanguagePairCondition(source_langs=["en"])
        assert condition.evaluate({"source_lang": "en", "target_lang": "ru"}) is True
        assert condition.evaluate({"source_lang": "de", "target_lang": "ru"}) is False

    def test_match_target_language(self) -> None:
        """Test matching target language."""
        condition = LanguagePairCondition(target_langs=["ru", "de"])
        assert condition.evaluate({"source_lang": "en", "target_lang": "ru"}) is True
        assert condition.evaluate({"source_lang": "en", "target_lang": "fr"}) is False

    def test_match_both_languages(self) -> None:
        """Test matching both source and target."""
        condition = LanguagePairCondition(source_langs=["en"], target_langs=["ru"])
        assert condition.evaluate({"source_lang": "en", "target_lang": "ru"}) is True
        assert condition.evaluate({"source_lang": "en", "target_lang": "de"}) is False
        assert condition.evaluate({"source_lang": "de", "target_lang": "ru"}) is False

    def test_case_insensitive(self) -> None:
        """Test case insensitive matching."""
        condition = LanguagePairCondition(source_langs=["EN"], target_langs=["RU"])
        assert condition.evaluate({"source_lang": "en", "target_lang": "ru"}) is True

    def test_no_filters(self) -> None:
        """Test with no language filters (matches any)."""
        condition = LanguagePairCondition()
        assert condition.evaluate({"source_lang": "en", "target_lang": "ru"}) is True
        assert condition.evaluate({"source_lang": "zh", "target_lang": "ja"}) is True


class TestErrorCountCondition:
    """Tests for ErrorCountCondition."""

    def test_min_errors(self) -> None:
        """Test minimum error count threshold."""
        condition = ErrorCountCondition(min_errors=5)
        assert condition.evaluate({"error_count": 10}) is True
        assert condition.evaluate({"error_count": 5}) is True
        assert condition.evaluate({"error_count": 3}) is False

    def test_max_errors(self) -> None:
        """Test maximum error count threshold."""
        condition = ErrorCountCondition(max_errors=5)
        assert condition.evaluate({"error_count": 3}) is True
        assert condition.evaluate({"error_count": 5}) is True
        assert condition.evaluate({"error_count": 10}) is False

    def test_missing_error_count(self) -> None:
        """Test with missing error count."""
        condition = ErrorCountCondition(min_errors=5)
        assert condition.evaluate({}) is False

    def test_invalid_error_count(self) -> None:
        """Test with invalid error count."""
        condition = ErrorCountCondition(min_errors=5)
        assert condition.evaluate({"error_count": "many"}) is False


class TestCompositeCondition:
    """Tests for CompositeCondition."""

    def test_and_operator(self) -> None:
        """Test AND operator."""
        cond1 = FilePatternCondition(patterns=["*.xliff"])
        cond2 = ScoreThresholdCondition(metric="mqm", min_score=85.0)

        composite = CompositeCondition([cond1, cond2], operator="and")

        # Both must be true
        context = {"file_path": "test.xliff", "mqm_score": 80.0}
        assert composite.evaluate(context) is True

        context = {"file_path": "test.json", "mqm_score": 80.0}
        assert composite.evaluate(context) is False

    def test_or_operator(self) -> None:
        """Test OR operator."""
        cond1 = FilePatternCondition(patterns=["*.xliff"])
        cond2 = FilePatternCondition(patterns=["*.po"])

        composite = CompositeCondition([cond1, cond2], operator="or")

        # Either can be true
        assert composite.evaluate({"file_path": "test.xliff"}) is True
        assert composite.evaluate({"file_path": "test.po"}) is True
        assert composite.evaluate({"file_path": "test.txt"}) is False

    def test_empty_conditions(self) -> None:
        """Test with empty conditions list."""
        composite = CompositeCondition([], operator="and")
        assert composite.evaluate({}) is True


class TestBaseConditionFromDict:
    """Tests for BaseCondition.from_dict factory method."""

    def test_create_file_pattern(self) -> None:
        """Test creating FilePatternCondition from dict."""
        config = {"type": "file_pattern", "patterns": ["*.xliff"]}
        condition = BaseCondition.from_dict(config)
        assert isinstance(condition, FilePatternCondition)

    def test_create_score_threshold(self) -> None:
        """Test creating ScoreThresholdCondition from dict."""
        config = {"type": "score_threshold", "metric": "mqm", "min_score": 85.0}
        condition = BaseCondition.from_dict(config)
        assert isinstance(condition, ScoreThresholdCondition)

    def test_create_time_based(self) -> None:
        """Test creating TimeBasedCondition from dict."""
        config = {"type": "time_based", "schedule": "0 9 * * *"}
        condition = BaseCondition.from_dict(config)
        assert isinstance(condition, TimeBasedCondition)

    def test_create_language_pair(self) -> None:
        """Test creating LanguagePairCondition from dict."""
        config = {"type": "language_pair", "source_langs": ["en"], "target_langs": ["ru"]}
        condition = BaseCondition.from_dict(config)
        assert isinstance(condition, LanguagePairCondition)

    def test_create_error_count(self) -> None:
        """Test creating ErrorCountCondition from dict."""
        config = {"type": "error_count", "min_errors": 5}
        condition = BaseCondition.from_dict(config)
        assert isinstance(condition, ErrorCountCondition)

    def test_unknown_type(self) -> None:
        """Test with unknown condition type."""
        config = {"type": "unknown_type"}
        with pytest.raises(ValueError, match="Unknown condition type"):
            BaseCondition.from_dict(config)


class TestTriggerManager:
    """Comprehensive tests for TriggerManager."""

    def test_register_trigger(self) -> None:
        """Test registering a trigger."""
        manager = TriggerManager()
        trigger = Trigger(id="test", name="Test", event=TriggerEvent.MANUAL)

        manager.register(trigger)

        assert "test" in manager.triggers
        assert manager.get("test") is trigger

    def test_unregister_trigger(self) -> None:
        """Test unregistering a trigger."""
        manager = TriggerManager()
        trigger = Trigger(id="test", name="Test", event=TriggerEvent.MANUAL)
        manager.register(trigger)

        result = manager.unregister("test")

        assert result is True
        assert manager.get("test") is None

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering non-existent trigger."""
        manager = TriggerManager()
        result = manager.unregister("nonexistent")
        assert result is False

    def test_list_triggers(self) -> None:
        """Test listing triggers."""
        manager = TriggerManager()
        manager.register(Trigger(id="t1", name="T1", event=TriggerEvent.MANUAL, priority=50))
        manager.register(Trigger(id="t2", name="T2", event=TriggerEvent.FILE_MODIFIED, priority=90))
        manager.register(Trigger(id="t3", name="T3", event=TriggerEvent.MANUAL, priority=30))

        # List all - sorted by priority descending
        all_triggers = manager.list_triggers()
        assert len(all_triggers) == 3
        assert all_triggers[0].id == "t2"  # Priority 90

        # Filter by event
        manual_triggers = manager.list_triggers(event=TriggerEvent.MANUAL)
        assert len(manual_triggers) == 2

    def test_create_from_template(self) -> None:
        """Test creating trigger from template."""
        manager = TriggerManager()
        trigger = manager.create_from_template("ci_cd_check")

        assert trigger.id == "ci_cd_check"
        assert trigger.event == TriggerEvent.FILE_MODIFIED
        assert "ci_cd_check" in manager.triggers

    def test_create_from_template_with_overrides(self) -> None:
        """Test creating trigger from template with overrides."""
        manager = TriggerManager()
        trigger = manager.create_from_template(
            "ci_cd_check",
            trigger_id="custom_id",
            priority=50,
        )

        assert trigger.id == "custom_id"
        assert trigger.priority == 50

    def test_create_from_template_unknown(self) -> None:
        """Test creating from unknown template."""
        manager = TriggerManager()
        with pytest.raises(ValueError, match="Unknown template"):
            manager.create_from_template("nonexistent_template")

    def test_enable_disable(self) -> None:
        """Test enabling and disabling triggers."""
        manager = TriggerManager()
        trigger = Trigger(id="test", name="Test", event=TriggerEvent.MANUAL, enabled=False)
        manager.register(trigger)

        assert manager.enable("test") is True
        assert trigger.enabled is True

        assert manager.disable("test") is True
        assert trigger.enabled is False

        # Non-existent
        assert manager.enable("nonexistent") is False
        assert manager.disable("nonexistent") is False

    def test_get_stats(self) -> None:
        """Test getting trigger statistics."""
        manager = TriggerManager()
        t1 = Trigger(id="t1", name="T1", event=TriggerEvent.MANUAL, enabled=True)
        t2 = Trigger(id="t2", name="T2", event=TriggerEvent.MANUAL, enabled=False)
        t1.fire_count = 5
        t2.fire_count = 3

        manager.register(t1)
        manager.register(t2)

        stats = manager.get_stats()

        assert stats["total_triggers"] == 2
        assert stats["enabled_triggers"] == 1
        assert stats["disabled_triggers"] == 1
        assert stats["total_fires"] == 8
        assert stats["by_event"]["manual"] == 2

    def test_export_import_config(self) -> None:
        """Test exporting and importing configuration."""
        manager = TriggerManager()
        trigger = Trigger(
            id="test",
            name="Test Trigger",
            description="A test trigger",
            event=TriggerEvent.MANUAL,
            actions=[TriggerAction.RUN_CHECK, TriggerAction.LOG_ONLY],
            priority=75,
        )
        manager.register(trigger)

        config = manager.export_config()

        # Create new manager and import
        new_manager = TriggerManager()
        count = new_manager.import_config(config)

        assert count == 1
        imported = new_manager.get("test")
        assert imported is not None
        assert imported.name == "Test Trigger"
        assert imported.priority == 75


class TestTriggerManagerFireEvent:
    """Tests for TriggerManager.fire_event."""

    @pytest.mark.asyncio
    async def test_fire_event_matching_trigger(self) -> None:
        """Test firing event with matching trigger."""
        manager = TriggerManager()
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.FILE_MODIFIED,
            conditions=[{"type": "file_pattern", "patterns": ["*.xliff"]}],
        )
        manager.register(trigger)

        results = await manager.fire_event(
            TriggerEvent.FILE_MODIFIED,
            {"file_path": "translations.xliff"},
        )

        assert len(results) == 1
        assert results[0].fired is True
        assert results[0].trigger_id == "test"

    @pytest.mark.asyncio
    async def test_fire_event_no_matching_trigger(self) -> None:
        """Test firing event with no matching trigger."""
        manager = TriggerManager()
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.FILE_MODIFIED,  # Different event
        )
        manager.register(trigger)

        results = await manager.fire_event(TriggerEvent.MANUAL, {})

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_fire_event_condition_not_met(self) -> None:
        """Test firing event when condition not met."""
        manager = TriggerManager()
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.FILE_MODIFIED,
            conditions=[{"type": "file_pattern", "patterns": ["*.xliff"]}],
        )
        manager.register(trigger)

        results = await manager.fire_event(
            TriggerEvent.FILE_MODIFIED,
            {"file_path": "data.json"},  # Wrong extension
        )

        assert len(results) == 1
        assert results[0].fired is False

    @pytest.mark.asyncio
    async def test_fire_event_disabled_trigger(self) -> None:
        """Test firing event with disabled trigger."""
        manager = TriggerManager()
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
            enabled=False,
        )
        manager.register(trigger)

        results = await manager.fire_event(TriggerEvent.MANUAL, {})

        assert len(results) == 1
        assert results[0].fired is False

    @pytest.mark.asyncio
    async def test_fire_event_custom_action_handler(self) -> None:
        """Test firing event with custom action handler."""
        manager = TriggerManager()
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
            actions=[TriggerAction.RUN_CHECK],
        )
        manager.register(trigger)

        handler_called = []

        def custom_handler(trigger: Trigger, context: dict) -> None:
            handler_called.append(trigger.id)

        manager.register_action_handler(TriggerAction.RUN_CHECK, custom_handler)

        await manager.fire_event(TriggerEvent.MANUAL, {})

        assert "test" in handler_called

    @pytest.mark.asyncio
    async def test_fire_event_async_action_handler(self) -> None:
        """Test firing event with async action handler."""
        manager = TriggerManager()
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
            actions=[TriggerAction.SEND_NOTIFICATION],
        )
        manager.register(trigger)

        async def async_handler(trigger: Trigger, context: dict) -> None:
            context["notification_sent"] = True

        manager.register_action_handler(TriggerAction.SEND_NOTIFICATION, async_handler)

        context: dict = {}
        await manager.fire_event(TriggerEvent.MANUAL, context)

        assert context.get("notification_sent") is True

    @pytest.mark.asyncio
    async def test_fire_event_records_fire(self) -> None:
        """Test that firing event records the fire."""
        manager = TriggerManager()
        trigger = Trigger(
            id="test",
            name="Test",
            event=TriggerEvent.MANUAL,
        )
        manager.register(trigger)

        assert trigger.fire_count == 0

        await manager.fire_event(TriggerEvent.MANUAL, {})

        assert trigger.fire_count == 1
        assert trigger.last_fired is not None
