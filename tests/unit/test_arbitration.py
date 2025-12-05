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

"""Strict tests for Arbitration module.

Tests arbitration workflow for translator objection handling.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.arbitration import (
    ArbitrationDecision,
    ArbitrationResult,
    ArbitrationStatus,
    ArbitrationWorkflow,
    Objection,
    ObjectionStatus,
)


class TestObjectionStatus:
    """Tests for ObjectionStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test that all expected statuses exist."""
        assert ObjectionStatus.PENDING.value == "pending"
        assert ObjectionStatus.ACCEPTED.value == "accepted"
        assert ObjectionStatus.REJECTED.value == "rejected"
        assert ObjectionStatus.PARTIAL.value == "partial"


class TestArbitrationStatus:
    """Tests for ArbitrationStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test that all expected statuses exist."""
        assert ArbitrationStatus.DRAFT.value == "draft"
        assert ArbitrationStatus.PENDING_OBJECTIONS.value == "pending_objections"
        assert ArbitrationStatus.PENDING_ARBITRATION.value == "pending_arbitration"
        assert ArbitrationStatus.COMPLETED.value == "completed"
        assert ArbitrationStatus.CLOSED.value == "closed"


class TestArbitrationDecision:
    """Tests for ArbitrationDecision enum."""

    def test_all_decisions_exist(self) -> None:
        """Test that all expected decisions exist."""
        assert ArbitrationDecision.UPHELD.value == "upheld"
        assert ArbitrationDecision.DISMISSED.value == "dismissed"
        assert ArbitrationDecision.REDUCED.value == "reduced"
        assert ArbitrationDecision.ESCALATED.value == "escalated"


class TestObjection:
    """Comprehensive tests for Objection model."""

    def test_basic_creation(self) -> None:
        """Test basic objection creation."""
        objection = Objection(error_id="err_001", reason="This is acceptable in Russian")
        assert objection.error_id == "err_001"
        assert objection.reason == "This is acceptable in Russian"
        assert objection.status == ObjectionStatus.PENDING
        assert objection.decision is None

    def test_creation_with_all_fields(self) -> None:
        """Test objection with all optional fields."""
        objection = Objection(
            error_id="err_002",
            translator_id="translator_123",
            reason="Style guide allows this",
            evidence=["Style guide v2.0, page 15", "Client feedback"],
            proposed_resolution="Remove error flag",
        )
        assert objection.translator_id == "translator_123"
        assert len(objection.evidence) == 2
        assert objection.proposed_resolution == "Remove error flag"

    def test_accept_method(self) -> None:
        """Test accepting an objection."""
        objection = Objection(error_id="err_001", reason="Valid concern")
        objection.accept("Translation is culturally appropriate")

        assert objection.status == ObjectionStatus.ACCEPTED
        assert objection.decision == ArbitrationDecision.DISMISSED
        assert "culturally appropriate" in objection.decision_reason
        assert objection.resolved_at is not None

    def test_accept_default_reason(self) -> None:
        """Test accepting with default reason."""
        objection = Objection(error_id="err_001", reason="Test")
        objection.accept()

        assert objection.decision_reason == "Objection accepted"

    def test_reject_method(self) -> None:
        """Test rejecting an objection."""
        objection = Objection(error_id="err_001", reason="Invalid concern")
        objection.reject("Error correctly identified per MQM guidelines")

        assert objection.status == ObjectionStatus.REJECTED
        assert objection.decision == ArbitrationDecision.UPHELD
        assert "MQM guidelines" in objection.decision_reason
        assert objection.resolved_at is not None

    def test_reject_default_reason(self) -> None:
        """Test rejecting with default reason."""
        objection = Objection(error_id="err_001", reason="Test")
        objection.reject()

        assert objection.decision_reason == "Original assessment upheld"

    def test_reduce_severity_method(self) -> None:
        """Test reducing error severity."""
        objection = Objection(error_id="err_001", reason="Severity too high")
        objection.reduce_severity("Changed from major to minor")

        assert objection.status == ObjectionStatus.PARTIAL
        assert objection.decision == ArbitrationDecision.REDUCED
        assert "major to minor" in objection.decision_reason
        assert objection.resolved_at is not None

    def test_reduce_severity_default_reason(self) -> None:
        """Test reducing severity with default reason."""
        objection = Objection(error_id="err_001", reason="Test")
        objection.reduce_severity()

        assert objection.decision_reason == "Error severity reduced"

    def test_created_at_auto_set(self) -> None:
        """Test that created_at is automatically set."""
        before = datetime.now()
        objection = Objection(error_id="err_001", reason="Test")
        after = datetime.now()

        assert before <= objection.created_at <= after

    def test_resolved_at_initially_none(self) -> None:
        """Test that resolved_at is initially None."""
        objection = Objection(error_id="err_001", reason="Test")
        assert objection.resolved_at is None


class TestArbitrationResult:
    """Comprehensive tests for ArbitrationResult model."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = ArbitrationResult(report_id="report_001")
        assert result.report_id == "report_001"
        assert result.status == ArbitrationStatus.DRAFT
        assert result.original_error_count == 0
        assert result.final_error_count == 0

    def test_creation_with_all_fields(self) -> None:
        """Test result with all fields."""
        result = ArbitrationResult(
            report_id="report_002",
            status=ArbitrationStatus.PENDING_ARBITRATION,
            original_error_count=10,
            final_error_count=7,
            original_mqm_score=85.0,
            final_mqm_score=90.0,
        )
        assert result.original_error_count == 10
        assert result.final_error_count == 7
        assert result.original_mqm_score == 85.0
        assert result.final_mqm_score == 90.0

    def test_add_objection(self) -> None:
        """Test adding objection to result."""
        result = ArbitrationResult(report_id="report_001")
        objection = Objection(error_id="err_001", reason="Test")

        result.add_objection(objection)

        assert len(result.objections) == 1
        assert result.objections[0].error_id == "err_001"

    def test_add_objection_updates_status(self) -> None:
        """Test that adding objection updates status from DRAFT."""
        result = ArbitrationResult(report_id="report_001")
        assert result.status == ArbitrationStatus.DRAFT

        objection = Objection(error_id="err_001", reason="Test")
        result.add_objection(objection)

        assert result.status == ArbitrationStatus.PENDING_OBJECTIONS

    def test_add_objection_preserves_non_draft_status(self) -> None:
        """Test that adding objection doesn't change non-DRAFT status."""
        result = ArbitrationResult(
            report_id="report_001",
            status=ArbitrationStatus.PENDING_ARBITRATION,
        )
        objection = Objection(error_id="err_001", reason="Test")

        result.add_objection(objection)

        # Status should remain unchanged
        assert result.status == ArbitrationStatus.PENDING_ARBITRATION

    def test_finalize(self) -> None:
        """Test finalizing arbitration result."""
        result = ArbitrationResult(report_id="report_001")

        # Add some objections with different statuses
        obj1 = Objection(error_id="err_001", reason="Test 1")
        obj1.accept()
        obj2 = Objection(error_id="err_002", reason="Test 2")
        obj2.reject()
        obj3 = Objection(error_id="err_003", reason="Test 3")
        obj3.accept()

        result.add_objection(obj1)
        result.add_objection(obj2)
        result.add_objection(obj3)

        result.finalize()

        assert result.status == ArbitrationStatus.COMPLETED
        assert result.completed_at is not None
        assert result.objections_accepted == 2
        assert result.objections_rejected == 1

    def test_objection_acceptance_rate(self) -> None:
        """Test objection acceptance rate calculation."""
        result = ArbitrationResult(report_id="report_001")

        # 2 accepted, 2 rejected
        for i in range(2):
            obj = Objection(error_id=f"accept_{i}", reason="Test")
            obj.accept()
            result.add_objection(obj)

        for i in range(2):
            obj = Objection(error_id=f"reject_{i}", reason="Test")
            obj.reject()
            result.add_objection(obj)

        result.finalize()

        assert result.objection_acceptance_rate == 50.0

    def test_objection_acceptance_rate_no_objections(self) -> None:
        """Test acceptance rate with no objections."""
        result = ArbitrationResult(report_id="report_001")
        assert result.objection_acceptance_rate == 0.0

    def test_score_improvement(self) -> None:
        """Test MQM score improvement calculation."""
        result = ArbitrationResult(
            report_id="report_001",
            original_mqm_score=85.0,
            final_mqm_score=92.5,
        )
        assert result.score_improvement == 7.5

    def test_score_improvement_negative(self) -> None:
        """Test negative score improvement (shouldn't happen but test it)."""
        result = ArbitrationResult(
            report_id="report_001",
            original_mqm_score=95.0,
            final_mqm_score=90.0,
        )
        assert result.score_improvement == -5.0


class TestArbitrationWorkflow:
    """Comprehensive tests for ArbitrationWorkflow."""

    def test_init_default_threshold(self) -> None:
        """Test default auto-approve threshold."""
        workflow = ArbitrationWorkflow()
        assert workflow.auto_approve_threshold == 0.9

    def test_init_custom_threshold(self) -> None:
        """Test custom auto-approve threshold."""
        workflow = ArbitrationWorkflow(auto_approve_threshold=0.8)
        assert workflow.auto_approve_threshold == 0.8

    def test_start_arbitration(self) -> None:
        """Test starting arbitration for a QA report."""
        # Create mock report
        mock_report = MagicMock()
        mock_report.id = "report_123"
        mock_report.errors = [MagicMock(), MagicMock(), MagicMock()]
        mock_report.mqm_score = 85.5
        mock_report.source_lang = "en"
        mock_report.target_lang = "ru"

        workflow = ArbitrationWorkflow()
        result = workflow.start_arbitration(mock_report)

        assert result.report_id == "report_123"
        assert result.status == ArbitrationStatus.DRAFT
        assert result.original_error_count == 3
        assert result.final_error_count == 3
        assert result.original_mqm_score == 85.5
        assert result.metadata["source_lang"] == "en"
        assert result.metadata["target_lang"] == "ru"

    def test_start_arbitration_no_id(self) -> None:
        """Test starting arbitration for report without ID."""
        mock_report = MagicMock(spec=[])  # No 'id' attribute
        mock_report.errors = []
        mock_report.mqm_score = 100.0

        workflow = ArbitrationWorkflow()
        result = workflow.start_arbitration(mock_report)

        # Should use object id as fallback
        assert result.report_id is not None

    def test_submit_objection(self) -> None:
        """Test submitting an objection."""
        workflow = ArbitrationWorkflow()
        result = ArbitrationResult(report_id="report_001")

        objection = workflow.submit_objection(
            result=result,
            error_id="err_001",
            reason="This is standard Russian punctuation",
            evidence=["Russian punctuation guide, section 5"],
            proposed_resolution="Remove error",
            translator_id="trans_123",
        )

        assert objection.error_id == "err_001"
        assert objection.reason == "This is standard Russian punctuation"
        assert len(objection.evidence) == 1
        assert objection.translator_id == "trans_123"
        assert result.status == ArbitrationStatus.PENDING_ARBITRATION
        assert len(result.objections) == 1

    def test_submit_objection_minimal(self) -> None:
        """Test submitting objection with minimal data."""
        workflow = ArbitrationWorkflow()
        result = ArbitrationResult(report_id="report_001")

        objection = workflow.submit_objection(
            result=result,
            error_id="err_001",
            reason="Disagree with assessment",
        )

        assert objection.evidence == []
        assert objection.proposed_resolution is None
        assert objection.translator_id == ""

    def test_decide_manually_dismissed(self) -> None:
        """Test manual decision to dismiss error."""
        workflow = ArbitrationWorkflow()
        objection = Objection(error_id="err_001", reason="Test")

        workflow.decide_manually(
            objection=objection,
            decision=ArbitrationDecision.DISMISSED,
            reason="Valid cultural adaptation",
        )

        assert objection.status == ObjectionStatus.ACCEPTED
        assert objection.decision == ArbitrationDecision.DISMISSED
        assert "cultural adaptation" in objection.decision_reason

    def test_decide_manually_upheld(self) -> None:
        """Test manual decision to uphold error."""
        workflow = ArbitrationWorkflow()
        objection = Objection(error_id="err_001", reason="Test")

        workflow.decide_manually(
            objection=objection,
            decision=ArbitrationDecision.UPHELD,
            reason="Error correctly identified",
        )

        assert objection.status == ObjectionStatus.REJECTED
        assert objection.decision == ArbitrationDecision.UPHELD

    def test_decide_manually_reduced(self) -> None:
        """Test manual decision to reduce severity."""
        workflow = ArbitrationWorkflow()
        objection = Objection(error_id="err_001", reason="Test")

        workflow.decide_manually(
            objection=objection,
            decision=ArbitrationDecision.REDUCED,
            reason="Changed from major to minor",
        )

        assert objection.status == ObjectionStatus.PARTIAL
        assert objection.decision == ArbitrationDecision.REDUCED

    def test_decide_manually_escalated(self) -> None:
        """Test manual decision to escalate."""
        workflow = ArbitrationWorkflow()
        objection = Objection(error_id="err_001", reason="Test")

        workflow.decide_manually(
            objection=objection,
            decision=ArbitrationDecision.ESCALATED,
            reason="Needs senior reviewer",
        )

        assert objection.decision == ArbitrationDecision.ESCALATED
        assert objection.resolved_at is not None

    def test_export_summary(self) -> None:
        """Test exporting arbitration summary."""
        result = ArbitrationResult(
            report_id="report_001",
            status=ArbitrationStatus.COMPLETED,
            original_error_count=10,
            final_error_count=7,
            original_mqm_score=85.0,
            final_mqm_score=91.0,
        )

        # Add some objections
        obj1 = Objection(error_id="err_001", reason="Test")
        obj1.accept()
        result.add_objection(obj1)
        result.finalize()

        workflow = ArbitrationWorkflow()
        summary = workflow.export_summary(result)

        assert summary["report_id"] == "report_001"
        assert summary["status"] == "completed"
        assert summary["original_errors"] == 10
        assert summary["final_errors"] == 7
        assert summary["errors_dismissed"] == 3
        assert summary["objections_total"] == 1
        assert summary["original_mqm_score"] == 85.0
        assert summary["final_mqm_score"] == 91.0
        assert summary["score_improvement"] == 6.0

    def test_update_final_counts(self) -> None:
        """Test updating final counts after arbitration."""
        workflow = ArbitrationWorkflow()

        # Create mock report
        mock_report = MagicMock()
        mock_report.errors = [MagicMock() for _ in range(10)]
        mock_report.mqm_score = 80.0

        result = ArbitrationResult(
            report_id="report_001",
            original_error_count=10,
            original_mqm_score=80.0,
        )

        # 3 objections dismissed
        for i in range(3):
            obj = Objection(error_id=f"err_{i}", reason="Test")
            obj.accept()  # Sets decision to DISMISSED
            result.add_objection(obj)

        workflow._update_final_counts(result, mock_report)

        assert result.final_error_count == 7  # 10 - 3 dismissed
        assert result.final_mqm_score > result.original_mqm_score


class TestArbitrationWorkflowAI:
    """Tests for AI-assisted arbitration (with mocked LLM)."""

    @pytest.mark.asyncio
    async def test_process_with_ai_success(self) -> None:
        """Test AI-assisted arbitration with successful response."""
        workflow = ArbitrationWorkflow(auto_approve_threshold=0.8)

        # Create mock LLM provider
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = """
        {
            "decision": "dismissed",
            "reason": "Translation is culturally appropriate for Russian",
            "confidence": 0.95,
            "suggested_severity": null
        }
        """

        # Create mock report and result
        mock_error = MagicMock()
        mock_error.id = "err_001"
        mock_error.category = "fluency"
        mock_error.severity = "major"
        mock_error.description = "Awkward phrasing"
        mock_error.location = "segment 5"

        mock_report = MagicMock()
        mock_report.id = "report_ai_001"
        mock_report.errors = [mock_error]
        mock_report.mqm_score = 85.0
        mock_report.source_lang = "en"
        mock_report.target_lang = "ru"

        result = workflow.start_arbitration(mock_report)
        workflow.submit_objection(
            result=result,
            error_id="err_001",
            reason="This is natural in Russian",
        )

        # Process with AI
        updated_result = await workflow.process_with_ai(result, mock_llm, mock_report)

        # Verify
        assert updated_result.status == ArbitrationStatus.COMPLETED
        assert mock_llm.complete.called
        objection = updated_result.objections[0]
        assert objection.decision == ArbitrationDecision.DISMISSED

    @pytest.mark.asyncio
    async def test_process_with_ai_low_confidence(self) -> None:
        """Test that low confidence leads to escalation."""
        workflow = ArbitrationWorkflow(auto_approve_threshold=0.9)

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = """
        {
            "decision": "dismissed",
            "reason": "Might be acceptable",
            "confidence": 0.7
        }
        """

        mock_error = MagicMock()
        mock_error.id = "err_001"
        mock_error.category = "accuracy"

        mock_report = MagicMock()
        mock_report.id = "report_ai_002"
        mock_report.errors = [mock_error]
        mock_report.mqm_score = 90.0

        result = workflow.start_arbitration(mock_report)
        workflow.submit_objection(result=result, error_id="err_001", reason="Test")

        await workflow.process_with_ai(result, mock_llm, mock_report)

        # Should be escalated due to low confidence
        objection = result.objections[0]
        assert objection.decision == ArbitrationDecision.ESCALATED
        assert "Low confidence" in objection.decision_reason

    @pytest.mark.asyncio
    async def test_process_with_ai_invalid_json(self) -> None:
        """Test handling of invalid JSON response from LLM."""
        workflow = ArbitrationWorkflow()

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "This is not JSON at all"

        mock_error = MagicMock()
        mock_error.id = "err_001"

        mock_report = MagicMock()
        mock_report.id = "report_ai_003"
        mock_report.errors = [mock_error]
        mock_report.mqm_score = 90.0

        result = workflow.start_arbitration(mock_report)
        workflow.submit_objection(result=result, error_id="err_001", reason="Test")

        await workflow.process_with_ai(result, mock_llm, mock_report)

        # Should be escalated due to parse error
        objection = result.objections[0]
        assert objection.decision == ArbitrationDecision.ESCALATED

    @pytest.mark.asyncio
    async def test_process_with_ai_error_not_found(self) -> None:
        """Test handling of objection for non-existent error."""
        workflow = ArbitrationWorkflow()

        mock_llm = AsyncMock()
        mock_report = MagicMock()
        mock_report.id = "report_ai_004"
        mock_report.errors = []  # No errors
        mock_report.mqm_score = 100.0

        result = workflow.start_arbitration(mock_report)
        # Manually add objection for non-existent error
        result.add_objection(Objection(error_id="nonexistent", reason="Test"))

        await workflow.process_with_ai(result, mock_llm, mock_report)

        # LLM should not be called for non-existent errors
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_with_ai_exception(self) -> None:
        """Test handling of LLM exception."""
        workflow = ArbitrationWorkflow()

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = Exception("API Error")

        mock_error = MagicMock()
        mock_error.id = "err_001"

        mock_report = MagicMock()
        mock_report.id = "report_ai_005"
        mock_report.errors = [mock_error]
        mock_report.mqm_score = 90.0

        result = workflow.start_arbitration(mock_report)
        workflow.submit_objection(result=result, error_id="err_001", reason="Test")

        # Should not raise, but escalate
        await workflow.process_with_ai(result, mock_llm, mock_report)

        objection = result.objections[0]
        assert objection.decision == ArbitrationDecision.ESCALATED
        assert "AI review failed" in objection.decision_reason
