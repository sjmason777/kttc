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

"""Data models for arbitration cycle.

Defines the structures for objections, decisions, and workflow states.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ObjectionStatus(str, Enum):
    """Status of a translator's objection."""

    PENDING = "pending"  # Awaiting arbiter review
    ACCEPTED = "accepted"  # Objection accepted, error dismissed
    REJECTED = "rejected"  # Objection rejected, error stands
    PARTIAL = "partial"  # Partially accepted (e.g., severity reduced)


class ArbitrationStatus(str, Enum):
    """Status of the entire arbitration cycle."""

    DRAFT = "draft"  # Initial QA report, not yet sent
    PENDING_OBJECTIONS = "pending_objections"  # Awaiting translator response
    PENDING_ARBITRATION = "pending_arbitration"  # Awaiting arbiter decision
    COMPLETED = "completed"  # All decisions finalized
    CLOSED = "closed"  # Archived


class ArbitrationDecision(str, Enum):
    """Arbiter's decision on an objection."""

    UPHELD = "upheld"  # Original error assessment upheld
    DISMISSED = "dismissed"  # Error dismissed, objection accepted
    REDUCED = "reduced"  # Error severity reduced
    ESCALATED = "escalated"  # Requires further review


class Objection(BaseModel):
    """Translator's objection to a flagged error.

    Attributes:
        error_id: ID of the error being objected
        translator_id: ID of the translator submitting objection
        reason: Detailed reason for objection
        evidence: Supporting evidence (references, style guides, etc.)
        proposed_resolution: Translator's suggested resolution
        status: Current objection status
        decision: Arbiter's decision (if reviewed)
        decision_reason: Arbiter's reasoning
        created_at: When objection was submitted
        resolved_at: When decision was made
    """

    error_id: str = Field(..., description="ID of the contested error")
    translator_id: str = Field(default="", description="Translator identifier")
    reason: str = Field(..., description="Detailed objection reason")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence")
    proposed_resolution: str | None = Field(default=None, description="Suggested resolution")
    status: ObjectionStatus = Field(default=ObjectionStatus.PENDING)
    decision: ArbitrationDecision | None = Field(default=None)
    decision_reason: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    resolved_at: datetime | None = Field(default=None)

    def accept(self, reason: str = "") -> None:
        """Accept the objection, dismissing the error."""
        self.status = ObjectionStatus.ACCEPTED
        self.decision = ArbitrationDecision.DISMISSED
        self.decision_reason = reason or "Objection accepted"
        self.resolved_at = datetime.now()

    def reject(self, reason: str = "") -> None:
        """Reject the objection, upholding the error."""
        self.status = ObjectionStatus.REJECTED
        self.decision = ArbitrationDecision.UPHELD
        self.decision_reason = reason or "Original assessment upheld"
        self.resolved_at = datetime.now()

    def reduce_severity(self, reason: str = "") -> None:
        """Partially accept, reducing error severity."""
        self.status = ObjectionStatus.PARTIAL
        self.decision = ArbitrationDecision.REDUCED
        self.decision_reason = reason or "Error severity reduced"
        self.resolved_at = datetime.now()


class ArbitrationResult(BaseModel):
    """Result of the complete arbitration cycle.

    Attributes:
        report_id: ID of the original QA report
        status: Overall arbitration status
        original_error_count: Errors before arbitration
        final_error_count: Errors after arbitration
        objections: List of submitted objections
        objections_accepted: Count of accepted objections
        objections_rejected: Count of rejected objections
        original_mqm_score: MQM score before arbitration
        final_mqm_score: MQM score after arbitration
        arbiter_notes: General notes from arbiter
        metadata: Additional metadata
        created_at: When arbitration started
        completed_at: When arbitration finished
    """

    report_id: str = Field(..., description="ID of the QA report")
    status: ArbitrationStatus = Field(default=ArbitrationStatus.DRAFT)
    original_error_count: int = Field(default=0)
    final_error_count: int = Field(default=0)
    objections: list[Objection] = Field(default_factory=list)
    objections_accepted: int = Field(default=0)
    objections_rejected: int = Field(default=0)
    original_mqm_score: float = Field(default=100.0)
    final_mqm_score: float = Field(default=100.0)
    arbiter_notes: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)

    @property
    def objection_acceptance_rate(self) -> float:
        """Calculate objection acceptance rate."""
        total = len(self.objections)
        if total == 0:
            return 0.0
        return round(self.objections_accepted / total * 100, 2)

    @property
    def score_improvement(self) -> float:
        """Calculate MQM score improvement after arbitration."""
        return round(self.final_mqm_score - self.original_mqm_score, 2)

    def add_objection(self, objection: Objection) -> None:
        """Add a new objection to the result."""
        self.objections.append(objection)
        if self.status == ArbitrationStatus.DRAFT:
            self.status = ArbitrationStatus.PENDING_OBJECTIONS

    def finalize(self) -> None:
        """Finalize the arbitration result."""
        self.status = ArbitrationStatus.COMPLETED
        self.completed_at = datetime.now()

        # Update counts
        self.objections_accepted = sum(
            1 for o in self.objections if o.status == ObjectionStatus.ACCEPTED
        )
        self.objections_rejected = sum(
            1 for o in self.objections if o.status == ObjectionStatus.REJECTED
        )
