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

"""Data models for QA triggers."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TriggerEvent(str, Enum):
    """Events that can trigger QA checks."""

    FILE_CREATED = "file_created"  # New translation file added
    FILE_MODIFIED = "file_modified"  # Translation file changed
    BATCH_COMPLETE = "batch_complete"  # Batch job finished
    THRESHOLD_VIOLATED = "threshold_violated"  # Score below threshold
    SCHEDULE = "schedule"  # Scheduled check
    MANUAL = "manual"  # Manual trigger
    WEBHOOK = "webhook"  # External webhook


class TriggerAction(str, Enum):
    """Actions to take when trigger fires."""

    RUN_CHECK = "run_check"  # Run QA check
    SEND_NOTIFICATION = "send_notification"  # Send alert
    BLOCK_DELIVERY = "block_delivery"  # Prevent delivery
    LOG_ONLY = "log_only"  # Just log the event
    ESCALATE = "escalate"  # Escalate to reviewer


class Trigger(BaseModel):
    """QA trigger definition.

    Attributes:
        id: Unique trigger identifier
        name: Human-readable name
        description: What this trigger does
        enabled: Whether trigger is active
        event: Event that fires this trigger
        conditions: Conditions that must be met
        actions: Actions to take when fired
        priority: Trigger priority (higher = first)
        cooldown_seconds: Minimum time between fires
        metadata: Additional configuration
        created_at: When trigger was created
        last_fired: When trigger last fired
        fire_count: Total times fired
    """

    id: str = Field(..., description="Unique trigger ID")
    name: str = Field(..., description="Trigger name")
    description: str = Field(default="", description="Trigger description")
    enabled: bool = Field(default=True, description="Is trigger active")
    event: TriggerEvent = Field(..., description="Triggering event")
    conditions: list[dict[str, Any]] = Field(
        default_factory=list, description="Conditions to evaluate"
    )
    actions: list[TriggerAction] = Field(
        default_factory=lambda: [TriggerAction.RUN_CHECK],
        description="Actions to take",
    )
    priority: int = Field(default=0, ge=0, le=100, description="Priority level")
    cooldown_seconds: int = Field(default=0, ge=0, description="Cooldown between fires")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_fired: datetime | None = Field(default=None)
    fire_count: int = Field(default=0, ge=0)

    def should_fire(self) -> bool:
        """Check if trigger should fire based on cooldown."""
        if not self.enabled:
            return False

        if self.last_fired is None:
            return True

        if self.cooldown_seconds == 0:
            return True

        elapsed = (datetime.now() - self.last_fired).total_seconds()
        return elapsed >= self.cooldown_seconds

    def record_fire(self) -> None:
        """Record that trigger has fired."""
        self.last_fired = datetime.now()
        self.fire_count += 1


class TriggerResult(BaseModel):
    """Result of trigger evaluation.

    Attributes:
        trigger_id: ID of the trigger
        fired: Whether trigger fired
        event: The event that occurred
        actions_taken: Actions that were executed
        details: Additional result details
        timestamp: When evaluation occurred
    """

    trigger_id: str
    fired: bool = False
    event: TriggerEvent | None = None
    actions_taken: list[TriggerAction] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# Pre-built trigger templates
TRIGGER_TEMPLATES: dict[str, dict[str, Any]] = {
    "ci_cd_check": {
        "name": "CI/CD QA Check",
        "description": "Run QA check on new/modified translation files in CI/CD",
        "event": TriggerEvent.FILE_MODIFIED,
        "conditions": [{"type": "file_pattern", "patterns": ["*.xliff", "*.tmx", "*.po"]}],
        "actions": [TriggerAction.RUN_CHECK, TriggerAction.BLOCK_DELIVERY],
        "priority": 90,
    },
    "threshold_alert": {
        "name": "Quality Threshold Alert",
        "description": "Alert when MQM score drops below threshold",
        "event": TriggerEvent.THRESHOLD_VIOLATED,
        "conditions": [{"type": "score_threshold", "metric": "mqm", "min_score": 85.0}],
        "actions": [TriggerAction.SEND_NOTIFICATION, TriggerAction.ESCALATE],
        "priority": 80,
    },
    "batch_review": {
        "name": "Batch Completion Review",
        "description": "Run QA check when batch translation completes",
        "event": TriggerEvent.BATCH_COMPLETE,
        "conditions": [],
        "actions": [TriggerAction.RUN_CHECK],
        "priority": 50,
    },
    "daily_audit": {
        "name": "Daily Quality Audit",
        "description": "Scheduled daily quality check",
        "event": TriggerEvent.SCHEDULE,
        "conditions": [{"type": "time_based", "schedule": "0 9 * * *"}],  # 9 AM daily
        "actions": [TriggerAction.RUN_CHECK, TriggerAction.LOG_ONLY],
        "priority": 30,
    },
}
