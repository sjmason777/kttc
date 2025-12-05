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

"""Trigger manager for orchestrating QA triggers.

Handles trigger registration, evaluation, and execution.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .conditions import BaseCondition
from .models import (
    TRIGGER_TEMPLATES,
    Trigger,
    TriggerAction,
    TriggerEvent,
    TriggerResult,
)

logger = logging.getLogger(__name__)


class TriggerManager:
    """Manages QA triggers and their execution.

    Example:
        >>> manager = TriggerManager()
        >>> manager.create_from_template("ci_cd_check")
        >>> results = await manager.fire_event(
        ...     TriggerEvent.FILE_MODIFIED,
        ...     {"file_path": "translations.xliff"}
        ... )
    """

    def __init__(self) -> None:
        """Initialize trigger manager."""
        self.triggers: dict[str, Trigger] = {}
        self.action_handlers: dict[TriggerAction, Callable[..., Any]] = {}

    def register(self, trigger: Trigger) -> None:
        """Register a trigger.

        Args:
            trigger: Trigger to register
        """
        self.triggers[trigger.id] = trigger
        logger.info("Registered trigger: %s (%s)", trigger.id, trigger.name)

    def unregister(self, trigger_id: str) -> bool:
        """Unregister a trigger.

        Args:
            trigger_id: ID of trigger to remove

        Returns:
            True if trigger was removed
        """
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            logger.info("Unregistered trigger: %s", trigger_id)
            return True
        return False

    def get(self, trigger_id: str) -> Trigger | None:
        """Get trigger by ID.

        Args:
            trigger_id: Trigger identifier

        Returns:
            Trigger if found, None otherwise
        """
        return self.triggers.get(trigger_id)

    def list_triggers(self, event: TriggerEvent | None = None) -> list[Trigger]:
        """List all triggers, optionally filtered by event.

        Args:
            event: Filter by event type (optional)

        Returns:
            List of triggers
        """
        triggers = list(self.triggers.values())

        if event is not None:
            triggers = [t for t in triggers if t.event == event]

        # Sort by priority (highest first)
        return sorted(triggers, key=lambda t: t.priority, reverse=True)

    def create_from_template(
        self,
        template_name: str,
        trigger_id: str | None = None,
        **overrides: Any,
    ) -> Trigger:
        """Create trigger from predefined template.

        Args:
            template_name: Name of template (ci_cd_check, threshold_alert, etc.)
            trigger_id: Custom ID (defaults to template name)
            **overrides: Override template values

        Returns:
            Created trigger

        Raises:
            ValueError: If template not found
        """
        if template_name not in TRIGGER_TEMPLATES:
            available = ", ".join(TRIGGER_TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")

        template = TRIGGER_TEMPLATES[template_name].copy()
        template.update(overrides)

        trigger = Trigger(
            id=trigger_id or template_name,
            **template,
        )

        self.register(trigger)
        return trigger

    def register_action_handler(
        self,
        action: TriggerAction,
        handler: Callable[[Trigger, dict[str, Any]], Any],
    ) -> None:
        """Register handler for an action type.

        Args:
            action: Action type
            handler: Callable that handles the action
        """
        self.action_handlers[action] = handler
        logger.debug("Registered handler for action: %s", action.value)

    async def fire_event(
        self,
        event: TriggerEvent,
        context: dict[str, Any],
    ) -> list[TriggerResult]:
        """Fire an event and evaluate matching triggers.

        Args:
            event: Event that occurred
            context: Context data for condition evaluation

        Returns:
            List of trigger results
        """
        results: list[TriggerResult] = []
        matching_triggers = self.list_triggers(event=event)

        for trigger in matching_triggers:
            result = await self._evaluate_trigger(trigger, event, context)
            results.append(result)

        fired_count = sum(1 for r in results if r.fired)
        logger.info(
            "Event %s: %d triggers evaluated, %d fired",
            event.value,
            len(results),
            fired_count,
        )

        return results

    async def _evaluate_trigger(
        self,
        trigger: Trigger,
        event: TriggerEvent,
        context: dict[str, Any],
    ) -> TriggerResult:
        """Evaluate a single trigger.

        Args:
            trigger: Trigger to evaluate
            event: Event that occurred
            context: Context data

        Returns:
            Trigger result
        """
        result = TriggerResult(
            trigger_id=trigger.id,
            event=event,
        )

        # Check if trigger should fire (cooldown, enabled)
        if not trigger.should_fire():
            result.details["reason"] = "Trigger disabled or in cooldown"
            return result

        # Evaluate conditions
        conditions_met = self._evaluate_conditions(trigger.conditions, context)

        if not conditions_met:
            result.details["reason"] = "Conditions not met"
            return result

        # Trigger fires!
        result.fired = True
        trigger.record_fire()

        # Execute actions
        actions_taken = await self._execute_actions(trigger, context)
        result.actions_taken = actions_taken

        logger.info(
            "Trigger %s fired: %d actions executed",
            trigger.id,
            len(actions_taken),
        )

        return result

    def _evaluate_conditions(
        self,
        conditions: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> bool:
        """Evaluate all conditions for a trigger.

        Args:
            conditions: List of condition configurations
            context: Context data

        Returns:
            True if all conditions are met
        """
        if not conditions:
            return True

        for condition_config in conditions:
            try:
                condition = BaseCondition.from_dict(condition_config)
                if not condition.evaluate(context):
                    return False
            except ValueError as e:
                logger.warning("Failed to evaluate condition: %s", e)
                continue

        return True

    async def _execute_actions(
        self,
        trigger: Trigger,
        context: dict[str, Any],
    ) -> list[TriggerAction]:
        """Execute all actions for a fired trigger.

        Args:
            trigger: The fired trigger
            context: Context data

        Returns:
            List of successfully executed actions
        """
        executed: list[TriggerAction] = []

        for action in trigger.actions:
            try:
                if action in self.action_handlers:
                    handler = self.action_handlers[action]
                    # Handle both sync and async handlers
                    result = handler(trigger, context)
                    if hasattr(result, "__await__"):
                        await result
                else:
                    # Default action handling
                    await self._default_action_handler(action, trigger, context)

                executed.append(action)

            except Exception as e:
                logger.error(
                    "Action %s failed for trigger %s: %s",
                    action.value,
                    trigger.id,
                    e,
                )

        return executed

    async def _default_action_handler(
        self,
        action: TriggerAction,
        trigger: Trigger,
        context: dict[str, Any],
    ) -> None:
        """Default handler for actions without custom handlers.

        Args:
            action: Action type
            trigger: The trigger
            context: Context data
        """
        if action == TriggerAction.LOG_ONLY:
            logger.info(
                "Trigger %s: %s (context: %s)",
                trigger.id,
                trigger.name,
                context,
            )
        elif action == TriggerAction.RUN_CHECK:
            # Store in context for external handling
            context["_run_check_requested"] = True
            context["_trigger_id"] = trigger.id
        elif action == TriggerAction.SEND_NOTIFICATION:
            # Log notification request
            logger.warning(
                "Notification requested by trigger %s: %s",
                trigger.id,
                trigger.description,
            )
        elif action == TriggerAction.BLOCK_DELIVERY:
            # Store block request
            context["_block_delivery"] = True
            logger.warning("Delivery blocked by trigger %s", trigger.id)
        elif action == TriggerAction.ESCALATE:
            # Store escalation request
            context["_escalate"] = True
            logger.warning(
                "Escalation requested by trigger %s: %s",
                trigger.id,
                trigger.description,
            )

    def enable(self, trigger_id: str) -> bool:
        """Enable a trigger.

        Args:
            trigger_id: Trigger to enable

        Returns:
            True if trigger was enabled
        """
        trigger = self.triggers.get(trigger_id)
        if trigger:
            trigger.enabled = True
            logger.info("Enabled trigger: %s", trigger_id)
            return True
        return False

    def disable(self, trigger_id: str) -> bool:
        """Disable a trigger.

        Args:
            trigger_id: Trigger to disable

        Returns:
            True if trigger was disabled
        """
        trigger = self.triggers.get(trigger_id)
        if trigger:
            trigger.enabled = False
            logger.info("Disabled trigger: %s", trigger_id)
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get trigger statistics.

        Returns:
            Dictionary with trigger stats
        """
        triggers = list(self.triggers.values())

        return {
            "total_triggers": len(triggers),
            "enabled_triggers": sum(1 for t in triggers if t.enabled),
            "disabled_triggers": sum(1 for t in triggers if not t.enabled),
            "total_fires": sum(t.fire_count for t in triggers),
            "by_event": {
                event.value: sum(1 for t in triggers if t.event == event) for event in TriggerEvent
            },
            "by_action": {
                action.value: sum(1 for t in triggers for a in t.actions if a == action)
                for action in TriggerAction
            },
        }

    def export_config(self) -> dict[str, Any]:
        """Export all triggers as configuration.

        Returns:
            Dictionary with all trigger configurations
        """
        return {
            trigger_id: {
                "name": t.name,
                "description": t.description,
                "enabled": t.enabled,
                "event": t.event.value,
                "conditions": t.conditions,
                "actions": [a.value for a in t.actions],
                "priority": t.priority,
                "cooldown_seconds": t.cooldown_seconds,
                "fire_count": t.fire_count,
                "last_fired": t.last_fired.isoformat() if t.last_fired else None,
            }
            for trigger_id, t in self.triggers.items()
        }

    def import_config(self, config: dict[str, Any]) -> int:
        """Import triggers from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Number of triggers imported
        """
        count = 0

        for trigger_id, trigger_config in config.items():
            try:
                # Convert event string to enum
                event = TriggerEvent(trigger_config["event"])

                # Convert action strings to enums
                actions = [TriggerAction(a) for a in trigger_config.get("actions", [])]

                trigger = Trigger(
                    id=trigger_id,
                    name=trigger_config["name"],
                    description=trigger_config.get("description", ""),
                    enabled=trigger_config.get("enabled", True),
                    event=event,
                    conditions=trigger_config.get("conditions", []),
                    actions=actions or [TriggerAction.RUN_CHECK],
                    priority=trigger_config.get("priority", 0),
                    cooldown_seconds=trigger_config.get("cooldown_seconds", 0),
                )

                self.register(trigger)
                count += 1

            except (KeyError, ValueError) as e:
                logger.error("Failed to import trigger %s: %s", trigger_id, e)

        logger.info("Imported %d triggers", count)
        return count
