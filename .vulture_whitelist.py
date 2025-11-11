"""Vulture whitelist for false positives.

This file contains intentional uses of code that Vulture flags as unused
but is actually required by frameworks (Pydantic, FastAPI, TypedDict).
"""

# Pydantic model_config (used internally by Pydantic v2)
model_config = None

# Pydantic Field objects (used in model definitions)
Field = None

# TypedDict fields (used for type hints)
total_evaluations = None
total_mqm_score = None
start_time = None
active_websockets = None
errors_count = None
average_mqm_score = None
uptime_seconds = None

# FastAPI response model fields
document_context = None

# Pydantic BaseModel fields used via JSON/dict serialization
agent_details = None
score_breakdown = None

# Model configuration fields
id = None
created_at = None
usage_count = None
found_in_translation = None

# Settings fields accessed via dict-like interface
request_timeout = None
max_retries = None
mqm_pass_threshold = None
log_level = None

# Domain adapter fields
style_guidelines = None

# Unused variables that are part of TypedDict definitions
reason = None
iterations = None
task_type = None

# SQLite row_factory attribute (used by aiosqlite)
row_factory = None

# CLI variables - used by Typer callbacks (is_eager=True)
version = None  # Used in main() callback

# ModelSelector parameters - part of public API
task_type = None  # Reserved for future model selection logic

# Constants defined for future use
RUSSIAN_CHECKS = None
FACTUAL_CONSISTENCY_THRESHOLD = None
