# Vulture whitelist - False positives for vulture dead code detection
#
# These are not dead code - they are:
# 1. Typer callback parameters (used by framework)
# 2. Interface parameters (required for API compatibility)

# Typer callback parameters - used by Typer framework for CLI options
ui_lang  # main.py - Typer callback parameter
version  # main.py - Typer callback parameter

# API parameters reserved for future use
task_type  # model_selector.py - Reserved for future use
