# config.py
from pydantic import BaseModel

class ExecutionAgentConfig(BaseModel):
    system_template: str
    instance_template: str
    action_observation_template: str
    format_error_template: str
    timeout_template: str

    # Parse exactly one JSON code block
    action_regex: str = r"```json\s*\n(.*?)\n```"

    # Step/cost limits
    step_limit: int = 0
    cost_limit: float = 0.0

    # ExecutionAgent-specific
    repetition_window: int = 6
    summarizer_model_name: str | None = None   # optional separate model
    summary_max_chars: int = 4000
