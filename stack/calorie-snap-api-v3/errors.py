"""Shared exception type."""


class PipelineError(Exception):
    """A failure attributable to one pipeline stage.

    Attributes:
        stage: One of config, llm_http, llm_parse, local_vlm, db.
        message: Human-readable cause, safe to show in the UI.
        status: HTTP status the API should return.
    """

    def __init__(self, stage: str, message: str, status: int = 502):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.status = status
