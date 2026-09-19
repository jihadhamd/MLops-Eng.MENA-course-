"""Structured JSON logging configuration."""

import json
import logging
import uuid
from contextvars import ContextVar

request_id_cv: ContextVar[str] = ContextVar("request_id", default="startup")


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": request_id_cv.get(),
        }
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_dict)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to emit JSON lines."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def new_correlation_id() -> str:
    """Generate and set a new correlation ID for the current context."""
    cid = str(uuid.uuid4())
    request_id_cv.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the correlation ID for the current context."""
    return request_id_cv.get()
