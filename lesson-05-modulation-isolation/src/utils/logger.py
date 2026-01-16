import logging
from typing import Dict, Any


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(level=level, format=fmt)
    return logging.getLogger(__name__)
