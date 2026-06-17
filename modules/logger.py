import logging
import logging.handlers
import os
import re

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

_INJECT_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

class SafeFormatter(logging.Formatter):
    """Formatter that strips control characters to prevent log injection."""

    def format(self, record):
        msg = super().format(record)
        return _INJECT_CHARS.sub('', msg)


class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, '')
        orig_levelname = record.levelname
        if color:
            record.levelname = f"{color}{orig_levelname}{self.RESET}"
        res = super().format(record)
        record.levelname = orig_levelname
        return res

def setup_logger(name, log_file, level=logging.INFO):
    """Configures a logger with file and console outputs."""
    handler = logging.handlers.RotatingFileHandler(
        log_file, mode='a', encoding='utf-8',
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    handler.setFormatter(SafeFormatter('%(asctime)s - %(levelname)s - %(message)s'))

    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

app_logger = setup_logger('app', 'logs/app.log')
