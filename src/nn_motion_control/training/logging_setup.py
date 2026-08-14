import json
import logging
import logging.config
import logging.handlers
import os

# Rotate the log at 10 MiB, keeping this many prior files.
MAX_LOG_BYTES = 10_485_760
LOG_BACKUP_COUNT = 2


class ModelLogger:
    def __init__(self, logging_path, timestamp):
        self.logging_path = logging_path
        self.timestamp = timestamp
        self.setup_logging()

    def setup_logging(self):
        """
        Build a file handler with a run-unique log filename.

        The json logging config cannot express a per-run filename,
        so this constructs the RotatingFileHandler by hand and appends
        it to the root logger after dictConfig sets up everything else.
        """

        with open("logging.json") as f:
            config = json.load(f)

        os.makedirs(self.logging_path, exist_ok=True)
        filename = f"{self.logging_path}/{self.timestamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=filename,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )

        formatter = config["formatters"]["standard"]["format"]
        file_handler.setFormatter(logging.Formatter(formatter))
        file_handler.setLevel(logging.DEBUG)

        logging.config.dictConfig(config)

        root = logging.getLogger()
        root.addHandler(file_handler)
