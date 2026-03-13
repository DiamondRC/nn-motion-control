import json
import logging
import logging.config
import logging.handlers


def setup_logging(logging_path, timestamp):
    """
    We want logs to autogenerate with a unqiue name per run,
    which is not possible to specify in the json config.

    This is something of a hacky workaround. Create a new
    file handler with hardcoded properties, such as the
    dynamically assigned filename.

    TODO - might be able to achieve the same thing with
    something like jinja?
    """

    # Load existing config
    with open("logging.json") as f:
        config = json.load(f)

    filename = f"{logging_path}/{timestamp}.log"

    # Set properties of the logging json
    file_handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        maxBytes=10485760,
        backupCount=2,
    )

    # Use the "standard" formatter from JSON config
    # This is a bit hacky!
    formatter = config["formatters"]["standard"]["format"]
    file_handler.setFormatter(logging.Formatter(formatter))
    file_handler.setLevel(logging.DEBUG)

    logging.config.dictConfig(config)

    # Append handler to root logger
    root = logging.getLogger()
    root.addHandler(file_handler)
