import sys
import unittest
from unittest.mock import patch

import rs.helper.logger as logger_module


class TestLoggerConfiguration(unittest.TestCase):
    def test_configure_console_logging_routes_loguru_to_stderr(self):
        with patch.object(logger_module.loguru_logger, "remove") as remove_mock, patch.object(
                logger_module.loguru_logger,
                "add",
        ) as add_mock:
            previous = logger_module._console_logger_configured
            logger_module._console_logger_configured = False
            try:
                logger_module.configure_console_logging()
            finally:
                logger_module._console_logger_configured = previous

        remove_mock.assert_called_once()
        add_mock.assert_called_once_with(sys.stderr)

    def test_configure_console_logging_is_idempotent(self):
        with patch.object(logger_module.loguru_logger, "remove") as remove_mock, patch.object(
                logger_module.loguru_logger,
                "add",
        ) as add_mock:
            previous = logger_module._console_logger_configured
            logger_module._console_logger_configured = True
            try:
                logger_module.configure_console_logging()
            finally:
                logger_module._console_logger_configured = previous

        remove_mock.assert_not_called()
        add_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
