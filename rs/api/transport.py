from __future__ import annotations

import sys
from typing import IO, Protocol

from rs.helper.logger import log, log_to_run


class TransportError(RuntimeError):
    """Raised when the underlying game transport cannot exchange data."""


class ProtocolError(RuntimeError):
    """Raised when transport data violates the expected line protocol."""


class GameTransport(Protocol):
    def send(self, message: str, *, silent: bool = False, before_run: bool = False) -> str:
        """Send one protocol line and return one response line."""


def _log_exchange(message: str, *, incoming: bool, before_run: bool) -> None:
    prefix = "Response" if incoming else "Sending message"
    log_message = f"{prefix}: {message}"
    if before_run:
        log(log_message)
    else:
        log_to_run(log_message)


class StdioGameTransport:
    """CommunicationMod transport over process stdin/stdout."""

    def __init__(
            self,
            reader: IO[str] | None = None,
            writer: IO[str] | None = None,
    ):
        self._reader = sys.stdin if reader is None else reader
        self._writer = sys.stdout if writer is None else writer

    def send(self, message: str, *, silent: bool = False, before_run: bool = False) -> str:
        if not silent:
            _log_exchange(message, incoming=False, before_run=before_run)

        self._writer.write(message + "\n")
        self._writer.flush()

        raw_response = self._reader.readline()
        if raw_response == "":
            raise TransportError("CommunicationMod transport closed while waiting for response")

        response = raw_response.rstrip("\r\n")
        if response == "":
            raise ProtocolError("CommunicationMod returned an empty response line")

        if not silent:
            _log_exchange(response, incoming=True, before_run=before_run)
        return response
