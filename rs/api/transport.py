from __future__ import annotations

import os
import socket
import sys
from typing import IO, Protocol

import json as _json
from pathlib import Path

from definitions import ROOT_DIR
from rs.helper.logger import log, log_to_run


COMMUNICATIONMOD_TRANSPORT_ENV = "COMMUNICATIONMOD_TRANSPORT"
_EXCHANGE_LOG_PATH = Path(ROOT_DIR) / "logs" / "game_state_exchange.jsonl"


def _append_exchange_jsonl(direction: str, message: str) -> None:
    _EXCHANGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _json.loads(message)
    except (ValueError, TypeError):
        payload = message
    entry = _json.dumps({"direction": direction, "data": payload}, ensure_ascii=False)
    with _EXCHANGE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry + "\n")
COMMUNICATIONMOD_HOST_ENV = "COMMUNICATIONMOD_HOST"
COMMUNICATIONMOD_PORT_ENV = "COMMUNICATIONMOD_PORT"
COMMUNICATIONMOD_CONNECT_TIMEOUT_ENV = "COMMUNICATIONMOD_CONNECT_TIMEOUT_SECONDS"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 120.0


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
        log_to_run(log_message, console=False)
    _append_exchange_jsonl(prefix, message)


def _send_and_receive_line(
        *,
        message: str,
        reader: IO[str],
        writer: IO[str],
        silent: bool,
        before_run: bool,
) -> str:
    if not silent:
        _log_exchange(message, incoming=False, before_run=before_run)

    writer.write(message + "\n")
    writer.flush()

    raw_response = reader.readline()
    if raw_response == "":
        raise TransportError("CommunicationMod transport closed while waiting for response")

    response = raw_response.rstrip("\r\n")
    if response == "":
        raise ProtocolError("CommunicationMod returned an empty response line")

    if not silent:
        _log_exchange(response, incoming=True, before_run=before_run)
    return response


class StdioGameTransport:
    """Legacy CommunicationMod transport over process stdin/stdout."""

    def __init__(
            self,
            reader: IO[str] | None = None,
            writer: IO[str] | None = None,
    ):
        self._reader = sys.stdin if reader is None else reader
        self._writer = sys.stdout if writer is None else writer

    def send(self, message: str, *, silent: bool = False, before_run: bool = False) -> str:
        return _send_and_receive_line(
            message=message,
            reader=self._reader,
            writer=self._writer,
            silent=silent,
            before_run=before_run,
        )


class SocketGameTransport:
    """CommunicationMod transport over a localhost TCP socket."""

    def __init__(
            self,
            host: str,
            port: int,
            connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
            sock: socket.socket | None = None,
    ):
        self._host = host
        self._port = port
        self._connect_timeout_seconds = connect_timeout_seconds
        self._socket = sock or self._connect()
        self._reader = self._socket.makefile("r", encoding="utf-8", newline="\n")
        self._writer = self._socket.makefile("w", encoding="utf-8", newline="\n")

    @classmethod
    def from_environment(cls) -> "SocketGameTransport":
        transport = os.environ.get(COMMUNICATIONMOD_TRANSPORT_ENV, "socket").strip().lower()
        if transport != "socket":
            raise TransportError(f"Unsupported CommunicationMod transport: {transport}")

        host = os.environ.get(COMMUNICATIONMOD_HOST_ENV)
        if not host:
            raise TransportError(
                f"Missing {COMMUNICATIONMOD_HOST_ENV}; CommunicationMod socket transport is not configured"
            )

        raw_port = os.environ.get(COMMUNICATIONMOD_PORT_ENV)
        if not raw_port:
            raise TransportError(
                f"Missing {COMMUNICATIONMOD_PORT_ENV}; CommunicationMod socket transport is not configured"
            )
        try:
            port = int(raw_port)
        except ValueError as error:
            raise TransportError(
                f"Invalid {COMMUNICATIONMOD_PORT_ENV} value: {raw_port}"
            ) from error

        raw_timeout = os.environ.get(COMMUNICATIONMOD_CONNECT_TIMEOUT_ENV, str(DEFAULT_CONNECT_TIMEOUT_SECONDS))
        try:
            connect_timeout = float(raw_timeout)
        except ValueError as error:
            raise TransportError(
                f"Invalid {COMMUNICATIONMOD_CONNECT_TIMEOUT_ENV} value: {raw_timeout}"
            ) from error

        return cls(host=host, port=port, connect_timeout_seconds=connect_timeout)

    def _connect(self) -> socket.socket:
        try:
            sock = socket.create_connection((self._host, self._port), timeout=self._connect_timeout_seconds)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as error:
            raise TransportError(
                f"Could not connect to CommunicationMod socket at {self._host}:{self._port}: {error}"
            ) from error

    def send(self, message: str, *, silent: bool = False, before_run: bool = False) -> str:
        try:
            return _send_and_receive_line(
                message=message,
                reader=self._reader,
                writer=self._writer,
                silent=silent,
                before_run=before_run,
            )
        except OSError as error:
            raise TransportError(
                f"CommunicationMod socket transport failed while exchanging data: {error}"
            ) from error

    def close(self) -> None:
        try:
            self._reader.close()
        finally:
            try:
                self._writer.close()
            finally:
                self._socket.close()
