from __future__ import annotations

import json
import time

from rs.api.transport import GameTransport, ProtocolError, SocketGameTransport
from rs.helper.logger import log_to_run


class Client:

    def __init__(self, transport: GameTransport | None = None):
        self._transport = SocketGameTransport.from_environment() if transport is None else transport
        self._connected = False
        self.connect()

    def connect(self) -> None:
        if self._connected:
            return
        self.send_message("ready")
        self._connected = True

    def send_message(self, message: str, silent: bool = False, before_run: bool = False) -> str:
        t0 = time.perf_counter()
        response = self._transport.send(message, silent=silent, before_run=before_run)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._validate_protocol_response(response)
        if elapsed_ms > 100:
            log_to_run(f"[TIMING] Client.send_message '{message[:40]}' took {elapsed_ms:.0f}ms")
        return response

    def _validate_protocol_response(self, response: str) -> None:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"CommunicationMod returned malformed JSON: {error.msg}") from error

        if isinstance(payload, dict) and payload.get("error"):
            raise ProtocolError(f"CommunicationMod returned error response: {payload['error']}")
