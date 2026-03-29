from __future__ import annotations

import json

from rs.api.transport import GameTransport, ProtocolError, SocketGameTransport


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
        response = self._transport.send(message, silent=silent, before_run=before_run)
        self._validate_protocol_response(response)
        return response

    def _validate_protocol_response(self, response: str) -> None:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"CommunicationMod returned malformed JSON: {error.msg}") from error

        if isinstance(payload, dict) and payload.get("error"):
            raise ProtocolError(f"CommunicationMod returned error response: {payload['error']}")
