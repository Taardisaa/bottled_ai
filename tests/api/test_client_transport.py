import io
import os
import socket
import threading
import unittest
from unittest.mock import patch

from rs.api.client import Client
from rs.api.transport import (
    COMMUNICATIONMOD_CONNECT_TIMEOUT_ENV,
    COMMUNICATIONMOD_HOST_ENV,
    COMMUNICATIONMOD_PORT_ENV,
    COMMUNICATIONMOD_TRANSPORT_ENV,
    ProtocolError,
    SocketGameTransport,
    StdioGameTransport,
    TransportError,
)


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def send(self, message, *, silent=False, before_run=False):
        self.calls.append({
            "message": message,
            "silent": silent,
            "before_run": before_run,
        })
        if not self.responses:
            raise AssertionError("No fake response available")
        return self.responses.pop(0)


class SocketHarness:
    def __init__(self, responses):
        self._responses = list(responses)
        self.received = []
        self._ready = threading.Event()
        self._done = threading.Event()
        self.port = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self._ready.wait(timeout=2)

    def _run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            self.port = server.getsockname()[1]
            self._ready.set()
            connection, _ = server.accept()
            with connection:
                reader = connection.makefile("r", encoding="utf-8", newline="\n")
                writer = connection.makefile("w", encoding="utf-8", newline="\n")
                try:
                    while True:
                        message = reader.readline()
                        if message == "":
                            break
                        self.received.append(message.rstrip("\r\n"))
                        if not self._responses:
                            break
                        writer.write(self._responses.pop(0) + "\n")
                        writer.flush()
                finally:
                    reader.close()
                    writer.close()
                    self._done.set()

    def join(self):
        self.thread.join(timeout=2)
        self._done.wait(timeout=2)


class TestStdioGameTransport(unittest.TestCase):
    def test_send_writes_message_and_returns_response(self):
        reader = io.StringIO('{"ok": true}\n')
        writer = io.StringIO()
        transport = StdioGameTransport(reader=reader, writer=writer)

        response = transport.send("ready")

        self.assertEqual('{"ok": true}', response)
        self.assertEqual("ready\n", writer.getvalue())

    def test_send_logs_request_and_response(self):
        reader = io.StringIO('{"ok": true}\n')
        writer = io.StringIO()
        transport = StdioGameTransport(reader=reader, writer=writer)

        with patch("rs.api.transport.log") as log_mock, patch("rs.api.transport.log_to_run") as log_to_run_mock:
            transport.send("start IRONCLAD", before_run=True)

        log_mock.assert_any_call("Sending message: start IRONCLAD")
        log_mock.assert_any_call('Response: {"ok": true}')
        log_to_run_mock.assert_not_called()

    def test_send_respects_silent_flag(self):
        reader = io.StringIO('{"ok": true}\n')
        writer = io.StringIO()
        transport = StdioGameTransport(reader=reader, writer=writer)

        with patch("rs.api.transport.log") as log_mock, patch("rs.api.transport.log_to_run") as log_to_run_mock:
            response = transport.send("noop", silent=True)

        self.assertEqual('{"ok": true}', response)
        log_mock.assert_not_called()
        log_to_run_mock.assert_not_called()

    def test_send_raises_on_eof(self):
        transport = StdioGameTransport(reader=io.StringIO(""), writer=io.StringIO())

        with self.assertRaises(TransportError):
            transport.send("ready")

    def test_send_raises_on_empty_response(self):
        transport = StdioGameTransport(reader=io.StringIO("\n"), writer=io.StringIO())

        with self.assertRaises(ProtocolError):
            transport.send("ready")


class TestSocketGameTransport(unittest.TestCase):
    def test_from_environment_connects_and_exchanges_messages(self):
        harness = SocketHarness(['{"status": "ok"}'])
        self.addCleanup(harness.join)

        with patch.dict(os.environ, {
            COMMUNICATIONMOD_TRANSPORT_ENV: "socket",
            COMMUNICATIONMOD_HOST_ENV: "127.0.0.1",
            COMMUNICATIONMOD_PORT_ENV: str(harness.port),
            COMMUNICATIONMOD_CONNECT_TIMEOUT_ENV: "2",
        }, clear=False):
            transport = SocketGameTransport.from_environment()
            self.addCleanup(transport.close)

            response = transport.send("ready")

        self.assertEqual('{"status": "ok"}', response)
        self.assertEqual(["ready"], harness.received)

    def test_from_environment_requires_socket_host(self):
        with patch.dict(os.environ, {
            COMMUNICATIONMOD_TRANSPORT_ENV: "socket",
        }, clear=True):
            with self.assertRaises(TransportError):
                SocketGameTransport.from_environment()

    def test_from_environment_raises_on_connection_failure(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        _, port = probe.getsockname()
        probe.close()

        with patch.dict(os.environ, {
            COMMUNICATIONMOD_TRANSPORT_ENV: "socket",
            COMMUNICATIONMOD_HOST_ENV: "127.0.0.1",
            COMMUNICATIONMOD_PORT_ENV: str(port),
            COMMUNICATIONMOD_CONNECT_TIMEOUT_ENV: "0.2",
        }, clear=False):
            with self.assertRaises(TransportError):
                SocketGameTransport.from_environment()


class TestClient(unittest.TestCase):
    def test_client_connect_sends_ready_on_init(self):
        transport = FakeTransport(['{"status": "ok"}'])

        client = Client(transport=transport)

        self.assertTrue(client._connected)
        self.assertEqual(1, len(transport.calls))
        self.assertEqual("ready", transport.calls[0]["message"])

    def test_client_send_message_delegates_to_transport(self):
        transport = FakeTransport([
            '{"status": "ok"}',
            '{"game_state": {"screen_type": "EVENT"}}',
        ])
        client = Client(transport=transport)

        response = client.send_message("choose 0", silent=True, before_run=True)

        self.assertEqual('{"game_state": {"screen_type": "EVENT"}}', response)
        self.assertEqual("choose 0", transport.calls[1]["message"])
        self.assertTrue(transport.calls[1]["silent"])
        self.assertTrue(transport.calls[1]["before_run"])

    def test_client_uses_socket_transport_by_default(self):
        harness = SocketHarness([
            '{"status": "ok"}',
            '{"game_state": {"screen_type": "EVENT"}}',
        ])
        self.addCleanup(harness.join)

        with patch.dict(os.environ, {
            COMMUNICATIONMOD_TRANSPORT_ENV: "socket",
            COMMUNICATIONMOD_HOST_ENV: "127.0.0.1",
            COMMUNICATIONMOD_PORT_ENV: str(harness.port),
            COMMUNICATIONMOD_CONNECT_TIMEOUT_ENV: "2",
        }, clear=False):
            client = Client()
            self.addCleanup(client._transport.close)
            response = client.send_message("choose 0")

        self.assertEqual('{"game_state": {"screen_type": "EVENT"}}', response)
        self.assertEqual(["ready", "choose 0"], harness.received)

    def test_client_raises_on_malformed_json_response(self):
        transport = FakeTransport(["not-json"])

        with self.assertRaises(ProtocolError):
            Client(transport=transport)

    def test_client_raises_on_communication_mod_error_payload(self):
        transport = FakeTransport(['{"error": "Invalid command: give", "ready_for_command": true}'])

        with self.assertRaises(ProtocolError):
            Client(transport=transport)

    def test_client_connect_is_idempotent(self):
        transport = FakeTransport(['{"status": "ok"}'])
        client = Client(transport=transport)

        client.connect()

        self.assertEqual(1, len(transport.calls))


if __name__ == "__main__":
    unittest.main()
