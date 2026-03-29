import io
import unittest
from unittest.mock import patch

from rs.api.client import Client
from rs.api.transport import ProtocolError, StdioGameTransport, TransportError


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

    def test_client_raises_on_malformed_json_response(self):
        transport = FakeTransport(["not-json"])

        with self.assertRaises(ProtocolError):
            Client(transport=transport)

    def test_client_connect_is_idempotent(self):
        transport = FakeTransport(['{"status": "ok"}'])
        client = Client(transport=transport)

        client.connect()

        self.assertEqual(1, len(transport.calls))


if __name__ == "__main__":
    unittest.main()
