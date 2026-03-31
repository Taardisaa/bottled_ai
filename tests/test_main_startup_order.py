import unittest
from types import SimpleNamespace

import main


class TestMainStartupOrder(unittest.TestCase):
    def test_langmem_service_is_initialized_after_client(self):
        order: list[str] = []

        def fake_client_factory():
            order.append("client")
            return object()

        def fake_langmem_factory():
            order.append("langmem")
            return SimpleNamespace(status=lambda: "ready")

        main.initialize_client_and_langmem(
            client_factory=fake_client_factory,
            langmem_factory=fake_langmem_factory,
        )

        self.assertEqual(["client", "langmem"], order)


if __name__ == "__main__":
    unittest.main()
