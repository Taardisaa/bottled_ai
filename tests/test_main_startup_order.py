import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_assert_langmem_ready_or_fail_raises_when_enabled_but_not_ready(self):
        service = SimpleNamespace(
            is_ready=lambda: False,
            status=lambda: "embeddings_unavailable:simulated_embedding_init_failure",
        )
        with patch("main.load_llm_config", return_value=SimpleNamespace(langmem_enabled=True)):
            with self.assertRaises(RuntimeError):
                main._assert_langmem_ready_or_fail(service)

    def test_assert_langmem_ready_or_fail_skips_when_langmem_disabled(self):
        service = SimpleNamespace(
            is_ready=lambda: False,
            status=lambda: "embeddings_unavailable:simulated_embedding_init_failure",
        )
        with patch("main.load_llm_config", return_value=SimpleNamespace(langmem_enabled=False)):
            main._assert_langmem_ready_or_fail(service)


if __name__ == "__main__":
    unittest.main()
