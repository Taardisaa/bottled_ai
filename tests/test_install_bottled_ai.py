import tempfile
import unittest
from pathlib import Path

import install_bottled_ai as installer


class TestInstallBottledAi(unittest.TestCase):
    def test_build_python_command(self):
        main_py = Path("/repo/main.py")
        python_path = Path("/repo/venv/bin/python")
        command = installer._build_command(
            main_py=main_py,
            python_path=python_path,
            mode="python",
            debugpy_listen="5678",
            wait_for_client=False,
        )
        self.assertEqual(f"{python_path.as_posix()} {main_py.as_posix()}", command)

    def test_build_debugpy_command(self):
        command = installer._build_command(
            main_py=Path("/repo/main.py"),
            python_path=Path("/repo/venv/bin/python"),
            mode="debugpy",
            debugpy_listen="127.0.0.1:5678",
            wait_for_client=True,
        )
        self.assertIn("-m debugpy", command)
        self.assertIn("--listen 127.0.0.1:5678", command)
        self.assertIn("--wait-for-client", command)

    def test_write_command_replaces_existing_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.properties"
            config_path.write_text("foo=1\ncommand=old\nbar=2\n", encoding="utf-8")

            installer._write_command_to_properties(config_path, "py main.py")
            updated = config_path.read_text(encoding="utf-8")

            self.assertIn("foo=1", updated)
            self.assertIn("bar=2", updated)
            self.assertIn("command=py main.py", updated)
            self.assertNotIn("command=old", updated)

    def test_write_command_appends_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.properties"
            config_path.write_text("foo=1\n", encoding="utf-8")

            installer._write_command_to_properties(config_path, "py main.py")
            updated = config_path.read_text(encoding="utf-8")

            self.assertTrue(updated.endswith("command=py main.py\n"))


if __name__ == "__main__":
    unittest.main()
