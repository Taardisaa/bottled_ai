import tempfile
import unittest
from pathlib import Path

from presentation_config import _load_presentation_values


class TestPresentationConfig(unittest.TestCase):
    def test_load_presentation_values_missing_file(self):
        values = _load_presentation_values("configs/does_not_exist.yaml")
        self.assertEqual({}, values)

    def test_load_presentation_values_from_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "presentation_config.yaml"
            config_path.write_text(
                "presentation_mode: true\n"
                "p_delay: wait 10\n"
                "p_delay_s: wait 5\n"
                "slow_events: true\n"
                "slow_pathing: false\n",
                encoding="utf-8",
            )

            values = _load_presentation_values(str(config_path))

            self.assertEqual(True, values["presentation_mode"])
            self.assertEqual("wait 10", values["p_delay"])
            self.assertEqual("wait 5", values["p_delay_s"])
            self.assertEqual(True, values["slow_events"])
            self.assertEqual(False, values["slow_pathing"])


if __name__ == "__main__":
    unittest.main()
