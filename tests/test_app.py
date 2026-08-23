import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:
    AppTest = None


@unittest.skipIf(AppTest is None, "Streamlit is not installed in this environment")
class StreamlitAppTests(unittest.TestCase):
    def test_app_starts_without_errors(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(app_path).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertIn("CreditWise", app.title[0].value)


if __name__ == "__main__":
    unittest.main()
