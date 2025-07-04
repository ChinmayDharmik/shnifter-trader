import unittest
from shnifter_frontend.llm_manager_popout import LLMManagerPopout
from PySide6.QtWidgets import QApplication
import sys

class TestLLMManagerPopout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.popout = LLMManagerPopout()

    def test_load_models(self):
        self.popout.load_models()
        self.assertTrue(self.popout.analyzer_dropdown.count() > 0)
        self.assertTrue(self.popout.verifier_dropdown.count() > 0)

    def test_toggle_dual_llm(self):
        from PySide6.QtCore import Qt
        self.popout.on_toggle_dual_llm(Qt.Checked)
        self.assertIn('green', self.popout.analyzer_status.text())
        self.popout.on_toggle_dual_llm(Qt.Unchecked)
        self.assertIn('red', self.popout.analyzer_status.text())

    def test_set_status_indicators(self):
        self.popout.set_status_indicators(True)
        self.assertIn('green', self.popout.analyzer_status.text())
        self.assertIn('green', self.popout.verifier_status.text())
        self.popout.set_status_indicators(False)
        self.assertIn('red', self.popout.analyzer_status.text())
        self.assertIn('red', self.popout.verifier_status.text())

if __name__ == '__main__':
    unittest.main()
