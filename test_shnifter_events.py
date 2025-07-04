import unittest
from core.events import EventLog, EventBus

class TestEventLog(unittest.TestCase):
    def setUp(self):
        # Clear subscribers and logs before each test
        EventBus._subscribers = {}
        EventBus.min_log_level = "DEBUG"
        EventLog.logs = []

    def test_subscribe_and_publish(self):
        events = []
        def handler(payload):
            events.append(payload)
        EventBus.subscribe('INFO', handler)
        EventBus.publish('INFO', {'foo': 'bar'})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], {'foo': 'bar'})

    def test_emit_and_log_level(self):
        EventBus.set_min_log_level('INFO')
        EventLog.emit('INFO', 'Test info message')
        EventLog.emit('DEBUG', 'Should not appear')
        logs = [log for log in EventLog.logs if log['level'] == 'INFO']
        self.assertTrue(any('Test info message' in log['message'] for log in logs))
        # DEBUG logs are always stored, but only INFO+ are published to subscribers
        logs_debug = [log for log in EventLog.logs if log['level'] == 'DEBUG']
        self.assertTrue(logs_debug)  # DEBUG logs are present in EventLog.logs

    def test_export_logs(self):
        EventLog.emit('INFO', 'Export test')
        EventLog.export_logs_txt('test_log.txt')
        EventLog.export_logs_json('test_log.json')
        EventLog.export_logs_csv('test_log.csv')
        # Check files exist and are not empty
        import os
        for fname in ['test_log.txt', 'test_log.json', 'test_log.csv']:
            self.assertTrue(os.path.exists(fname))
            self.assertTrue(os.path.getsize(fname) > 0)
            os.remove(fname)

if __name__ == '__main__':
    unittest.main()
