import unittest
from core.shnifter_providers.shnifter_registry import ProviderRegistry
from core.shnifter_providers.abstract.shnifter_provider import ShnifterProvider

class DummyProvider(ShnifterProvider):
    def __init__(self):
        super().__init__(name="dummy")
    def fetch(self, *args, **kwargs):
        return 'dummy-data'

class TestProviderRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ProviderRegistry()
        self.dummy = DummyProvider()

    def test_register_and_get(self):
        self.registry.register('dummy', self.dummy)
        provider = self.registry.get('dummy')
        self.assertIs(provider, self.dummy)

    def test_list_providers(self):
        self.registry.register('dummy', self.dummy)
        providers = self.registry.list_providers()
        self.assertIn('dummy', providers)

    def test_unregister(self):
        self.registry.register('dummy', self.dummy)
        self.registry.unregister('dummy')
        self.assertIsNone(self.registry.get('dummy'))

if __name__ == '__main__':
    unittest.main()
