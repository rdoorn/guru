"""Adapter registry (guru.repositories.adapters) and Adapter.remote."""
import pytest

from guru.adapters.base import Adapter
from guru.adapters.ollama import OllamaAdapter
from guru.repositories.adapters import AdapterRegistry


class FakeAdapter(Adapter):
    """Minimal concrete Adapter for registry tests."""

    def __init__(self, name: str, remote: bool = True,
                 enabled: bool = True) -> None:
        self.name = name
        self.remote = remote
        self.enabled = enabled

    def available(self) -> bool:
        return True

    def list_models(self) -> list:
        return []

    def activate(self, model_id: str) -> None:
        pass

    def run_turn(self) -> None:
        pass

    def summarise(self, transcript: str) -> str:
        return ''


class TestRemoteFlag:
    def test_base_default_is_remote(self) -> None:
        assert Adapter.remote is True
        assert FakeAdapter('x').remote is True

    def test_ollama_is_local(self) -> None:
        assert OllamaAdapter.remote is False
        assert OllamaAdapter().remote is False


class TestAdapterRegistry:
    def test_register_get_names(self) -> None:
        reg = AdapterRegistry()
        a, b = FakeAdapter('Ollama', remote=False), FakeAdapter('Anthropic')
        reg.register(a)
        reg.register(b)
        assert reg.get('Ollama') is a
        assert reg.get('Anthropic') is b
        assert reg.names() == ['Ollama', 'Anthropic']

    def test_get_unknown_is_none(self) -> None:
        assert AdapterRegistry().get('nope') is None

    def test_is_remote(self) -> None:
        reg = AdapterRegistry()
        reg.register(FakeAdapter('Ollama', remote=False))
        reg.register(FakeAdapter('Anthropic', remote=True))
        assert reg.is_remote('Ollama') is False
        assert reg.is_remote('Anthropic') is True

    def test_is_remote_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            AdapterRegistry().is_remote('nope')

    def test_iter_and_len(self) -> None:
        reg = AdapterRegistry()
        adapters = [FakeAdapter('a'), FakeAdapter('b')]
        for a in adapters:
            reg.register(a)
        assert list(reg) == adapters
        assert len(reg) == 2
        assert len(AdapterRegistry()) == 0

    def test_register_duplicate_name_replaces_in_place(self) -> None:
        reg = AdapterRegistry()
        old, new = FakeAdapter('Ollama'), FakeAdapter('Ollama', remote=False)
        reg.register(old)
        reg.register(new)
        assert reg.get('Ollama') is new
        assert reg.names() == ['Ollama']
        assert len(reg) == 1

    def test_from_iterable(self) -> None:
        reg = AdapterRegistry([FakeAdapter('a'), FakeAdapter('b')])
        assert reg.names() == ['a', 'b']


class TestCliWiring:
    def test_build_registry_from_adapters(self) -> None:
        from guru import cli
        adapters = [FakeAdapter('Ollama', remote=False), FakeAdapter('X')]
        reg = cli.build_registry(adapters)
        assert isinstance(reg, AdapterRegistry)
        assert reg.names() == ['Ollama', 'X']

    def test_orchestrator_accepts_registry(self) -> None:
        from guru.orchestrator import Orchestrator
        reg = AdapterRegistry()
        assert Orchestrator(registry=reg).registry is reg
        assert Orchestrator().registry is None


class TestRegistryFrom:
    """``registry_from``: the registry over built adapters, enabled or not."""

    def test_keeps_every_adapter_in_order(self) -> None:
        from guru.repositories.adapters import registry_from
        a = FakeAdapter('A', remote=True)
        b = FakeAdapter('B', remote=False, enabled=False)
        reg = registry_from([a, b])
        assert isinstance(reg, AdapterRegistry)
        assert reg.names() == ['A', 'B']
        assert reg.get('B') is b and reg.is_remote('B') is False

    def test_cli_build_registry_delegates(self) -> None:
        from guru import cli
        a = FakeAdapter('A')
        assert cli.build_registry([a]).get('A') is a
