"""Tests for guru.domain.pricing."""
import pytest

from guru import config
from guru.domain import pricing


class TestLookup:
    """Price lookup: exact id, longest-contained key, overrides."""

    def test_exact_id(self) -> None:
        p = pricing.prices_for('claude-sonnet-5')
        assert p == {'input_per_m': 2.0, 'cache_write_5m_per_m': 2.5,
                     'cache_write_1h_per_m': 4.0, 'cache_read_per_m': 0.2,
                     'output_per_m': 10.0}

    def test_longest_contained_key_wins(self) -> None:
        opus55 = pricing.prices_for('anthropic/claude-opus-5-5')
        assert opus55 is not None and opus55['input_per_m'] == 4.0
        opus5 = pricing.prices_for('azure/claude-opus-5')
        assert opus5 is not None and opus5['input_per_m'] == 5.0

    def test_recent_claude_models_are_priced(self) -> None:
        for m, inp, out in (('claude-opus-4-7', 5.0, 25.0),
                            ('claude-opus-4-6', 5.0, 25.0),
                            ('claude-sonnet-4-6', 3.0, 15.0)):
            p = pricing.prices_for(m)
            assert p is not None, m
            assert (p['input_per_m'], p['output_per_m']) == (inp, out)

    def test_unknown_is_none(self) -> None:
        assert pricing.prices_for('gpt-4.1') is None

    def test_override_merges_over_table(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'PRICING_OVERRIDES',
                            {'claude-sonnet-5': {'input_per_m': 2.5}})
        p = pricing.prices_for('claude-sonnet-5')
        assert p is not None
        assert p['input_per_m'] == 2.5 and p['output_per_m'] == 10.0

    def test_new_model_override_needs_input_and_output(
            self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'PRICING_OVERRIDES',
                            {'gpt-4.1': {'input_per_m': 2.0}})
        assert pricing.prices_for('gpt-4.1') is None
        monkeypatch.setattr(config, 'PRICING_OVERRIDES',
                            {'gpt-4.1': {'input_per_m': 2.0,
                                         'output_per_m': 8.0}})
        assert pricing.prices_for('azure/gpt-4.1') == {'input_per_m': 2.0,
                                                       'output_per_m': 8.0}


class TestCost:
    """Per-call cost in USD."""

    def test_exact_cents(self) -> None:
        usage = pricing.Usage(input_tokens=1_000_000, output_tokens=100_000,
                              cache_read_tokens=500_000,
                              cache_write_tokens=200_000)
        # 2.0 + 1.0 + 0.5*0.2 + 0.2*2.5 = 3.6
        assert pricing.cost_usd('claude-sonnet-5', usage) == pytest.approx(3.6)

    def test_local_is_zero_and_unknown_is_none(self) -> None:
        usage = pricing.Usage(input_tokens=10, output_tokens=10)
        assert pricing.cost_usd('qwen3:14b', usage, local=True) == 0.0
        assert pricing.cost_usd('gpt-4.1', usage) is None
