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


class TestNormalise:
    """Bedrock/LiteLLM-style ids resolve to the first-party table keys."""

    @pytest.mark.parametrize('model_id, key', [
        ('aws/claude-5-sonnet', 'claude-sonnet-5'),
        ('claude-5-sonnet', 'claude-sonnet-5'),
        ('claude-4-5-haiku', 'claude-haiku-4-5'),
        ('claude-5-5-opus', 'claude-opus-5-5'),
        ('claude-4-8-opus', 'claude-opus-4-8'),
        ('anthropic.claude-sonnet-5', 'claude-sonnet-5'),
        ('claude-sonnet-5-20260101', 'claude-sonnet-5'),
        ('bedrock/anthropic.claude-4-5-haiku-20260101', 'claude-haiku-4-5'),
        ('anthropic/claude-opus-5-5', 'claude-opus-5-5'),
        ('claude-sonnet-5', 'claude-sonnet-5'),
    ])
    def test_aliases_resolve(self, model_id: str, key: str) -> None:
        assert pricing.normalise_model_id(model_id) == key
        assert pricing.prices_for(model_id) == pricing.prices_for(key)
        assert pricing.prices_for(model_id) is not None

    def test_unknown_ids_stay_none(self) -> None:
        for m in ('gpt-4.1', 'aws/claude-9-sonnet', 'claude-3-turbo',
                  'aws/llama-3-70b'):
            assert pricing.prices_for(m) is None, m

    def test_normalise_leaves_unknown_shapes_alone(self) -> None:
        assert pricing.normalise_model_id('gpt-4.1') == 'gpt-4.1'
        assert pricing.normalise_model_id('aws/llama-3-70b') == 'llama-3-70b'

    def test_cost_through_alias(self) -> None:
        usage = pricing.Usage(input_tokens=1_000_000)
        assert pricing.cost_usd('aws/claude-5-sonnet', usage) == \
            pytest.approx(2.0)
