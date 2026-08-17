"""Tests for the Ollama GPU auto-fit context sizing."""
from types import SimpleNamespace

from guru import config
from guru.adapters.ollama import OllamaAdapter


class TestGpuAutoFit:
    """GPU auto-fit default: KV math, budget clamp, and the default policy."""

    def _adapter(self) -> OllamaAdapter:
        return OllamaAdapter()

    def test_kv_bytes_per_token(self, monkeypatch) -> None:
        info = SimpleNamespace(modelinfo={
            'general.architecture': 'qwen3',
            'qwen3.block_count': 40,
            'qwen3.attention.head_count': 40,
            'qwen3.attention.head_count_kv': 8,
            'qwen3.embedding_length': 5120,
        })
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        # head_dim = 5120/40 = 128; 2*40*8*128*2.0 = 163840
        assert self._adapter()._kv_bytes_per_token('m') == 163840

    def test_kv_bytes_missing_metadata_is_zero(self, monkeypatch) -> None:
        info = SimpleNamespace(modelinfo={'general.architecture': 'qwen3'})
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        assert self._adapter()._kv_bytes_per_token('m') == 0

    def test_max_gpu_ctx_math(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 24 * 1024 ** 3)
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 163840)
        monkeypatch.setattr(a, '_weight_bytes', lambda m: 9 * 1024 ** 3)
        assert a._max_gpu_ctx('m', 262144) == 63488      # not capped
        assert a._max_gpu_ctx('m', 40960) == 40960       # capped at ceiling

    def test_max_gpu_ctx_zero_when_no_budget(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 163840)
        monkeypatch.setattr(a, '_weight_bytes', lambda m: 0)
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 0)
        assert a._max_gpu_ctx('m', 40960) == 0

    def test_max_gpu_ctx_zero_when_weights_exceed(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 20 * 1024 ** 3)
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 163840)
        monkeypatch.setattr(a, '_weight_bytes', lambda m: 19 * 1024 ** 3)
        # budget 16G, weights 19G -> avail < 0 -> 0 (not the 2k floor)
        assert a._max_gpu_ctx('m', 40960) == 0

    def test_calibrated_ctx_from_measurement(self, monkeypatch) -> None:
        a = self._adapter()
        # weights 15e9, kv 163840/token; high probe spills to an 18e9 budget.
        measured = {
            2048: (15335544320, 15335544320),     # fits (vram == size)
            32768: (20368709120, 18000000000),    # spills -> budget 18e9
        }
        monkeypatch.setattr(a, '_measure_at', lambda ctx: measured[int(ctx)])
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 163840)
        # (18e9*0.95 - 15e9)/163840 -> 12288 after rounding to 1024
        assert a._calibrated_ctx('m', 262144) == 12288

    # sizes chosen so the probe deltas give kv = 100000 bytes/token exactly:
    #   (8_072_000_000 - 5_000_000_000) / (32768 - 2048) = 100000
    _FIT = {2048: (5_000_000_000, 5_000_000_000),
            32768: (8_072_000_000, 8_072_000_000)}      # both 100% GPU

    def test_calibrated_ctx_both_fit_ceiling_probe_fits(self, monkeypatch):
        a = self._adapter()
        measured = dict(self._FIT)
        measured[131072] = (13_000_000_000, 13_000_000_000)   # ceiling fits
        monkeypatch.setattr(a, '_measure_at', lambda ctx: measured[int(ctx)])
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 100000)
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 10 ** 12)  # huge
        # ceiling probe fits -> use the full ceiling, not the 32k probe cap
        assert a._calibrated_ctx('m', 131072) == 131072

    def test_calibrated_ctx_extend_probe_spills(self, monkeypatch):
        a = self._adapter()
        measured = dict(self._FIT)
        measured[131072] = (30_000_000_000, 18_000_000_000)   # spills
        monkeypatch.setattr(a, '_measure_at', lambda ctx: measured[int(ctx)])
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 100000)
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 10 ** 12)
        # weights = 5e9 - 2048*100000 = 4_795_200_000
        # (18e9*0.95 - weights)/100000 = 123048 -> round down to 122880
        assert a._calibrated_ctx('m', 131072) == 122880

    def test_calibrated_ctx_no_budget_keeps_probe(self, monkeypatch):
        a = self._adapter()
        measured = dict(self._FIT)
        monkeypatch.setattr(a, '_measure_at', lambda ctx: measured[int(ctx)])
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 100000)
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 0)   # unknown
        assert a._calibrated_ctx('m', 262144) == 32768

    def test_calibrated_ctx_zero_when_measure_fails(self, monkeypatch):
        a = self._adapter()
        monkeypatch.setattr(a, '_measure_at', lambda ctx: (0, 0))
        assert a._calibrated_ctx('m', 262144) == 0

    def test_default_ctx_prefers_stored(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(config, 'load_model_ctx', lambda: {'m': 16384})
        assert a._default_ctx('m', 4096, 40960) == 16384
        assert a._default_ctx('m', 4096, 8192) == 8192   # capped at ceiling

    def test_default_ctx_autofit_when_no_stored(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(config, 'load_model_ctx', lambda: {})
        monkeypatch.setattr(a, '_max_gpu_ctx', lambda m, c: 32768)
        assert a._default_ctx('m', 4096, 40960) == 32768

    def test_default_ctx_falls_back_to_resolved(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(config, 'load_model_ctx', lambda: {})
        monkeypatch.setattr(a, '_max_gpu_ctx', lambda m, c: 0)
        assert a._default_ctx('m', 4096, 40960) == 4096

    def test_list_models_reports_ceiling(self, monkeypatch) -> None:
        models = SimpleNamespace(models=[
            SimpleNamespace(model='qwen3:14b', size=9_000_000_000)])
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.list', lambda: models)
        a = self._adapter()
        monkeypatch.setattr(
            a, '_resolve_context_window', lambda m: (4096, 40960))
        monkeypatch.setattr(a, '_param_size', lambda m: '14B')
        infos = a.list_models()
        assert infos[0].context_window == 40960

    def test_list_models_sorted_by_basename(self, monkeypatch) -> None:
        names = ['batiai/qwen3.6-27b:q3', 'devstral-small-2:24b',
                 'hf.co/unsloth/Qwen3-14B-128K-GGUF:Q4_K_M',
                 'huihui_ai/qwen3-abliterated:8b', 'gpt-oss-20b-32k:latest']
        models = SimpleNamespace(models=[
            SimpleNamespace(model=n, size=1) for n in names])
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.list', lambda: models)
        a = self._adapter()
        monkeypatch.setattr(
            a, '_resolve_context_window', lambda m: (4096, 4096))
        monkeypatch.setattr(a, '_param_size', lambda m: '?')
        order = [mi.model_id for mi in a.list_models()]
        assert order == [
            'devstral-small-2:24b',
            'gpt-oss-20b-32k:latest',
            'hf.co/unsloth/Qwen3-14B-128K-GGUF:Q4_K_M',   # qwen3-14b-128k
            'huihui_ai/qwen3-abliterated:8b',             # qwen3-abliterated
            'batiai/qwen3.6-27b:q3',                      # qwen3.6
        ]
