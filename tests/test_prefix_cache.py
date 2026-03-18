from aster.cache.prefix_cache import PrefixCache
from aster.core.config import CacheSettings
from aster.telemetry.metrics import MetricsRegistry


def test_prefix_cache_hit():
    cache = PrefixCache(CacheSettings(), MetricsRegistry("test"))
    cache.store("m", [1, 2, 3], [4, 5], 128)
    hit = cache.lookup("m", [1, 2, 3])
    assert hit is not None
