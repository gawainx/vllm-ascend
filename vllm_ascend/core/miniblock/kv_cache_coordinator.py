from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from .single_type_kv_manager import FullAttentionManagerWithMini


class UnitaryKVCacheCoordinatorWithMini(UnitaryKVCacheCoordinator):

    def find_extra_hits(self, extra_block_hashes: list[BlockHash]):
        return self.single_type_managers[0].find_extra_hits(extra_block_hashes)


    @classmethod
    def init_from_coordinator(cls, coordinator: UnitaryKVCacheCoordinator, mini_block_size):
        if len(coordinator.single_type_managers) == 1 and (
                isinstance(coordinator.single_type_managers[0], FullAttentionManager)
        ):
            manager = coordinator.single_type_managers[0]
            obj = cls(coordinator.kv_cache_config, coordinator.max_model_len, coordinator.use_eagle,
                      enable_caching=coordinator.block_pool.enable_caching,
                      caching_hash_fn=manager.caching_hash_fn,
                      enable_kv_cache_events=coordinator.block_pool.enable_kv_cache_events)
            obj.single_type_managers = (FullAttentionManagerWithMini.init_with_full_attn_manager(manager,
                                                                                                 mini_block_size))
            return obj
        else:
            return coordinator
