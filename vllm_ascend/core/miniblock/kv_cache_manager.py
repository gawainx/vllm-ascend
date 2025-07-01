import dataclasses
from collections import defaultdict
from typing import Optional

from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm_ascend.core.miniblock.kv_cache_utils import hash_mini_tokens
from .kv_cache_coordinator import UnitaryKVCacheCoordinatorWithMini


@dataclasses.dataclass
class BlockCopyMetadata:
    src_blk_id: int = -1
    dst_blk_id: int = -1
    num_copied_tokens: int = 0

    @property
    def required_copy(self):
        return self.src_blk_id >= 0 and self.dst_blk_id >= 0 and self.num_copied_tokens != 0

    @property
    def required_allocate(self):
        return self.src_blk_id >= 0 and self.num_copied_tokens != 0


class KVCacheManagerWithMini(KVCacheManager):

    def __init__(self,
                 kv_cache_config: KVCacheConfig,
                 max_model_len: int,
                 enable_caching: bool = True,
                 caching_hash_algo: str = "builtin",
                 use_eagle: bool = False,
                 log_stats: bool = False,
                 enable_kv_cache_events: bool = False, mini_block_size: int = -1):
        super().__init__(kv_cache_config, max_model_len, enable_caching,
                         caching_hash_algo, use_eagle, log_stats,
                         enable_kv_cache_events)
        self._mini_block_size = mini_block_size
        self.req_to_copy_meta = defaultdict(lambda: BlockCopyMetadata())

    def reinit_coordinator(self):
        if isinstance(self.coordinator, UnitaryKVCacheCoordinator):
            self.coordinator = UnitaryKVCacheCoordinatorWithMini.init_from_coordinator(self.coordinator,
                                                                                       self._mini_block_size)

    def reset_req_meta(self, req: Request):
        self.req_to_copy_meta[req.request_id] = BlockCopyMetadata()

    def obtain_req_meta(self, req: Request):
        return self.req_to_copy_meta[req.request_id]

    @property
    def enable_mini_block(self):
        return 0 < self._mini_block_size < self.block_size

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        kv_blocks, num_computed_tokens = super().get_computed_blocks(request)
        rest_tokens = request.all_token_ids[num_computed_tokens:]
        if self.enable_mini_block and len(rest_tokens) > self._mini_block_size:
            extra_tokens = rest_tokens[:self.block_size]
            parent_block_hash = kv_blocks.blocks[0][-1].block_hash
            extra_hashes = hash_mini_tokens(self.caching_hash_fn, extra_tokens, self._mini_block_size,
                                            parent_block_hash)
            num_extra_matches, src_id = self.coordinator.find_extra_hits(extra_hashes)
            if num_extra_matches != 0:
                metadata = BlockCopyMetadata()
                metadata.src_blk_id = src_id
                metadata.num_copied_tokens = num_extra_matches
                self.req_to_copy_meta[request.request_id] = metadata

            num_computed_tokens = num_computed_tokens + num_extra_matches
            if self.log_stats:
                self.prefix_cache_stats.hits += num_extra_matches

        return kv_blocks, num_computed_tokens

    def allocate_slots(
            self,
            request: Request,
            num_new_tokens: int,
            num_new_computed_tokens: int = 0,
            new_computed_blocks: Optional[KVCacheBlocks] = None,
            num_draft_tokens: int = 0,
            num_lookahead_tokens: int = 0,
            delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        kv_blocks = super().allocate_slots(request, num_new_tokens, num_new_computed_tokens, new_computed_blocks,
                                           num_draft_tokens, num_lookahead_tokens, delay_cache_blocks)
        if self.req_to_copy_meta[request.request_id].required_allocate and kv_blocks is not None:
            self.req_to_copy_meta[request.request_id].dst_blk_id = kv_blocks.blocks[0][0].block_id
