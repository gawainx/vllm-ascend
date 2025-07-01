from collections import defaultdict
from typing import Callable

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.request import Request

from vllm_ascend.core.miniblock.kv_cache_utils import hash_mini_tokens


class FullAttentionManagerWithMini(FullAttentionManager):

    def __init__(self,
                 kv_cache_spec: KVCacheSpec,
                 block_pool: BlockPool,
                 kv_cache_group_id: int,
                 caching_hash_fn: Callable,
                 mini_block_size: int = -1
                 ):
        super().__init__(kv_cache_spec, block_pool, kv_cache_group_id, caching_hash_fn)
        self._mini_block_size = -1
        self.head_mini_to_blocks: dict[BlockHash, list[KVCacheBlock]] = defaultdict(list)


    @classmethod
    def init_with_full_attn_manager(cls, manager: FullAttentionManager, mini_block_size: int):
        obj = cls(kv_cache_spec=manager.kv_cache_spec, block_pool=manager.block_pool,
                  kv_cache_group_id=manager.kv_cache_group_id, caching_hash_fn=manager.caching_hash_fn,
                  mini_block_size=mini_block_size)
        return obj

    @property
    def enable_mini_block(self):
        return 0 < self._mini_block_size < self.block_size

    def find_extra_hits(self, mini_hashes):
        if not self.enable_mini_block:
            return 0, -1
        max_hit_counts = 0
        src_blk_id = -1

        for block in self.head_mini_to_blocks[mini_hashes[0]]:
            match_blocks = block.mini_match(mini_hashes)
            if max_hit_counts < match_blocks:
                max_hit_counts = match_blocks
                src_blk_id = block.block_id
        return max_hit_counts * self._mini_block_size, src_blk_id

    def cache_blocks(self, request: Request, block_hashes: list[BlockHash],
                     num_tokens: int) -> None:
        super().cache_blocks(request, block_hashes, num_tokens)
        if self.enable_mini_block:
            req_blocks = self.req_to_blocks[request.request_id]
            parent_block_hash = None
            for blk in req_blocks:
                if len(blk.mini_block_hashes) == 0:
                    full_block_tokens = blk.block_hash.block_hash.token_ids
                    blk.mini_block_hashes = hash_mini_tokens(self.caching_hash_fn, full_block_tokens,
                                                             self._mini_block_size, parent_block_hash)
                    self.head_mini_to_blocks[blk.mini_block_hashes[0]].append(blk)
                parent_block_hash = blk.block_hash

