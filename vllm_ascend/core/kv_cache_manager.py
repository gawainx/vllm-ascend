from typing import Optional
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request
from vllm_ascend.core.kv_cache_utils import hash_mini_tokens


class KVCacheManagerWithMini(KVCacheManager):

    def __init__(
            self,
            block_size: int,
            num_gpu_blocks: int,
            max_model_len: int,
            sliding_window: Optional[int] = None,
            enable_caching: bool = True,
            num_preallocate_tokens: int = 64,
            log_stats: bool = False,
            mini_block_size: int = -1
    ) -> None:
        super().__init__(block_size, num_gpu_blocks, max_model_len,
                         sliding_window, enable_caching, num_preallocate_tokens,
                         log_stats)
        self.mini_block_size = mini_block_size

    @property
    def enable_mini_block(self):
        return 0 < self.mini_block_size < self.block_size

    def _maybe_evict_miniblock(self, block: KVCacheBlock) -> bool:
        if block.mini_block_hashes and self.full_block_hash_to_mini[block.block_hash]:
            head_mini = block.head_mini_block
            del self.full_block_hash_to_mini[block.block_hash]
            del self.head_mini_to_full[head_mini]
            return True
        return False

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        block_hash = block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_block:
            if self._maybe_evict_miniblock(block):
                block.evict_miniblock()
            block.reset_hash()
            del self.cached_block_hash_to_block[block_hash][block.block_id]

            if len(self.cached_block_hash_to_block[block_hash]) == 0:
                del self.cached_block_hash_to_block[block_hash]

            return True
        return False

    def get_computed_blocks(self, request: Request):
        computed_blocks, num_computed_tokens_full = super().get_computed_blocks(request)
        num_mini_tokens_matched = 0
        tokens_for_mini_match = request.all_token_ids[
                                num_computed_tokens_full:num_computed_tokens_full + self.block_size]

        if self.enable_mini_block and tokens_for_mini_match > self.mini_block_size:
            if computed_blocks:
                parent_block_hash = computed_blocks[-1].block_hash.hash_value
            else:
                parent_block_hash = None
            mini_hash_to_match = hash_mini_tokens(tokens_for_mini_match, self.mini_block_size, parent_block_hash)
            max_mini_match = 0
            src_block = None
            for blk_ in self.head_mini_to_full[mini_hash_to_match[0]]:
                match_count = blk_.mini_match(mini_hash_to_match)
                if match_count > max_mini_match:
                    max_mini_match = match_count
                    src_block = blk_.block_id
            if src_block is not None:
                request.src_block = src_block
                num_mini_tokens_matched += max_mini_match * self.mini_block_size

        num_computed_tokens = num_computed_tokens_full + num_mini_tokens_matched
        return computed_blocks, num_computed_tokens
