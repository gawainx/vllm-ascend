from dataclasses import dataclass
from typing import Optional
import vllm.v1.core.kv_cache_utils
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, BlockHash


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    _block_hash: Optional[BlockHashWithGroupId] = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    mini_block_hashes: list[BlockHash] = None

    def evict_miniblock(self):
        self.mini_block_hashes.clear()

    @property
    def head_mini_block(self):
        if self.mini_block_hashes:
            return self.mini_block_hashes[0]
        return None

    def mini_match(self, hash_list: list[BlockHash]) -> int:
        if self.mini_block_hashes and hash_list:
            matches = sum(a.hash_value == b.hash_value for a, b in zip(self.mini_block_hashes, hash_list))
            return matches
        else:
            return 0

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[BlockHashWithGroupId]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen.")
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self.mini_block_hashes.clear()
        self._block_hash = None

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = (self.prev_free_block.block_id
                         if self.prev_free_block else None)
        next_block_id = (self.next_free_block.block_id
                         if self.next_free_block else None)
        return (f"KVCacheBlock(block_id={self.block_id}, "
                f"ref_cnt={self.ref_cnt}, "
                f"_block_hash={self._block_hash}, "
                f"prev_free_block={prev_block_id}, "
                f"next_free_block={next_block_id})")


vllm.v1.core.kv_cache_utils.KVCacheBlock = KVCacheBlock
