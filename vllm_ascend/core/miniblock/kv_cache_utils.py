from typing import List

from vllm.v1.core.kv_cache_utils import KVCacheBlock, BlockHashWithGroupId, hash_block_tokens, BlockHash


def hash_mini_tokens(hash_fn, token_ids, mini_block_size: int, parent_block_hash=None) -> List[BlockHash]:
    """Performing mini match"""
    ret = []
    parent_block_hash_value = parent_block_hash.get_hash_value() if parent_block_hash is not None else None
    for start in range(0, len(token_ids), mini_block_size):
        end = start + mini_block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < mini_block_size:
            break
        block_hash = hash_block_tokens(hash_fn, parent_block_hash_value,
                                       block_token_ids, extra_keys=None)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret

