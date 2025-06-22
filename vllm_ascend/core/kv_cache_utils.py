from vllm.v1.core.kv_cache_utils import KVCacheBlock, BlockHashWithGroupId, hash_block_tokens


class KVCacheBlockWithMini(KVCacheBlock):
    mini_block_hashes: list[BlockHashWithGroupId] = None

    def evict_miniblock(self):
        self.mini_block_hashes.clear()

    @property
    def head_mini_block(self):
        if self.mini_block_hashes:
            return self.mini_block_hashes[0]
        return None

    def mini_match(self, hash_list: list[BlockHashWithGroupId]) -> int:
        if self.mini_block_hashes and hash_list:
            matches = sum(a.hash_value == b.hash_value for a, b in zip(self.mini_block_hashes, hash_list))
            return matches
        else:
            return 0

def hash_mini_tokens(token_ids, mini_block_size: int, parent_block_hash_value=None) -> List[BlockHashType]:
    """Performing mini match"""
    ret = []
    for start in range(0, len(token_ids), mini_block_size):
        end = start + mini_block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < mini_block_size:
            break
        block_hash = hash_block_tokens(parent_block_hash_value,
                                       block_token_ids, extra_keys=None)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret

