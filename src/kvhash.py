import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache

class KVHashCache(Cache):
    def __init__(self, 
                 config, 
                 cache_budget,
                 sink_protect_tokens,
                 recent_protect_budget,
                 num_planes: int = 4) -> None:
        super().__init__()
        self.config = config
        self.cache_budget = cache_budget
        self.sink_protect_tokens = sink_protect_tokens
        self.recent_protect_budget = recent_protect_budget

        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers

        self.hash_values: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.attn_sum: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.div_planes = torch.randn((num_planes, self.config.head_dim))
        self.powers_of_two = 2 ** torch.arange(num_planes - 1, -1, -1, dtype=torch.int32)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)
    
    def update_attn_sum(self, layer_idx, attn_scores):  # Shape: (b, num_head, q_len, k_len)
        # if layer_idx == 0:
        #     print(f"layer idx {layer_idx}=======attn shape: {attn_scores.shape}")
        summation = torch.sum(attn_scores, dim=2)  # Shape: (b, num_head, k_len)
        n_per_group = self.config.num_attention_heads // self.config.num_key_value_heads
        temp = summation.view(summation.shape[0], self.config.num_key_value_heads, n_per_group, -1)
        summation = temp.sum(dim=2)
        # if layer_idx == 0:
        #     print(f"summation.shape = {summation.shape}")

        if self.attn_sum[layer_idx] is None:
            self.attn_sum[layer_idx] = summation
        else:
            self.attn_sum[layer_idx] += summation[:, :, :self.attn_sum[layer_idx].shape[-1]]
            q_len = summation.shape[-1] - self.attn_sum[layer_idx].shape[-1]
            new_in = summation[:, :, -q_len:]
            self.attn_sum[layer_idx] = torch.cat([self.attn_sum[layer_idx], new_in], dim=2)
        # if layer_idx == 0 and self.attn_sum[layer_idx] is not None:
        #     print(f"attn_sum shape: {self.attn_sum[layer_idx].shape}")

    def update_hash_values(self, layer_idx, key_states):   # Shape: (b, num_head, q_len, k_len)
        hash_bits = torch.matmul(key_states, self.div_planes.transpose(-1, -2))
        hash_bits = (hash_bits >= 0).int()
        hash_vals = torch.matmul(hash_bits, self.powers_of_two)  # Shape: (b, num_head, s_len)
        # if layer_idx == 0:
        #     print(f"======= hash_vals shape: {hash_vals.shape}")

        if self.hash_values[layer_idx] is None:
            # Initialize hash_values if it is None
            self.hash_values[layer_idx] = hash_vals
        else:
            q_len = key_states.shape[2]
            self.hash_values[layer_idx] = torch.cat([self.hash_values[layer_idx], hash_vals[:, :, -q_len:]], dim=2)
        # if layer_idx == 0 and self.hash_values[layer_idx] is not None:
        #     print(f"self.hash_values shape: {self.hash_values[layer_idx].shape}")


    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.key_cache[layer_idx] is None or self.value_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        assert self.key_cache[layer_idx].shape[2] == self.value_cache[layer_idx].shape[2], (
            f"Mismatch in the sequence length of K and V Cache"
        )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def evict(
        self,
        layer_idx: int
    ):
        assert self.hash_values[layer_idx].shape == self.attn_sum[layer_idx].shape, (
            f"Dimension of hash_values and attn_sum does not match"
        )
        q_len = self.key_cache[layer_idx].shape[2]
        recent_protect_tokens = int(self.recent_protect_budget * q_len)
        eviction_protect_tokens = int(self.cache_budget * q_len) - recent_protect_tokens - self.sink_protect_tokens
        if eviction_protect_tokens <= 0:    # q_len is too small
            return  
        evict_tokens = q_len - int(self.cache_budget * q_len)
        assert evict_tokens > 0, (
            f"the number of tokens need to be evicted should be larger than 0"
        )

        if self.layer_idx == 0:
            print(f'need to evict = {evict_tokens}')

        if q_len > 1:
            evict_hash = self.hash_values[layer_idx][:,:,self.sink_protect_tokens:-recent_protect_tokens]    # Shape (b, num_heads, qlen)
            evict_attn = self.attn_sum[layer_idx][:,:,self.sink_protect_tokens:-recent_protect_tokens]       # Shape (b, num_heads, qlen)

            keep_ids = []
            if torch.cuda.is_available():
                streams = [torch.cuda.Stream() for _ in range(evict_hash.shape[1])]
                for i, stream in enumerate(streams):
                    with torch.cuda.stream(stream):
                        keep_ids.append(self.head_eviction(evict_hash, evict_attn, i, evict_tokens))
                torch.cuda.synchronize() 
            else:
                for i in range(evict_hash.shape[1]):
                    keep_ids.append(self.head_eviction(evict_hash, evict_attn, i, evict_tokens))
            return
        else:
            return
        
    def head_eviction(self, hash, attn, head_idx, evict_num):
        evict_id_per_head = []
        unique_values, counts = torch.unique(hash[:, head_idx], return_counts=True)
        probabilities = counts.float() / counts.sum()

        return evict_id_per_head

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None