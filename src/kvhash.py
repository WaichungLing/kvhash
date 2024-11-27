import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache

class KVHashCache(Cache):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.hash_values: List[torch.Tensor] = [None] * num_hidden_layers
        self.attn_sum: List[torch.Tensor] = [None] * num_hidden_layers

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

        if self.attn_sum[layer_idx] is None:
            self.attn_sum[layer_idx] = summation
        else:
            self.attn_sum[layer_idx] += summation[:, :, :self.attn_sum[layer_idx].shape[-1]]
            q_len = summation.shape[-1] - self.attn_sum[layer_idx].shape[-1]
            new_in = summation[:, :, -q_len:]
            self.attn_sum[layer_idx] = torch.cat([self.attn_sum[layer_idx], new_in], dim=2)
        # if layer_idx == 0 and self.attn_sum[layer_idx] is not None:
        #     print(f"attn_sum shape: {self.attn_sum[layer_idx].shape}")

    def update_hash_values(self, layer_idx, q_len, key_states):
        hash_bits = torch.matmul(key_states, self.hash_planes.transpose(-1, -2))  # Shape: (b, slen, num_head, num_plane)
        hash_bits = (hash_bits >= 0).int()  # Convert to 1s and 0s
        hash_vals = torch.matmul(hash_bits, self.powers_of_two)  # Shape: (b, slen, num_head)
        hash_vals = hash_vals.permute(0, 2, 1)  # Transpose to (b, num_head, s_len)

        if self.hash_values is None:
            # Initialize hash_values if it is None
            self.hash_values = hash_vals
        else:
            self.hash_values = torch.cat([self.hash_values, hash_vals[:, :, -q_len:]], dim=2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None
