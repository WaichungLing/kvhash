import torch
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import repeat_kv
from hqq.core.quantize import BaseQuantizeConfig, quantize_tensor as hqq_quantize_tensor

class PCQuantCache(Cache):
    def __init__(
        self,
        config,
        cache_budget,
        sink_protect_tokens,
        recent_protect_budget,
        top_rank,
        quant_bit,
        device: str | None
    ) -> None:
        super().__init__()
        self.config = config
        self.cache_budget = cache_budget
        self.sink_protect_tokens = sink_protect_tokens
        self.recent_protect_budget = recent_protect_budget
        self._seen_tokens = 0
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # pcquant
        self.top_rank = top_rank
        self.quant_bit = quant_bit

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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO
        assert cache_kwargs is not None, "cache_kwargs must be provided for KVHashCache"
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.key_cache[layer_idx] is None or self.value_cache[layer_idx] is None:
            # prefill
            self.quantize(key_states, value_states, layer_idx)
            return key_states, value_states
        else:
            # auto-regressive
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        assert self.key_cache[layer_idx].shape[2] == self.value_cache[layer_idx].shape[2], "Mismatch in the sequence length of K and V Cache"
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_indices(self, cache: torch.Tensor):
        bsz, num_head, slen, hidden = cache.shape
        key_cache_reshaped = cache.view(bsz * num_head, slen, hidden)

        key_cache_centered = key_cache_reshaped - key_cache_reshaped.mean(dim=1, keepdim=True)
        U, S, _ = torch.linalg.svd(key_cache_centered, full_matrices=False)

        U_reduced = U[:, :, :self.top_rank]  
        S_reduced = S[:, :self.top_rank]  
        projections = U_reduced * S_reduced.unsqueeze(1)
        row_contributions = torch.sum(projections**2, dim=2).view(bsz, num_head, slen)

        lower_quantile = torch.quantile(row_contributions, 0.25, dim=-1, keepdim=True)
        upper_quantile = torch.quantile(row_contributions, 0.75, dim=-1, keepdim=True)

        below_indices = row_contributions < lower_quantile 
        above_indices = row_contributions > upper_quantile
        middle_indices = (row_contributions >= lower_quantile) & (row_contributions <= upper_quantile)
        return below_indices, middle_indices, above_indices     # bool matrices
    
    def quant_indices(self, cache: torch.Tensor, indices: torch.Tensor, target_bits: int):
        one = cache[indices]       # (# of True, hidden)
        quant_config = BaseQuantizeConfig(
            nbits=target_bits,  
            group_size=64,      
            quant_scale=True,  
            quant_zero=True,   
            axis=0     
        )

        quantized_below, quant_meta = hqq_quantize_tensor(one, quant_config)

        quantized_storage = {
            "quantized_data": quantized_below,  
            "quant_meta": quant_meta,           
            "indices": indices,           
            "target_bits": target_bits,     
        }

        return quantized_storage

    def quantize(self, key_states, value_states, layer_idx):
        k_below_indices, k_middle_indices, k_above_indices = self.get_indices(key_states)
        v_below_indices, v_middle_indices, v_above_indices = self.get_indices(value_states)

    def clear(self):
        # TODO
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.hash_values: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.attn_sum: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.div_planes = torch.randn((self.num_planes, self.config.head_dim), dtype=torch.bfloat16, device=self.div_planes.device)
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None