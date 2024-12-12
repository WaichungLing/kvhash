import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache


class KVHashCache(Cache):
    def __init__(
        self,
        config,
        cache_budget,
        sink_protect_tokens,
        recent_protect_budget,
        num_planes: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.cache_budget = cache_budget
        self.sink_protect_tokens = sink_protect_tokens
        self.recent_protect_budget = recent_protect_budget
        self.num_planes = num_planes
        self._seen_tokens = 0

        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers

        self.hash_values: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.attn_sum: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.register_buffer("div_planes", torch.randn((self.num_planes, self.config.head_dim), dtype=torch.float32))
        self.register_buffer("powers_of_two", 2 ** torch.arange(self.num_planes - 1, -1, -1, dtype=torch.float32))

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
        summation = torch.sum(attn_scores, dim=2)  # Shape: (b, num_head, k_len)
        # GQA
        # if layer_idx == 0:
        #     print(f'==== attn score shape {attn_scores.shape}')
        #     print(f'==== attn score {attn_scores[:,0:4,:,:]}')
        n_per_group = self.config.num_attention_heads // self.config.num_key_value_heads
        # if layer_idx == 0:
        #     print(f'======== summation {summation.shape} ')
        #     print(f'======== summation {summation[:,0:4,:]}')
        #     print(f'========n_per_group {n_per_group}')
        if n_per_group > 1:
            temp = summation.view(summation.shape[0], self.config.num_key_value_heads, n_per_group, -1)
            summation = temp.sum(dim=2)
        # if layer_idx == 0:
        #     print(f"===== summation.shape = {summation.shape}")
        #     print(f'===== summation after {summation[:,0,:]}')

        if self.attn_sum[layer_idx] is None:
            self.attn_sum[layer_idx] = summation
        else:
            self.attn_sum[layer_idx] += summation[:, :, : self.attn_sum[layer_idx].shape[-1]]
            q_len = summation.shape[-1] - self.attn_sum[layer_idx].shape[-1]
            new_in = summation[:, :, -q_len:]
            self.attn_sum[layer_idx] = torch.cat([self.attn_sum[layer_idx], new_in], dim=2)
        # if layer_idx == 0 and self.attn_sum[layer_idx] is not None:
        #     print(f"attn_sum shape: {self.attn_sum[layer_idx].shape}")

    def update_hash_values(self, layer_idx, key_states):  # Shape: (b, num_head, q_len, k_len)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        hash_bits = torch.matmul(key_states, self.div_planes.transpose(-1, -2))
        hash_bits = (hash_bits >= 0).to(torch.float32)
        hash_vals = torch.matmul(hash_bits, self.powers_of_two)  # Shape: (b, num_head, s_len)
        # if layer_idx == 0:
        #     print(f"======= hash_bits shape {hash_bits.shape} === {hash_bits[:,0,:,:]}")
        #     print(f"======= hash_vals shape {hash_vals.shape} === {hash_vals[:,0,:]}")

        if self.hash_values[layer_idx] is None:
            # Initialize hash_values if it is None
            self.hash_values[layer_idx] = hash_vals
        else:
            q_len = key_states.shape[2]
            self.hash_values[layer_idx] = torch.cat([self.hash_values[layer_idx], hash_vals[:, :, -q_len:]], dim=2)
            # if layer_idx == 0:
            #     print(f'====== self.hash_values shape {self.hash_values[layer_idx].shape} === {self.hash_values[layer_idx]}')

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

        assert self.key_cache[layer_idx].shape[2] == self.value_cache[layer_idx].shape[2], f"Mismatch in the sequence length of K and V Cache"
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def evict(self, layer_idx: int):
        assert (
            self.hash_values[layer_idx].shape == self.attn_sum[layer_idx].shape
        ), f"Dimension of hash_values {self.hash_values[layer_idx].shape} and attn_sum {self.attn_sum[layer_idx].shape} does not match"
        q_len = self.key_cache[layer_idx].shape[2]
        recent_protect_tokens = int(self.recent_protect_budget * q_len)
        if recent_protect_tokens == 0:
            recent_protect_tokens = 1
        eviction_zone = q_len - recent_protect_tokens - self.sink_protect_tokens
        evict_tokens = q_len - int(self.cache_budget * self._seen_tokens)
        if evict_tokens > eviction_zone:
            return
        assert evict_tokens > 0, f"the number of tokens need to be evicted should be larger than 0"

        # if layer_idx == 0:
        #     print(f'q_len = {q_len}, recent_protect = {recent_protect_tokens}, eviction_zone = {eviction_zone}, evict_tokens = {evict_tokens}')

        if q_len > 1:
            evict_hash = self.hash_values[layer_idx][:, :, self.sink_protect_tokens : -recent_protect_tokens]  # Shape (b, num_heads, qlen)
            evict_attn = self.attn_sum[layer_idx][:, :, self.sink_protect_tokens : -recent_protect_tokens]  # Shape (b, num_heads, qlen)

            evict_ids = []
            for i in range(self.config.num_key_value_heads):
                evict_id_per_head = self.head_eviction(evict_hash, evict_attn, i, evict_tokens)
                evict_id_per_head += self.sink_protect_tokens
                evict_ids.append(evict_id_per_head)

            min_evict_tokens = min(len(evict_id_per_head) for evict_id_per_head in evict_ids)
            aligned_evict_ids = torch.stack([evict_id_per_head[:min_evict_tokens] for evict_id_per_head in evict_ids])

            keep_indices = torch.ones((self.config.num_key_value_heads, q_len), dtype=torch.bool, device=evict_hash.device)
            keep_indices.scatter_(1, aligned_evict_ids, False)
            valid_indices = torch.masked_select(torch.arange(q_len, device=evict_hash.device).unsqueeze(0).expand(self.config.num_key_value_heads, -1), keep_indices).view(
                self.config.num_key_value_heads, -1
            )

            expanded_indices = (
                valid_indices.unsqueeze(0)
                .unsqueeze(-1)
                .expand(
                    self.key_cache[layer_idx].shape[0],  # batch_size
                    self.key_cache[layer_idx].shape[1],  # num_heads
                    -1,  # new_q_len
                    self.key_cache[layer_idx].shape[-1],  # k_len
                )
            )

            self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(dim=2, index=expanded_indices)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(dim=2, index=expanded_indices)

            expanded_indices_hash_attn = valid_indices.unsqueeze(0).expand(
                self.hash_values[layer_idx].shape[0],  # batch_size
                self.hash_values[layer_idx].shape[1],  # num_heads
                -1,  # new_q_len
            )
            self.hash_values[layer_idx] = self.hash_values[layer_idx].gather(dim=2, index=expanded_indices_hash_attn)
            self.attn_sum[layer_idx] = self.attn_sum[layer_idx].gather(dim=2, index=expanded_indices_hash_attn)

            assert self.key_cache[layer_idx].shape[2] == valid_indices.shape[1], "Mismatch in q_len after eviction!"
            # if layer_idx == 0:
            #     print(f"After Eviction: key_cache shape {self.key_cache[layer_idx].shape}, value_cache shape {self.value_cache[layer_idx].shape}")
            return
            #     streams = [torch.cuda.Stream() for _ in range(evict_hash.shape[1])]
            #     for i, stream in enumerate(streams):
            #         with torch.cuda.stream(stream):
            #             keep_ids.append(self.head_eviction(evict_hash, evict_attn, i, evict_tokens))
            #     torch.cuda.synchronize()
            # else:
        # if torch.cuda.is_available():
        else:
            return

    def head_eviction(self, hash, attn, head_idx, evict_num):
        unique_values, inverse_indices, counts = torch.unique(hash[:, head_idx, :], return_inverse=True, return_counts=True)
        frequencies = counts.float() / counts.sum()  # Frequency of each unique hash value
        evict_num_per_hash = (frequencies * evict_num).floor().long()  # Dynamic K values (one per unique hash)

        num_unique = unique_values.size(0)
        inverse_indices_expanded = inverse_indices.view(1, -1)  # Expand for broadcasting
        mask = inverse_indices_expanded == torch.arange(num_unique, device=hash.device).view(-1, 1)

        attn_sums_per_unique = torch.where(mask, attn[:, head_idx, :].expand(num_unique, -1), torch.tensor(float("inf"), device=hash.device))

        evict_id_per_head = torch.empty(evict_num_per_hash.sum().item(), dtype=torch.long, device=hash.device)
        start_idx = 0
        for i in range(num_unique):
            k = evict_num_per_hash[i]  # Dynamic K for this unique value
            if k > 0:
                # Use `largest=False` to select smallest K values
                _, topk_indices = torch.topk(attn_sums_per_unique[i], k=k, largest=False)
                evict_id_per_head[start_idx : start_idx + k] = topk_indices
                start_idx += k

        return evict_id_per_head

    def is_eviction_needed(self, layer_idx):
        # if layer_idx == 0:
        #     print(f"===== {self.key_cache[layer_idx].shape[2]} -- {self._seen_tokens * self.cache_budget}/{self._seen_tokens}=====")
        return self.key_cache[layer_idx].shape[2] > self._seen_tokens * self.cache_budget

    def clear(self):
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.hash_values: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.attn_sum: List[torch.Tensor] = [None] * self.config.num_hidden_layers

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def evict_h2o(self, layer_idx: int):
        pass

