import torch
import math
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
from torch import nn


class KVHashCache(Cache):
    def __init__(
        self, config, cache_budget, sink_protect_tokens, recent_protect_budget
    ) -> None:
        super().__init__()
        self.config = config
        self.cache_budget = cache_budget
        self.sink_protect_tokens = sink_protect_tokens
        self.recent_protect_budget = recent_protect_budget

        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.attn_sparsity: List[torch.Tensor] = [
            None
        ] * self.config.num_hidden_layers  # NOTE: temporary
        self.attn_sparsity_tail: List[torch.Tensor] = [
            None
        ] * self.config.num_hidden_layers  # NOTE: temporary
        self.attn_sparsity_pca_qk: List[torch.Tensor] = [
            None
        ] * self.config.num_hidden_layers  # NOTE: temporary
        self.attn_sparsity_pca_qq: List[torch.Tensor] = [
            None
        ] * self.config.num_hidden_layers  # NOTE: temporary
        # self.attn_sparstiy_hash:  List[torch.Tensor] = [None] * self.config.num_hidden_layers   # NOTE: temporary
        # self.hash_values: List[torch.Tensor] = [None] * self.config.num_hidden_layers           # NOTE: temporary
        # self.register_buffer("div_planes", torch.randn((8, self.config.head_dim), dtype=torch.float32))    # NOTE: temporary
        # self.register_buffer("powers_of_two", 2 ** torch.arange(7, -1, -1, dtype=torch.float32))

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

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

    def update_hash_values(
        self, layer_idx, query_states
    ):  # Shape: (b, num_head, q_len, k_len)
        hash_bits = torch.matmul(query_states, self.div_planes.transpose(-1, -2))
        hash_bits = (hash_bits >= 0).to(torch.float32)
        hash_vals = torch.matmul(
            hash_bits, self.powers_of_two
        )  # Shape: (b, num_head, s_len)

        if self.hash_values[layer_idx] is None:
            self.hash_values[layer_idx] = hash_vals

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.key_cache[layer_idx] is None or self.value_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        assert (
            self.key_cache[layer_idx].shape[2] == self.value_cache[layer_idx].shape[2]
        ), f"Mismatch in the sequence length of K and V Cache"
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def pca_select(
        self, query: torch.Tensor, key: torch.Tensor, r: int, total: int, latest: int
    ):  # select proxy from query
        s, h = query.shape
        pca_center = key - key.mean(dim=0)
        if pca_center.dtype != torch.float32:
            pca_center = pca_center.to(torch.float32)
        U, S, Vh = torch.linalg.svd(pca_center, full_matrices=False)  # Vh: (h, h)
        w = min(r, h)
        top_PCs = S[:w, None] * Vh[:w, :]  # (r, h)
        projection = torch.matmul(query, top_PCs.T)  # (s, r)
        importance_scores = projection.pow(2).sum(dim=1)  # (s,)
        _, top_k_importance_indices = torch.topk(
            importance_scores, k=total, largest=True
        )
        last_16_indices = torch.arange(s - latest, s, device=key.device)
        top_k_importance_indices = top_k_importance_indices[
            ~torch.isin(top_k_importance_indices, last_16_indices)
        ]
        remaining_top_indices = top_k_importance_indices[: total - latest]
        combined_indices = torch.cat([remaining_top_indices, last_16_indices])
        return combined_indices

    def update_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attn_scores: torch.Tensor,
        layer_idx: int,
    ):
        if self.attn_sparsity[layer_idx] is not None:  # only prefill
            return

        sparsities = []
        for i in range(self.config.num_attention_heads):
            attn_per_head = attn_scores[0, i]
            mask = torch.tril(torch.ones_like(attn_per_head, dtype=torch.bool))
            lower_triangle_values = attn_per_head[mask]
            min_val = torch.min(lower_triangle_values)
            max_val = torch.max(lower_triangle_values)
            threshold = min_val + 0.0005 * (max_val - min_val)
            sparsity = (
                torch.sum(torch.abs(lower_triangle_values) < threshold).item()
                / lower_triangle_values.numel()
            )
            sparsities.append(sparsity)
        self.attn_sparsity[layer_idx] = sparsities

        # sparsities_tail = []
        # # query_proxy = query_states[:,:,-64:,:]
        # initial_32 = query_states[:, :, :32, :]
        # last_32 = query_states[:, :, -32:, :]
        # query_proxy = torch.cat([initial_32, last_32], dim=2)
        # attn_weights = torch.matmul(query_proxy, key_states.transpose(2, 3)) / math.sqrt(self.config.head_dim)
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_sum = torch.sum(attn_weights, dim=-2)
        # for i in range(self.config.num_attention_heads):
        #     attn_sum_per_head = attn_sum[0, i]
        #     min_val = torch.min(attn_sum_per_head)
        #     max_val = torch.max(attn_sum_per_head)
        #     threshold = min_val + 0.001 * (max_val - min_val)
        #     sparsity_tail = torch.sum(attn_sum_per_head < threshold).item() / query_states.shape[2]
        #     sparsities_tail.append(sparsity_tail)
        # self.attn_sparsity_tail[layer_idx] = sparsities_tail

        sparsities_pca_qk = []
        for i in range(self.config.num_attention_heads):
            one_q = query_states[0, i]
            one_k = key_states[0, i]
            indices = self.pca_select(one_q, one_k, 4, 64, 32)
            query_proxy = one_q[indices, :]
            attn = torch.matmul(query_proxy, one_k.transpose(0, 1)) / math.sqrt(
                self.config.head_dim
            )
            attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_sum = torch.sum(attn, dim=0)
            min_val = torch.min(attn_sum)
            max_val = torch.max(attn_sum)
            threshold = min_val + 0.001 * (max_val - min_val)
            sparsity_pca_qk = (
                torch.sum(attn_sum < threshold).item() / query_states.shape[2]
            )
            sparsities_pca_qk.append(sparsity_pca_qk)
        self.attn_sparsity_pca_qk[layer_idx] = sparsities_pca_qk

        sparsities_pca_qq = []
        for i in range(self.config.num_attention_heads):
            one_q = query_states[0, i]
            one_k = key_states[0, i]
            indices = self.pca_select(one_q, one_q, 4, 64, 32)
            query_proxy = one_q[indices, :]
            attn = torch.matmul(query_proxy, one_k.transpose(0, 1)) / math.sqrt(
                self.config.head_dim
            )
            attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_sum = torch.sum(attn, dim=0)
            min_val = torch.min(attn_sum)
            max_val = torch.max(attn_sum)
            threshold = min_val + 0.001 * (max_val - min_val)
            sparsity_pca_qq = (
                torch.sum(attn_sum < threshold).item() / query_states.shape[2]
            )
            sparsities_pca_qq.append(sparsity_pca_qq)
        self.attn_sparsity_pca_qq[layer_idx] = sparsities_pca_qq

        # sparsities_hash = []
        # for i in range(self.config.num_attention_heads):
        #     last_16 = query_states[0, i, -16:, :]               # Shape: (16, query_dim)
        #     q_hash = self.hash_values[layer_idx][0,i,:-16]
        #     unique_values, counts = torch.unique(q_hash, return_counts=True)
        #     frequency_distribution = dict(zip(unique_values.tolist(), counts.tolist()))
        #     # if i == 0:
        #     #     print(frequency_distribution)
        #     probabilities = torch.tensor([frequency_distribution[val.item()] for val in q_hash], dtype=torch.float32)
        #     probabilities /= probabilities.sum()
        #     indices = torch.multinomial(probabilities, 48, replacement=False)
        #     sampled_queries = query_states[0, i, indices, :]  # Shape: (48, query_dim)
        #     query_proxy = torch.cat((last_16, sampled_queries), dim=0) # Shape (64, h_dim)
        #     # if i == 0:
        #     #     print(indices)

        #     one_k = key_states[0,i] # Shape (4096, h_dim)
        #     attn = torch.matmul(query_proxy, one_k.transpose(0, 1)) / math.sqrt(self.config.head_dim)
        #     attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
        #     attn_sum = torch.sum(attn, dim=0)
        #     min_val = torch.min(attn_sum)
        #     max_val = torch.max(attn_sum)
        #     threshold = min_val + 0.001 * (max_val - min_val)
        #     sparsity_hash = torch.sum(attn_sum < threshold).item() / query_states.shape[2]
        #     sparsities_hash.append(sparsity_hash)
        # self.attn_sparstiy_hash[layer_idx] = sparsities_hash

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
        assert (
            evict_tokens > 0
        ), f"the number of tokens need to be evicted should be larger than 0"

        # if layer_idx == 0:
        #     print(f'q_len = {q_len}, recent_protect = {recent_protect_tokens}, eviction_zone = {eviction_zone}, evict_tokens = {evict_tokens}')

        if q_len > 1:
            evict_hash = self.hash_values[layer_idx][
                :, :, self.sink_protect_tokens : -recent_protect_tokens
            ]  # Shape (b, num_heads, qlen)
            evict_attn = self.attn_sum[layer_idx][
                :, :, self.sink_protect_tokens : -recent_protect_tokens
            ]  # Shape (b, num_heads, qlen)

            evict_ids = []
            for i in range(self.config.num_key_value_heads):
                evict_id_per_head = self.head_eviction(
                    evict_hash, evict_attn, i, evict_tokens
                )
                evict_id_per_head += self.sink_protect_tokens
                evict_ids.append(evict_id_per_head)

            min_evict_tokens = min(
                len(evict_id_per_head) for evict_id_per_head in evict_ids
            )
            aligned_evict_ids = torch.stack(
                [
                    evict_id_per_head[:min_evict_tokens]
                    for evict_id_per_head in evict_ids
                ]
            )

            keep_indices = torch.ones(
                (self.config.num_key_value_heads, q_len),
                dtype=torch.bool,
                device=evict_hash.device,
            )
            keep_indices.scatter_(1, aligned_evict_ids, False)
            valid_indices = torch.masked_select(
                torch.arange(q_len, device=evict_hash.device)
                .unsqueeze(0)
                .expand(self.config.num_key_value_heads, -1),
                keep_indices,
            ).view(self.config.num_key_value_heads, -1)

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

            self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(
                dim=2, index=expanded_indices
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(
                dim=2, index=expanded_indices
            )

            expanded_indices_hash_attn = valid_indices.unsqueeze(0).expand(
                self.hash_values[layer_idx].shape[0],  # batch_size
                self.hash_values[layer_idx].shape[1],  # num_heads
                -1,  # new_q_len
            )
            self.hash_values[layer_idx] = self.hash_values[layer_idx].gather(
                dim=2, index=expanded_indices_hash_attn
            )
            self.attn_sum[layer_idx] = self.attn_sum[layer_idx].gather(
                dim=2, index=expanded_indices_hash_attn
            )

            assert (
                self.key_cache[layer_idx].shape[2] == valid_indices.shape[1]
            ), "Mismatch in q_len after eviction!"
        else:
            return

    def head_eviction(self, hash, attn, head_idx, evict_num):
        unique_values, inverse_indices, counts = torch.unique(
            hash[:, head_idx, :], return_inverse=True, return_counts=True
        )
        frequencies = (
            counts.float() / counts.sum()
        )  # Frequency of each unique hash value
        evict_num_per_hash = (
            (frequencies * evict_num).floor().long()
        )  # Dynamic K values (one per unique hash)

        num_unique = unique_values.size(0)
        inverse_indices_expanded = inverse_indices.view(
            1, -1
        )  # Expand for broadcasting
        mask = inverse_indices_expanded == torch.arange(
            num_unique, device=hash.device
        ).view(-1, 1)

        attn_sums_per_unique = torch.where(
            mask,
            attn[:, head_idx, :].expand(num_unique, -1),
            torch.tensor(float("inf"), device=hash.device),
        )

        evict_id_per_head = torch.empty(
            evict_num_per_hash.sum().item(), dtype=torch.long, device=hash.device
        )
        start_idx = 0
        for i in range(num_unique):
            k = evict_num_per_hash[i]  # Dynamic K for this unique value
            if k > 0:
                # Use `largest=False` to select smallest K values
                _, topk_indices = torch.topk(
                    attn_sums_per_unique[i], k=k, largest=False
                )
                evict_id_per_head[start_idx : start_idx + k] = topk_indices
                start_idx += k

        return evict_id_per_head

    def is_eviction_needed(self, layer_idx):
        # if layer_idx == 0:
        #     print(f"===== {self.key_cache[layer_idx].shape[2]} -- {self._seen_tokens * self.cache_budget}/{self._seen_tokens}=====")
        return (
            self.key_cache[layer_idx].shape[2] > self._seen_tokens * self.cache_budget
        )

    def clear(self):
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.attn_sparsity: List[torch.Tensor] = [
            None
        ] * self.config.num_hidden_layers  # NOTE: temporary
        self.attn_sparsity_tail: List[torch.Tensor] = [
            None
        ] * self.config.num_hidden_layers  # NOTE: temporary
        self.attn_sparsity_pca_qk: List[torch.Tensor] = [
            None
        ] * self.config.num_hidden_layers  # NOTE: temporary
        self.attn_sparsity_pca_qq: List[torch.Tensor] = [
            None
        ] * self.config.num_hidden_layers  # NOTE: temporary
        # self.attn_sparstiy_hash: List[torch.Tensor] = [None] * self.config.num_hidden_layers # NOTE: temporary
        # self.hash_values: List[torch.Tensor] = [None] * self.config.num_hidden_layers   # NOTE: temporary
        # self.div_planes = torch.randn((8, self.config.head_dim), dtype=torch.float32, device=self.div_planes.device)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None
