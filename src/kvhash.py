import torch
import math
import time
from torch import nn
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import repeat_kv


class KVHashCache(Cache):
    def __init__(
        self,
        config,
        cache_budget,
        recent_protect_budget,
        device: str | None,
        proxy_total: int = 64,
        proxy_latest: int = 16,
        top_rank: int = 4,
        n_recursion: int = 1
    ) -> None:
        super().__init__()
        self.config = config

        self.cache_budget = cache_budget
        self.recent_protect_budget = recent_protect_budget
        self.proxy_total = proxy_total
        self.proxy_latest = proxy_latest
        self.top_rank = top_rank
        self.n_recursion = n_recursion

        self._seen_tokens = 0

        # [(batch, num_key_value_head, qlen, hidden)] * layer
        self.key_cache: List[torch.Tensor] = [
            None] * self.config.num_hidden_layers
        # [(batch, num_key_value_head, qlen, hidden)] * layer
        self.value_cache: List[torch.Tensor] = [
            None] * self.config.num_hidden_layers
        # [(batch, num_attention_head, proxy_total, hidden)] * layer
        self.query_proxy: List[torch.Tensor] = [
            None] * self.config.num_hidden_layers

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

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

    # select proxy from query
    def pca_select(self, query: torch.Tensor, key: torch.Tensor, r: int,  total: int, latest: int):
        s, h = query.shape
        pca_center = key - key.mean(dim=0)
        if pca_center.dtype != torch.float32:
            pca_center = pca_center.to(torch.float32)
        U, S, Vh = torch.linalg.svd(
            pca_center, full_matrices=False)  # Vh: (h, h)
        top_PCs = (S[:r, None] * Vh[:r, :])  # (r, h)
        projection = torch.matmul(query.to(top_PCs.dtype), top_PCs.T)  # (s, r)
        importance_scores = projection.pow(2).sum(dim=1)  # (s,)
        _, top_k_importance_indices = torch.topk(
            importance_scores, k=total, largest=True)
        last_16_indices = torch.arange(s - latest, s, device=key.device)
        top_k_importance_indices = top_k_importance_indices[~torch.isin(
            top_k_importance_indices, last_16_indices)]
        remaining_top_indices = top_k_importance_indices[:total - latest]
        combined_indices = torch.cat([remaining_top_indices, last_16_indices])
        return combined_indices

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

        # prefill only
        if self.key_cache[layer_idx] is None or self.value_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

            l_query_proxy = []
            for i in range(self.config.num_attention_heads):
                one_q = query_states[0, i]
                one_k = key_states[0, int(i//self.config.num_key_value_heads)]
                t1 = time.time_ns()
                indices = self.pca_select(
                    one_q, one_k, self.top_rank, self.proxy_total, self.proxy_latest)
                dt = time.time_ns() - t1
                # print(f"[DEBUG] dt {dt}")
                one_proxy = one_q[indices, :]      # (proxy_total, num_hidden)
                l_query_proxy.append(one_proxy)
            self.query_proxy[layer_idx] = torch.stack(
                l_query_proxy, dim=0).unsqueeze(0)
            # print(
            #     f"[DEBUG update] l = {layer_idx}, proxy_dim = {self.query_proxy[layer_idx].shape}")
        else:
            """
                key_states/value_cache (batch, num_key_value_head, 1, hidden)
                self.key_cache: [[(keep_i, hidden) * num_key_value_heads]]*layer
                self.value_cache: [[(keep_i, hidden) * num_key_value_heads]]*layer
            """
            # print(f"DEBUG: DECODING")
            for h_id in range(self.config.num_key_value_heads):
                old_k = self.key_cache[layer_idx][h_id]   # shape (keep_i, hidden)
                old_v = self.value_cache[layer_idx][h_id] # shape (keep_i, hidden)
                new_k = key_states[0, h_id] # shape (1, hidden)
                new_v = value_states[0, h_id]

                # print(f"DEBUG [DECODING]: l = {layer_idx}, h = {h_id}, pre key {old_k.shape}")
                updated_k = torch.cat([old_k, new_k], dim=0)
                updated_v = torch.cat([old_v, new_v], dim=0)
                self.key_cache[layer_idx][h_id]   = updated_k
                self.value_cache[layer_idx][h_id] = updated_v
                # print(f"DEBUG [DECODING]: l = {layer_idx}, h = {h_id}, after key {self.key_cache[layer_idx][h_id].shape}")
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def evict(self):
        """
            self.key_cache      # [(batch, num_key_value_head, qlen, hidden)] * layer
            self.value_cache    # [(batch, num_key_value_head, qlen, hidden)] * layer
            self.query_proxy    # [(batch, num_attention_head, proxy_total, hidden)] * layer
        """
        key_cache = torch.stack(
            self.key_cache)         # Shape: (num_layers, batch, num_key_value_head, qlen, hidden)
        # Shape: (num_layers, batch, num_key_value_head, qlen, hidden)
        value_cache = torch.stack(self.value_cache)
        # Shape: (num_layers, batch, num_attention_head, proxy_total, hidden)
        query_proxy = torch.stack(self.query_proxy)
        # print(
        #    f"DEBUG, key {key_cache.shape}, value {value_cache.shape}, query_proxy {query_proxy.shape}")
        l, b, gqah, qlen, hidden = key_cache.shape

        repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads
        # Shape: (num_layers, batch, num_attention_head, qlen, hidden)
        key_cache_repeated = key_cache.repeat_interleave(repeat_factor, dim=2)

        # Compute attention scores
        attn_scores = torch.einsum(
            'lbpqh,lbpkh->lbpqk',
            query_proxy,
            key_cache_repeated
        ) / math.sqrt(self.config.head_dim)             # Shape: (num_layers, batch, num_attention_heads, proxy_total, qlen)
        # Shape: (num_layers, batch, num_attention_head, proxy_total, qlen)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # print(f"DEBUG, proxy attn {attn_probs.shape}")
        # Shape: (num_layers, batch, num_attention_head, qlen)
        summation = torch.sum(attn_probs, dim=-2)
        # print(f"DEBUG, summation {summation.shape}")
        # Shape: (num_layers, batch, num_attention_head)
        maximum, _ = torch.max(summation, dim=-1)
        # Shape: (num_layers, batch, num_attention_head)
        minimum, _ = torch.min(summation, dim=-1)
        # print(f"DEBUG, min/max {maximum.shape}")
        threshold = minimum + 0.0005 * (maximum - minimum)
        # Shape: (num_layers, batch, num_attention_head)
        sparsity = torch.sum(
            summation < threshold.unsqueeze(-1), dim=-1) / summation.shape[-1]
        # print(f"DEBUG, sparsity {sparsity}")

        # GQA processing
        grouped_sparsities = sparsity.view(l, b, self.config.num_key_value_heads, repeat_factor).mean(
            dim=-1)    # Shape: (num_layers, batch, num_key_value_head)
        # print(f"DEBUG, g_sparsity {grouped_sparsities.shape}")
        # (num_layer * num_key_value_heads, ), ASSUMES BATCH = 1
        flattened_sparsities = grouped_sparsities.flatten()
        sorted_sparsities, sorted_indices = torch.sort(flattened_sparsities)

        # Per-head separation group
        separation = self.find_elbow_and_separate_recursive(
            sorted_sparsities, sorted_indices, self.n_recursion)
        # print("DEBUG", separation)

        # Budget allocation for each group
        budget_to_token = self.cache_budget * \
            self.config.num_key_value_heads * self.config.num_hidden_layers
        separation_gap = torch.zeros(len(separation))
        separation_size = torch.tensor(
            [len(group) for group in separation], dtype=separation_gap.dtype)
        for i in range(len(separation)):
            group_max = sorted_sparsities[separation[i][0]]
            separation_gap[i] = group_max
        allocation_weights = torch.softmax(separation_gap, dim=0)
        budget_allocation = allocation_weights * \
            budget_to_token / separation_size       # Shape: (2^n)

        # eviction
        old_key_cache = self.key_cache
        old_value_cache = self.value_cache
        summation_gqa = summation.view(
            l, b, -1, repeat_factor, qlen).mean(axis=3)  # NOTE: tuning point
        # print(f"DEBUG, g_summation {summation_gqa.shape}")
        # take summation_gqa (l,b, num_kv_head, qlen), separation, budget_allocation
        new_key_cache = [
            [None for _ in range(self.config.num_key_value_heads)]
            for _ in range(self.config.num_hidden_layers)
        ]
        new_value_cache = [
            [None for _ in range(self.config.num_key_value_heads)]
            for _ in range(self.config.num_hidden_layers)
        ]
        for i in range(len(separation)):
            b_i = int(budget_allocation[i])
            for j in range(len(separation[i])):
                g_id = separation[i][j]
                layer_idx = g_id // self.config.num_key_value_heads
                kv_head_idx = g_id % self.config.num_key_value_heads
                keep_idx = self.head_eviction(
                    summation_gqa[layer_idx, 0, kv_head_idx, :], layer_idx, kv_head_idx, b_i)
                new_key_cache[layer_idx][kv_head_idx] = \
                    self.key_cache[layer_idx][0, kv_head_idx, keep_idx, :].clone()
                new_value_cache[layer_idx][kv_head_idx] = \
                    self.value_cache[layer_idx][0, kv_head_idx, keep_idx, :].clone()
        self.key_cache = new_key_cache
        self.value_cache = new_value_cache

        # Explicit delete
        del old_key_cache
        del old_value_cache

        # print("---- EVICTION DONE ----")

    def head_eviction(self, scorer, l_id, h_id, budget):
        """
            budget is the number of token to keep
        """
        # print(
        #    f"DEBUG [head_eviction], scorer {scorer.shape}, l={l_id}, h={h_id}, b={budget}")
        if budget < self.recent_protect_budget:
        #    print(
        #        "DEBUG [head_eviction] Force retaining the latest window, return")
            return torch.arange(
            scorer.size(0) - self.recent_protect_budget, scorer.size(0), device=scorer.device)
        
        if budget - self.recent_protect_budget > scorer[:-self.recent_protect_budget].numel():
            return torch.arange(0, scorer.size(0), device=scorer.device)

        # Identify indices to keep using top-k and protect the recent window
        _, keep_idx = torch.topk(
            scorer[:-self.recent_protect_budget], k=budget - self.recent_protect_budget, largest=True)
        keep_idx = torch.cat((keep_idx, torch.arange(
            scorer.size(0) - self.recent_protect_budget, scorer.size(0), device=keep_idx.device)))
        keep_idx = keep_idx.sort().values
        # print(f"DEBUG [head_eviction] Keeping indices: {keep_idx.tolist()}")
        return keep_idx

    def find_elbow_and_separate_recursive(self, sorted_sparsities, sorted_indices, n_recursion, depth=0):
        """
        Recursively partitions sparsities into 2^n_recursive segments by finding elbow points.

        Parameters:
        sorted_sparsities (torch.Tensor): Flattened and sorted sparsity values.
        sorted_indices (torch.Tensor): Indices corresponding to the sorted sparsity values.
        n_recursive (int): Number of recursive splits (resulting in 2^n_recursive segments).
        depth (int): Current depth of recursion.

        Returns:
        list of list of int: Partitioned indices after recursive splitting.
        """
        if depth == n_recursion:
            return [sorted_indices.tolist()]

        # Find the elbow point
        n = sorted_sparsities.size(0)
        x = torch.arange(n, dtype=torch.float32).to(sorted_sparsities.device)
        start = torch.stack((x[0], sorted_sparsities[0]))
        end = torch.stack((x[-1], sorted_sparsities[-1]))
        line_vec = end - start
        line_vec_norm = line_vec / torch.norm(line_vec)

        # Vectorize
        points = torch.stack((x, sorted_sparsities), dim=1)
        vec_to_points = points - start
        proj_lengths = torch.matmul(vec_to_points, line_vec_norm)
        proj_points = start + proj_lengths.unsqueeze(1) * line_vec_norm
        distances = torch.norm(points - proj_points, dim=1)
        elbow_index = torch.argmax(distances).item()

        first_indices = sorted_indices[:elbow_index + 1]
        second_indices = sorted_indices[elbow_index + 1:]

        # Recursion
        first_partition = self.find_elbow_and_separate_recursive(
            sorted_sparsities[:elbow_index + 1],
            first_indices,
            n_recursion,
            depth + 1
        )
        second_partition = self.find_elbow_and_separate_recursive(
            sorted_sparsities[elbow_index + 1:],
            second_indices,
            n_recursion,
            depth + 1
        )
        return first_partition + second_partition

    def clear(self):
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = [
            None] * self.config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [
            None] * self.config.num_hidden_layers
        self.query_proxy: List[torch.Tensor] = [
            None] * self.config.num_hidden_layers

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None
