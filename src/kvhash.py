import torch
import math
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
        n_recursive: int = 1
    ) -> None:
        super().__init__()
        self.config = config

        self.cache_budget = cache_budget
        self.recent_protect_budget = recent_protect_budget
        self.proxy_total = proxy_total
        self.proxy_latest = proxy_latest
        self.top_rank = top_rank
        self.n_recursive = n_recursive

        self._seen_tokens = 0

        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers     # [(batch, num_key_value_head, qlen, hidden)] * layer
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers   # [(batch, num_key_value_head, qlen, hidden)] * layer
        self.query_proxy: List[torch.Tensor] = [None] * self.config.num_hidden_layers   # [(batch, num_attention_head, proxy_total, hidden)] * layer

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

    def pca_select(self, query: torch.Tensor, key: torch.Tensor, r:int,  total: int, latest: int):     # select proxy from query
        s, h = query.shape    
        pca_center = key - key.mean(dim=0)
        if pca_center.dtype != torch.float32:
            pca_center = pca_center.to(torch.float32)
        U, S, Vh = torch.linalg.svd(pca_center, full_matrices=False)  # Vh: (h, h)
        w = min(r, h)
        top_PCs = (S[:w, None] * Vh[:w, :])  # (r, h)
        projection = torch.matmul(query, top_PCs.T)  # (s, r)
        importance_scores = projection.pow(2).sum(dim=1)  # (s,)
        _, top_k_importance_indices = torch.topk(importance_scores, k=total, largest=True)
        last_16_indices = torch.arange(s - latest, s, device=key.device)
        top_k_importance_indices = top_k_importance_indices[~torch.isin(top_k_importance_indices, last_16_indices)]
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

        if self.key_cache[layer_idx] is None or self.value_cache[layer_idx] is None:    # prefill only
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

            l_query_proxy = []
            for i in range(self.config.num_attention_heads):
                one_q = query_states[0,i]
                one_k = key_states[0,int(i//self.config.num_key_value_heads)]
                indices = self.pca_select(one_q, one_k ,self.top_rank, self.proxy_total, self.proxy_latest)
                one_proxy = one_q[indices,:]      # (proxy_total, num_hidden)
                l_query_proxy.append[one_proxy]
                # attn = torch.matmul(query_proxy, one_k.transpose(0, 1)) / math.sqrt(self.config.head_dim)
                # attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
                # attn_sum = torch.sum(attn, dim=0)
                # min_val = torch.min(attn_sum)
                # max_val = torch.max(attn_sum)
                # threshold = min_val + 0.001 * (max_val - min_val)
                # sparsity = torch.sum(attn_sum < threshold).item() / query_states.shape[2]
                # l_sparsity.append(sparsity)
            self.query_proxy[layer_idx] = torch.stack(l_query_proxy, dim=0).unsqueeze(0)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def evict(self):
        """
            self.key_cache      # [(batch, num_key_value_head, qlen, hidden)] * layer
            self.value_cache    # [(batch, num_key_value_head, qlen, hidden)] * layer
            self.query_proxy    # [(batch, num_attention_head, proxy_total, hidden)] * layer
        """
        key_cache = torch.stack(self.key_cache)         # Shape: (num_layers, batch, num_key_value_head, qlen, hidden)
        value_cache = torch.stack(self.value_cache)     # Shape: (num_layers, batch, num_key_value_head, qlen, hidden)
        query_proxy = torch.stack(self.query_proxy)     # Shape: (num_layers, batch, num_attention_head, proxy_total, hidden)
        l, b, gqah, qlen, hidden = key_cache.shape

        repeat_factor = self.config.num_attention_head // self.config.num_key_value_head
        key_cache_repeated = key_cache.repeat_interleave(repeat_factor, dim=2)  # Shape: (num_layers, batch, num_attention_head, qlen, hidden)

        # Compute attention scores
        attn_scores = torch.einsum(
            'lbqph,lbqkh->lbqpk',
            query_proxy,
            key_cache_repeated.transpose(-1, -2)
        ) / math.sqrt(self.config.head_dim)             # Shape: (num_layers, batch, num_attention_head, proxy_total, qlen)
        attn_probs = torch.softmax(attn_scores, dim=-1) # Shape: (num_layers, batch, num_attention_head, proxy_total, qlen)
        print(f"DEBUG, proxy attn {attn_probs.shape}")
        summation = torch.sum(attn_probs, dim=-2)       # Shape: (num_layers, batch, num_attention_head, qlen)
        print(f"DEBUG, summation {summation.shape}")
        maximum = torch.max(summation, dim=-1)          # Shape: (num_layers, batch, num_attention_head)
        minimum = torch.min(summation, dim=-1)          # Shape: (num_layers, batch, num_attention_head)
        print(f"DEBUG, min/max {maximum.shape}")
        threshold = minimum + 0.0005 * (maximum - minimum)
        sparsity = torch.sum(summation < threshold.unsqueeze(-1), dim=-1) / summation.shape[-1] # Shape: (num_layers, batch, num_attention_head)
        print(f"DEBUG, sparsity {sparsity.shape}")
        
        # GQA processing
        # Shape: (num_layers, batch, num_key_value_head)
        grouped_sparsities = sparsity.view(sparsity.shape[0], sparsity.shape[1], sparsity.shape[2], repeat_factor).mean(dim=-1)
        print(f"DEBUG, g_sparsity {grouped_sparsities.shape}")
        flattened_sparsities = torch.cat(grouped_sparsities, dim=1).squeeze(0)
        sorted_sparsities, sorted_indices = torch.sort(flattened_sparsities)

        # Per-head separation group
        separation = self.find_elbow_and_separate_recursive(sorted_sparsities, sorted_indices, self.n_recursive)
        print("DEBUG",separation)

        # Budget allocation for each group
        budget_to_token = self.cache_budget * self._seen_tokens * self.config.num_key_value_heads * self.config.num_hidden_layers   # for key_states only
        separation_gap = torch.zeros(len(separation))
        for i in range(len(separation)):
            gap = sorted_sparsities[separation[i][-1]] - sorted_sparsities[separation[i][0]]
            separation_gap[i] = gap
        allocation_weights = torch.softmax(separation_gap, dim=0)
        budget_allocation = allocation_weights * budget_to_token
        
        evict_ids = []  # Pure two d array, [[head1], [head2], ...]
        for i in range(self.config.num_key_value_heads):
            # head_eviction: random plane
            # head_eviction_topk: flat score
            
            #evict_id_per_head = self.head_eviction(evict_hash, evict_attn, i, evict_tokens)
            evict_id_per_head = self.head_eviction_topk(evict_attn, i, evict_tokens)[0]
            evict_id_per_head += self.sink_protect_tokens  # evict_hash and evict_attn trimmed by sink_protect_tokens
            evict_ids.append(evict_id_per_head)

        # align eviction length
        min_evict_tokens = min(len(evict_id_per_head) for evict_id_per_head in evict_ids)
        aligned_evict_ids = torch.stack([evict_id_per_head[:min_evict_tokens] for evict_id_per_head in evict_ids])

        # convert eviction idx to keep idx
        keep_indices = torch.ones((self.config.num_key_value_heads, q_len), dtype=torch.bool, device=evict_hash.device)
        keep_indices.scatter_(1, aligned_evict_ids, False)
        valid_indices = torch.masked_select(torch.arange(q_len, device=evict_hash.device).unsqueeze(0).expand(self.config.num_key_value_heads, -1), keep_indices).view(
            self.config.num_key_value_heads, -1
        )

        # house keeping (key_cache, value_cache, hash_values)
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

        assert self.key_cache[layer_idx].shape[2] == valid_indices.shape[1], "Mismatch in q_len after eviction!"

    def head_eviction(self, layer_idx, head_idx, evict_num):
        

    def find_elbow_and_separate_recursive(self, sorted_sparsities, sorted_indices, n_recursive, depth=0):
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
        if depth == self.n_recursive:
            return [sorted_indices.tolist()]

        # Find the elbow point
        n = sorted_sparsities.size(0)
        x = torch.arange(n, dtype=torch.float32)
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
            n_recursive,
            depth + 1
        )
        second_partition = self.find_elbow_and_separate_recursive(
            sorted_sparsities[elbow_index + 1:],
            second_indices,
            n_recursive,
            depth + 1
        )
        return first_partition + second_partition

    def clear(self):
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.query_proxy: List[torch.Tensor] = [None] * self.config.num_hidden_layers
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None