import torch
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import repeat_kv


class KVHashCache(Cache):
    def __init__(
        self,
        config,
        cache_budget,
        sink_protect_tokens,
        recent_protect_budget,
        device: str | None,
        top_k: int = 8,
        num_planes: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.cache_budget = cache_budget
        self.sink_protect_tokens = sink_protect_tokens
        self.recent_protect_budget = recent_protect_budget
        self._seen_tokens = 0
        self.num_planes = num_planes

        self.key_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers  # [(batch, num_head, qlen, hidden)] * layer
        self.value_cache: List[torch.Tensor] = [None] * self.config.num_hidden_layers  # [(batch, num_head, qlen, hidden)] * layer

        self.hash_values: List[torch.Tensor] = [None] * self.config.num_hidden_layers  # [(batch, num_head, qlen)] * layer
        # self.attn_sum: List[torch.Tensor] = [None] * self.config.num_hidden_layers
        self.register_buffer("div_planes", torch.randn((self.num_planes, self.config.head_dim), dtype=torch.bfloat16))
        self.register_buffer("powers_of_two", 2 ** torch.arange(self.num_planes - 1, -1, -1, dtype=torch.float32))

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.top_k = top_k

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

    def update_hash_values(self, layer_idx, key_states, q_len):  # Shape: (b, num_head, q_len, k_len)
        hash_bits = torch.matmul(key_states[:, :, -q_len:, :], self.div_planes.transpose(-1, -2))
        hash_bits = (hash_bits >= 0).to(torch.float32)
        hash_vals = torch.matmul(hash_bits, self.powers_of_two)  # Shape: (b, num_head, s_len)

        if self.hash_values[layer_idx] is None:
            self.hash_values[layer_idx] = hash_vals
        else:
            self.hash_values[layer_idx] = torch.cat([self.hash_values[layer_idx], hash_vals[:, :, -q_len:]], dim=2)

    def proxy_token_selection(self, query_states, k=128, tail=128):
        hash_bits = torch.matmul(query_states, self.div_planes.transpose(-1, -2))
        hash_bits = (hash_bits >= 0).to(torch.float32)
        hash_vals = torch.matmul(hash_bits, self.powers_of_two)  # shape (b, num_head, qlen)

        b, num_head, qlen, hidden_d = query_states.shape

        reshaped_hash_vals = hash_vals.view(-1, hash_vals.shape[-1])  # Shape: (b * num_head, qlen)
        range_limit = qlen - tail
        reshaped_hash_vals = hash_vals[:, :, :range_limit].reshape(-1, range_limit)
        sampled_indices = torch.empty((reshaped_hash_vals.shape[0], k), dtype=torch.long, device=self.device)
        for row_idx, row in enumerate(reshaped_hash_vals):
            unique_vals, counts = torch.unique(row, return_counts=True)
            probabilities = counts.float() / counts.sum()

            sampled_vals = torch.multinomial(probabilities, k, replacement=True)
            all_indices = torch.nonzero(row.unsqueeze(0) == unique_vals[sampled_vals].unsqueeze(1), as_tuple=False)[:, 1]
            shuffled_indices = all_indices[torch.randperm(all_indices.size(0))]
            sampled_indices[row_idx, :] = shuffled_indices[:k]
        sampled_indices = sampled_indices.view(b, num_head, k)  # (b, num_head, k)
        tail_indices = torch.arange(range_limit, qlen).expand(b, num_head, tail).to(self.device)  # (b, num_head, tail)
        proxy_indices = torch.cat([sampled_indices, tail_indices], dim=-1).unsqueeze(-1).expand(-1, -1, -1, hidden_d)  # (b, num_head, tail+k, hidden_d)
        return proxy_indices

    def pca_select(self, data: torch.Tensor, k: int):
        """pca_select: select seq_len from (seq_len, hidden) data using PCA.

        Args:
            data: torch.Tensor: (seq_len, hidden)
            k: number of top PCs to select

        Returns:
            indices: torch.Tensor: (k,)
        """
        assert data.dim() == 2, "Input tensor must have 2 dimensions (seqlen, hidden)."
        s, h = data.shape
        data_center = data - data.mean(dim=0)
        if data_center.dtype != torch.float32:
            data_center = data_center.to(torch.float32)
        U, S, Vh = torch.linalg.svd(data_center, full_matrices=False)
        top_PCs = Vh[:k, :]  # Shape: (k, h)
        # Projection: (s, k)
        projection = torch.matmul(data_center, top_PCs.T)
        importance_scores = projection.pow(2).sum(dim=1)
        _, top_k_indices = torch.topk(importance_scores, k=k, largest=True, sorted=True)  # Shape: (k,)
        return top_k_indices

    def select_q_pca_svd(self, data: torch.Tensor, k: int) -> torch.Tensor:
        """
        Select top k sequence indices based on their contribution to the variance using PCA via SVD.
        (For every b, n, select top k indices from seqlen)

        Args:
            data (torch.Tensor): Input tensor of shape (batch, num_head, seqlen, hidden).
            k (int): Number of sequence indices to select.

        Returns:
            torch.Tensor: Indices of shape (batch, num_head, k) representing the selected sequence positions.
        """
        b, n, s, h = data.shape

        data_centered = data - data.mean(dim=2, keepdim=True)  # Shape: (b, n, s, h)

        data_reshaped = data_centered.view(b * n, s, h)  # Shape: (b*n, s, h)

        # U: (b*n, s, s), S: (b*n, min(s, h)), Vh: (b*n, h, h)
        U, S, Vh = torch.linalg.svd(data_reshaped, full_matrices=False)  # SVD is batched

        top_PCs = Vh[:, :k, :]  # Shape: (b*n, k, h)

        # Projection: (b*n, s, k)
        projection = torch.matmul(data_reshaped, top_PCs.transpose(-2, -1))  # (b*n, s, k)

        # Shape: (b*n, s)
        importance_scores = projection.pow(2).sum(dim=2)  # Sum over the k principal components

        # Indices with the highest scores are most important
        _, top_k_indices = torch.topk(importance_scores, k=k, dim=1, largest=True, sorted=True)  # Shape: (b*n, k)

        selected_indices = top_k_indices.view(b, n, k)  # Shape: (b, n, k)

        return selected_indices

    def select_q_pca2(self, data: torch.Tensor, k: int):
        b, n, s, h = data.shape
        # Every b, n would choose a different set of indices tokens
        indices = torch.zeros((b, n, k), device=data.device, dtype=torch.long)
        for i in range(b):
            for j in range(n):
                indices[i, j] = self.pca_select(data[i, j], k)
        return indices

    def select_q_pca(self, x: torch.Tensor, k: int):
        """
        Select top k sequence indices based on their contribution to the variance using PCA.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_head, seqlen, hidden).
            k (int): Number of sequence indices to select.

        Returns:
            torch.Tensor: Indices of shape (k,) representing the selected sequence positions.
                --> no matter of the "batch" or "num_head" or "hidden" dimensions
        """
        # Step 1: Get the dimensions
        batch, num_head, seqlen, hidden = x.shape
        # print(f"Input shape: {x.shape}")

        # New shape: (batch, num_head, hidden, seqlen)
        x_permuted = x.permute(0, 1, 3, 2)
        # print(f"Permuted shape: {x_permuted.shape}")

        # New shape: (batch * num_head * hidden, seqlen)
        x_reshaped = x_permuted.contiguous().view(-1, seqlen)
        # print(f"Reshaped shape for PCA: {x_reshaped.shape}")

        # Shape: (batch*num_head*hidden, 1)
        mean = torch.mean(x_reshaped, dim=1, keepdim=True)
        x_centered = x_reshaped - mean
        # print(f"Data centered. Mean shape: {mean.shape}")

        if x_centered.dtype != torch.float32:
            x_centered = x_centered.to(torch.float32)
            # print("Data cast to float32 for covariance computation.")

        # Compute covariance matrix: (seqlen, seqlen)
        covariance = torch.matmul(x_centered.T, x_centered) / (x_centered.size(0) - 1)
        # print(f"Covariance matrix shape: {covariance.shape}")

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        except RuntimeError as e:
            # print(f"Eigen decomposition failed: {e}")
            # As a fallback, move data to CPU and perform eigen decomposition
            covariance_cpu = covariance.cpu()
            eigenvalues_cpu, eigenvectors_cpu = torch.linalg.eigh(covariance_cpu)
            eigenvalues = eigenvalues_cpu.to(covariance.device)
            eigenvectors = eigenvectors_cpu.to(covariance.device)
            # print("Eigen decomposition performed on CPU and moved back to original device.")

        sorted_indices = torch.argsort(eigenvalues, descending=True)
        top_eigenvectors = eigenvectors[:, sorted_indices[:k]]  # Shape: (seqlen, k)

        importance_scores = torch.sum(top_eigenvectors**2, dim=1)  # Shape: (seqlen,)
        # print(f"Importance scores shape: {importance_scores.shape}")

        # Higher scores indicate higher contribution to variance
        _, top_k_indices = torch.topk(importance_scores, k=k, largest=True, sorted=True)
        # print(f"Top {k} indices: {top_k_indices}")

        # Optionally, cast indices to long type if not already
        top_k_indices = top_k_indices.long()

        return top_k_indices

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert cache_kwargs is not None, "cache_kwargs must be provided for KVHashCache"
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.key_cache[layer_idx] is None or self.value_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        assert self.key_cache[layer_idx].shape[2] == self.value_cache[layer_idx].shape[2], "Mismatch in the sequence length of K and V Cache"
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def evict(self, layer_idx: int, query_states: torch.Tensor):
        if query_states.shape[-2] == 1:  # skip autoregressive
            return

        self.update_hash_values(layer_idx, self.key_cache[layer_idx], query_states.shape[-2])

        # compute budgets
        q_len = self.key_cache[layer_idx].shape[2]
        recent_protect_tokens = int(self.recent_protect_budget * q_len)
        if recent_protect_tokens == 0:
            recent_protect_tokens = 1
        eviction_zone = q_len - recent_protect_tokens - self.sink_protect_tokens
        evict_tokens = q_len - int(self.cache_budget * self._seen_tokens)
        if evict_tokens > eviction_zone:
            return
        if evict_tokens == 0:
            return
        assert evict_tokens > 0, "the number of tokens need to be evicted should be larger than 0"

        if layer_idx == 0:
            print(f"q_len = {q_len}, recent_protect = {recent_protect_tokens}, eviction_zone = {eviction_zone}, evict_tokens = {evict_tokens}")

        # print(f"self.device: {self.device}")
        # print(f"query_states shape: {query_states.shape}, on {query_states.device}")

        # proxy token selection
        # proxy_indices = self.proxy_token_selection(query_states)  # (b, num_head, tail+k, hidden_d)
        # print(f"DEBUG: PXY: {proxy_indices.shape}")
        # proxy_query_states = torch.gather(query_states, dim=2, index=proxy_indices)  # (b, num_head, tail+k, hidden_d)

        # NOTE: PCA implementation
        proxy_indices = self.select_q_pca2(query_states, self.top_k)  # (b, num_head, k)
        # print(f"DEBUG: PCA: {proxy_indices.shape}")
        # proxy_indices = proxy_indices.unsqueeze(-1).repeat(1, 1, 1, query_states.shape[-1])# (b, num_head, k, hidden_d)
        proxy_indices = proxy_indices.unsqueeze(-1).expand(-1, -1, -1, query_states.shape[-1])  # (b, num_head, k, hidden_d)
        proxy_query_states = torch.gather(query_states, dim=2, index=proxy_indices)  # (b, num_head, tail+k, hidden_d)
        # proxy_query_states = query_states[:, :, proxy_indices, :]

        # calculate proxy attention scores
        key_to_mul = repeat_kv(self.key_cache[layer_idx], self.config.num_attention_heads // self.config.num_key_value_heads)
        attn_weights = torch.matmul(proxy_query_states, key_to_mul.transpose(2, 3)) / math.sqrt(self.config.head_dim)
        summation = torch.sum(attn_weights, dim=2)
        n_per_group = self.config.num_attention_heads // self.config.num_key_value_heads
        if n_per_group > 1:
            temp = summation.view(summation.shape[0], self.config.num_key_value_heads, n_per_group, -1)
            summation = temp.sum(dim=2)  # summation shape = (b, 8, qlen)

        evict_hash = self.hash_values[layer_idx][:, :, self.sink_protect_tokens : -recent_protect_tokens]  # Shape (b, num_heads, qlen)
        evict_attn = summation[:, :, self.sink_protect_tokens : -recent_protect_tokens]  # Shape (b, num_heads, qlen)
        evict_ids = []  # Pure two d array, [[head1], [head2], ...]
        for i in range(self.config.num_key_value_heads):
            # head_eviction: random plane
            # head_eviction_topk: h2o implementation
            # pca_selection: can be used to directly select k cache
            evict_id_per_head = self.head_eviction(evict_hash, evict_attn, i, evict_tokens)
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
        #     print(f"===== {self.key_cache[layer_idx].shape[2]} -- {int(self._seen_tokens * self.cache_budget)}/{self._seen_tokens}=====")
        return self.key_cache[layer_idx].shape[2] > self._seen_tokens * self.cache_budget

    def clear(self):
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

    def head_eviction_topk(self, hash, attn, head_idx, evict_num):
        # return eviction id purely based on attention sum, no hash involves
        _, indices = torch.topk(attn[:, head_idx, :], evict_num, largest=False)  # TODO: I think the return indices here needs to be reshape. Need GPT
        return indices

    def pca_selection(self, k, head_idx, evict_num):
        # This function serves as a pca-informed indices selection
        # for Q,K,V matrices like (bsz, num_head, q_len, hidden_d),
        # this function use pca of hidden_d to inform most informative
        # tokens on dim = -2
        # TODO fix dimemsion
        bsz, num_head, q_len, hidden_d = self.key_cache[head_idx].shape
        row_covariance = torch.matmul(self.key_cache[head_idx].transpose(-1, -2), self.key_cache[head_idx]) / q_len
        U, S, V = torch.linalg.svd(row_covariance, full_matrices=False)
        top_PCs = U[:, :k]  # (128k, k)
        projected_k_cache = torch.matmul(top_PCs.T, self.key_cache[head_idx])  # (k, 128k)@(128k, 128) -> (bsz, num_head, k, q_len)

        column_sum = torch.sum(projected_k_cache, dim=2)  # (bsz, num_head, q_len)
        _, indicees = torch.topk(column_sum, evict_num)
        return indicees
