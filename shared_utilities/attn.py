import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import sqrt
from masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn



class SparseAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(SparseAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _sparse_QK(self, Q, K, n_top):

        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Calculate Q_K
        Q_K = torch.matmul(Q, K.transpose(-2, -1))
        
        # Find the Top_k query with sparsity measurement
        M = Q_K.max(-1)[0] - torch.div(Q_K.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K_reduced = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K_reduced, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        u = u if u < L_Q else L_Q

        scores_top, index = self._sparse_QK(queries, keys, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class MicroscaleAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, window_size=12, overlap=6, dropout=0.1, mask_flag=False):
        """
        Implements Microscale Multi-Head Attention with local windowing.

        Args:
            d_model (int): Feature dimension (D).
            n_heads (int): Number of attention heads.
            window_size (int): Size of each local attention window.
            overlap (int): Overlap between adjacent windows.
            dropout (float): Dropout rate.
            mask_flag (bool): Whether to apply a causal mask (used in the decoder).
        """
        super(MicroscaleAttention, self).__init__()

        assert d_model % n_heads == 0, "Feature dimension must be divisible by n_heads."
        self.head_dim = d_model // n_heads  # Dimension per head
        self.n_heads = n_heads
        self.d_model = d_model
        self.window_size = window_size
        self.overlap = overlap
        
        self.mask_flag = mask_flag  # Store mask flag

        # Multi-head projections
        self.proj_q = nn.Linear(self.head_dim, self.head_dim, bias=False)  # ✅ Correct
        self.proj_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.proj_v = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)

    def _apply_mask(self, attn_scores, attn_mask=None):
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask == 0, float('-inf'))
        elif self.mask_flag:
            L = attn_scores.size(-1)
            causal_mask = torch.tril(torch.ones(L, L, device=attn_scores.device))
            attn_scores.masked_fill_(causal_mask == 0, float('-inf'))
        return attn_scores
    


    def forward(self, queries, keys, values, attn_mask=None):
        # print(f"Before MicroscaleAttention - Queries Shape: {queries.shape}")
        B, H, L, D = queries.shape  # (B, H, L, D)
        step = self.window_size - self.overlap  # ✅ step = 12 - 6 = 6
        num_windows = (L - self.overlap) // step  # ✅ num_windows = (72 - 6) / 6 = 11

        # print(f"Before MicroscaleAttention - Queries Shape: {queries.shape}")
        # print(f"Debug - window_size: {self.window_size}, overlap: {self.overlap}, step: {step}, num_windows: {num_windows}")

        # === 1. Unfold along sequence length (L) ===
        # Ensure the unfolding is applied along the sequence length (dim=2)
        # print(f"Before Unfolding - Queries Shape: {queries.shape}")
        queries = queries.permute(0, 2, 1, 3)  # (B, H, L, D)
        # print(f"After Permutation - Queries Shape: {queries.shape}")
        queries_unfold = queries.unfold(2, self.window_size, step)  # Unfold along sequence length (L)
        # print(f"After Unfolding - Queries Unfolded Shape: {queries_unfold.shape}")
        queries_unfold = queries_unfold.permute(0, 1, 2, 4, 3)  # (B, H, num_windows, window_size, D)
        # print(f"After out permute - Queries Unfolded Shape: {queries_unfold.shape}")

        keys_unfold = keys.permute(0, 2, 1, 3).unfold(2, self.window_size, step)
        keys_unfold = keys_unfold.permute(0, 1, 2, 4, 3)   # (B, H, num_windows, window_size, D)
        # print(f"After Unfolding - keys Unfolded Shape: {keys_unfold.shape}")

        values_unfold = values.permute(0, 2, 1, 3).unfold(2, self.window_size, step)
        values_unfold = values_unfold.permute(0, 1, 2, 4, 3)  # (B, H, num_windows, window_size, D)
        # print(f"After Unfolding - values Unfolded Shape: {values_unfold.shape}")



        # === 2. Validate shape ===
        B, H, num_windows, window_size, head_dim = queries_unfold.shape

        assert window_size == self.window_size, f"Window size mismatch! Expected {self.window_size}, got {window_size}"
        assert head_dim == self.head_dim, f"Head dimension changed! Expected {self.head_dim}, got {head_dim}"

        # === 3. Apply Linear Projections (Before Reshaping) ===
        Q = self.proj_q(queries_unfold)  # (B, H, num_windows, window_size, D)
        K = self.proj_k(keys_unfold)
        V = self.proj_v(values_unfold)

        # === 4. Compute Attention Scores (B, H, num_windows, window_size, window_size) ===
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # === 5. Apply Attention Mask ===
        if self.mask_flag and attn_mask is not None:
            attn_scores = self._apply_mask(attn_scores, attn_mask)

        # === 6. Apply Softmax and Dropout ===
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)

        # === 7. Compute Attention Output ===
        attn_output = torch.matmul(attn_weights, V)  # (B, H, num_windows, window_size, D)
        # print(f"After Attention Output - Output Shape: {attn_output.shape}")

        # === 8. Reshape for Aggregation ===
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(B, num_windows, window_size, H * head_dim)
        # Now: (B, num_windows, window_size, D_total)
        # print(f"After reshaping for aggregation - Output Shape: {attn_output.shape}")

        # """Alternative weighted aggregation"""
        # # === 9. Aggregate Overlapping Windows with Weighted Summation ===
        # output = torch.zeros(B, self.window_size + step * (num_windows - 1), H * head_dim, device=queries.device)
        # weight_sum = torch.zeros(B, self.window_size + step * (num_windows - 1), device=queries.device)

        # for i in range(num_windows):
        #     start_idx = i * step
        #     end_idx = start_idx + self.window_size

        #     # Compute attention-based weights
        #     attention_weights = attn_weights[:, :, i, :, :].mean(dim=1)  # Average over heads (B, window_size, window_size)
        #     window_weight = attention_weights.sum(dim=-1)  # Sum attention scores over the query axis to get a weight per time step (B, window_size)

        #     # Apply weighted summation
        #     output[:, start_idx:end_idx, :] += attn_output[:, i, :, :] * window_weight.unsqueeze(-1)
        #     weight_sum[:, start_idx:end_idx] += window_weight  # Accumulate weights

        # # Normalize by total weight
        # output /= (weight_sum.unsqueeze(-1) + 1e-6)  # Avoid division by zero

        # === 9. Aggregate Overlapping Windows ===
        # Ensure correct length for `output` to match attention mechanism's sequence length
        output = torch.zeros(B, self.window_size + step * (num_windows - 1), H * head_dim, device=queries.device)
        count = torch.zeros(B, self.window_size + step * (num_windows - 1), device=queries.device)

        # print(f"Before Aggregation - Output Shape: {output.shape}")  # Should match sequence length

        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + self.window_size
            output[:, start_idx:end_idx, :] += attn_output[:, i, :, :]
            count[:, start_idx:end_idx] += 1

        output /= count.unsqueeze(-1)  # Normalize overlapping regions
        # # print(f"After Aggregation - Output Shape: {output.shape}")
        # === 10. Merge Heads Back (B, L, D) ===
        # Ensure correct output length
        correct_L = self.window_size + step * (num_windows - 1)  # Compute correct sequence length

        output = output.reshape(B, correct_L, H * self.head_dim)
        # print(f"Final Output Shape After Reshaping: {output.shape}")  # Should be (256, 72, 128)
        # print(f"queries shape: {queries.shape}")    
        queries = queries.permute(0, 2, 1, 3)
        # print(f"queries shape: {queries.shape}")
        B, L, H, D = queries.shape  # Dynamically get the correct sequence length
        queries = queries.reshape(B, L, H * D)  # Ensure queries are (B, L, H * D)
        # print(f"queries shape: {queries.shape}")
        # print(f"queries shape: {queries.shape}")




        # print(f"After merging heads - Output Shape: {output.shape}")
        # === 11. First Residual Connection & Normalization ===
        z = self.norm1(output + queries)
        # print(f"z shape: {z.shape}")
        # print(f"After First Residual Connection - Output Shape: {z.shape}")
        # === 12. Max Pooling ===
        # if output.dim() == 3:  # Ensure `output` is still in (B, L, D_model)
        #     output = self.pool(output.permute(0, 2, 1)).permute(0, 2, 1)
        # # print(f"After Max Pooling - Output Shape: {output.shape}")
        # # if output in not == L, pad with zeros at the start
        # if output.shape[1] < length:  # Only pad if output is shorter
        #     pad = torch.zeros(B, length - output.shape[1], output.shape[2], device=queries.device)
        #     output = torch.cat((pad, output), dim=1)
        # print(f"After Padding - Output Shape: {output.shape}")

        # === 13. Second Residual Connection & Normalization ===
        output = z

        return output, attn_weights






class DynamicMicroscaleAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, window_size=6, overlap=3, dropout=0.1, mask_flag=False):
        """
        Implements Microscale Multi-Head Attention with dynamically adjustable temperature scaling
        and a dual attention mechanism (DAGA) for long-term dependencies.
        """
        super(DynamicMicroscaleAttention, self).__init__()

        assert d_model % n_heads == 0, "Feature dimension must be divisible by n_heads."
        self.head_dim = d_model // n_heads  
        self.n_heads = n_heads
        self.d_model = d_model
        self.window_size = window_size
        self.overlap = overlap
        self.mask_flag = mask_flag  

        # Multi-head projections
        self.proj_q = nn.Linear(self.head_dim, self.head_dim, bias=False)  
        self.proj_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.proj_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # Global Attention Projections
        self.global_proj_q = nn.Linear(d_model, d_model, bias=False)
        self.global_proj_k = nn.Linear(d_model, d_model, bias=False)
        self.global_proj_v = nn.Linear(d_model, d_model, bias=False)

        # **Learnable temperature parameter for dynamic scaling in local attention**
        self.localtemperature = nn.Parameter(torch.sigmoid(torch.ones(1)) + 0.1)
        self.globaltemperature = nn.Parameter(torch.sigmoid(torch.ones(1)) + 0.1)
        # **Learnable weight for fusing local and global attention outputs**
        self.alpha = nn.Parameter(torch.sigmoid(torch.ones(1)))  # Constrained to [0,1]

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, D = queries.shape  
        step = self.window_size - self.overlap  
        num_windows = (L - self.overlap) // step  

        # === 1. Unfold along sequence length (Microscale Attention) ===
        queries_unfold = queries.unfold(1, self.window_size, step).permute(0, 1, 4, 2, 3)
        keys_unfold = keys.unfold(1, self.window_size, step).permute(0, 1, 4, 2, 3)
        values_unfold = values.unfold(1, self.window_size, step).permute(0, 1, 4, 2, 3)

        B, num_windows, window_size, H, head_dim = queries_unfold.shape

        # === 2. Apply Linear Projections ===
        Q_micro = self.proj_q(queries_unfold)
        K_micro = self.proj_k(keys_unfold)
        V_micro = self.proj_v(values_unfold)

        # Restore shape
        Q_micro = Q_micro.reshape(B, H, num_windows, window_size, head_dim)
        K_micro = K_micro.reshape(B, H, num_windows, window_size, head_dim)
        V_micro = V_micro.reshape(B, H, num_windows, window_size, head_dim)

        # === 3. Compute Microscale Attention Scores (WITH Temperature Adjustment) ===
        attn_scores_micro = torch.matmul(Q_micro, K_micro.transpose(-2, -1)) / self.localtemperature
        attn_weights_micro = self.softmax(attn_scores_micro)
        attn_weights_micro = self.dropout(attn_weights_micro)

        attn_output_micro = torch.matmul(attn_weights_micro, V_micro)

        # === 4. Weighted Local Aggregation (From Microscale Attention) ===
        output = torch.zeros(B, L, H * head_dim, device=queries.device)
        weight_sum = torch.zeros(B, L, device=queries.device)

        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + self.window_size

            # Compute attention-based weights
            local_weights = attn_weights_micro[:, :, i, :, :].mean(dim=1)  # (B, window_size, window_size)
            local_weights = local_weights.sum(dim=-1)  # (B, window_size)
            local_weights = local_weights.unsqueeze(-1)  # (B, window_size, 1)

            # Reshape attention output before adding to output
            attn_output_window = attn_output_micro[:, :, i, :, :].reshape(B, H, window_size, head_dim)
            attn_output_window = attn_output_window.permute(0, 2, 1, 3).reshape(B, window_size, H * head_dim)

            # Apply weighted summation
            output[:, start_idx:end_idx, :] += attn_output_window * local_weights
            weight_sum[:, start_idx:end_idx] += local_weights.squeeze(-1)

        # Normalize by total weight
        output /= (weight_sum.unsqueeze(-1) + 1e-6)  # Avoid division by zero

        # === 5. Compute Global Attention (DAGA) ===
        Q_global = self.global_proj_q(queries.reshape(B, L, H * head_dim))  # Flatten heads
        K_global = self.global_proj_k(keys.reshape(B, L, H * head_dim))
        V_global = self.global_proj_v(values.reshape(B, L, H * head_dim))

        attn_scores_global = torch.matmul(Q_global, K_global.transpose(-2, -1)) / self.globaltemperature
        attn_weights_global = self.softmax(attn_scores_global)
        attn_weights_global = self.dropout(attn_weights_global)

        attn_output_global = torch.matmul(attn_weights_global, V_global)

        # === 6. Fuse Local and Global Attention Outputs Using Alpha ===
        fused_output = self.alpha * output + (1 - self.alpha) * attn_output_global

        # === 7. Final Residual Connection & Normalization ===
        queries = queries.reshape(B, L, H * head_dim)
        z = self.norm1(fused_output + queries)

        return z, attn_weights_micro


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
