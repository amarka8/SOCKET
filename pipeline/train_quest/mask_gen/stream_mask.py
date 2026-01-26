import torch


def get_init_local(config, prompt_len: torch.Tensor):
    """
    Args:
        config: dict with keys:
            - 'n_init'  : float or int; <1 -> ratio of prompt_len, >=1 -> absolute count
            - 'n_local' : float or int; <1 -> ratio of prompt_len, >=1 -> absolute count
        prompt_len: (B,) int tensor = number of non-pad prompt tokens per sample

    Returns:
        init_len:  (B,) int tensor  -- number of sink tokens per sample
        local_len: (B,) int tensor  -- local window size per sample (computed from prompt_len if ratio)
    """
    n_init  = config['n_init']
    n_local = config['n_local']

    def to_count(value):
        # ratio if <1.0; absolute otherwise
        pl_f = prompt_len.to(torch.float32)
        if isinstance(value, float) and value < 1.0:
            cnt = torch.floor(pl_f * value).to(dtype=prompt_len.dtype)
        else:
            cnt = torch.full_like(prompt_len, int(value))
        cnt = torch.clamp(cnt, min=0)
        return cnt

    init_len  = torch.minimum(to_count(n_init), prompt_len)

    local_len = torch.minimum(to_count(n_local), prompt_len)

    return init_len, local_len

def gen_stream_mask(config, attention_mask, prompt_len, dtype):
    """
    StreamLLM additive attention bias where each query can attend to:
      - the first `sink_len[i]` *prompt* tokens of sample i (global sinks)
      - a sliding window of size `window_size` of recent tokens
      - with causality + padding respected

    Args:
        attention_mask: (B, T)  1=real, 0=pad
        prompt_len:     (B,)    # non-pad prompt tokens per sample
    Returns:
        attn_bias: (B, 1, T, T) additive mask (0 allowed, -inf blocked)
    """
    B, T = attention_mask.shape
    device = attention_mask.device

    # --- sanitize sink_len ---
    sink_len, local_len = get_init_local(config, prompt_len)

    # --- identify sink keys: "first sink_len[i] non-pad tokens" ---
    # token_idx counts non-pad positions 1..L irrespective of left/right padding.
    token_idx   = attention_mask.cumsum(dim=1)                    # (B, T)
    is_sink_key = (token_idx <= sink_len.view(B, 1)) & (attention_mask == 1)  # (B, T)

    # --- causal ---
    t = torch.arange(T, device=device)
    q = t.view(1, T, 1)            # (1, T, 1)
    k = t.view(1, 1, T)            # (1, 1, T)
    causal_ok = (k <= q)           # (1, T, T)
    
    # recent_ok[i, q, k] := k >= q - (local_len[i]-1)
    # broadcast local_len over (T,T) per batch
    qB  = q.expand(B, T, 1)                                      # (B, T, 1)
    kB  = k.expand(B, 1, T)                                      # (B, 1, T)
    win = (local_len.to(dtype=q.dtype).view(B, 1, 1) - 1)        # (B, 1, 1)
    recent_ok = (kB >= (qB - win))                               # (B, T, T)

    # enforce causality
    recent_ok = recent_ok & causal_ok                            # (B, T, T) via broadcast

    # --- broadcast & combine ---
    allow_recent = recent_ok.view(B, 1, T, T)                    # (B, 1, T, T)
    sink_ok      = is_sink_key.view(B, 1, 1, T).expand(B, 1, T, T)
    valid_key    = (attention_mask == 1).view(B, 1, 1, T)
    valid_query  = (attention_mask == 1).view(B, 1, T, 1)

    allow = (allow_recent | sink_ok) & valid_key & valid_query

    # --- build additive bias ---
    attn_bias = torch.zeros((B, 1, T, T), device=device, dtype=dtype)
    attn_bias.masked_fill_(~allow, torch.finfo(dtype).min)
    return attn_bias

def gen_stream_mask_decode(config, attention_mask, prompt_len, dtype):
    """
    Decoding-time StreamLLM additive attention bias for a *single query* step.

    Args:
        config: dict consumed by get_init_local (produces sink_len, local_len)
        attention_mask: (B, K)  1 for existing KV keys, 0 for pads
        prompt_len:         (B,)    count of *non-pad* prompt tokens
        dtype:              torch.dtype (must match queries, e.g., inputs_embeds.dtype)

    Returns:
        attn_bias: (B, 1, 1, K) additive bias (0 allowed, -inf blocked)
    """
    B, K = attention_mask.shape
    device = attention_mask.device

    # per-sample lengths
    sink_len, local_len = get_init_local(config, prompt_len)   # sink_len: (B,), local_len: (B,) or (B,1)
    sink_len  = sink_len.to(device=device)
    local_len = local_len.to(device=device)
    if local_len.dim() == 2 and local_len.size(1) == 1:
        local_len = local_len.squeeze(1)
    local_len = torch.clamp(local_len, min=0)

    # Identify sink keys: the first sink_len[i] *non-pad* tokens
    token_idx   = attention_mask.cumsum(dim=1)                     # (B, K) counts 1..L over non-pads
    is_sink_key = (token_idx <= sink_len.view(B, 1)) & (attention_mask == 1)  # (B, K) bool

    # Recent window: keep only the last local_len[i] valid keys
    # Compute the last non-pad position per sample (K is the buffer len; some may be padded).
    valid_counts = attention_mask.sum(dim=1)                        # (B,)
    # Indices 0..K-1; for sample i, allow k >= valid_counts[i] - local_len[i]
    k_idx = torch.arange(K, device=device).view(1, K).expand(B, K)      # (B, K)
    start_idx = (valid_counts - local_len).clamp(min=0).view(B, 1)      # (B, 1)
    is_recent = k_idx >= start_idx                                      # (B, K) bool

    # Combine: (sink OR recent) AND valid key
    allow = (is_sink_key | is_recent) & (attention_mask == 1)       # (B, K)

    # Build additive bias (B,1,1,K) in the SAME dtype as queries
    attn_bias = torch.zeros((B, 1, 1, K), device=device, dtype=dtype)
    attn_bias.masked_fill_(~allow.view(B, 1, 1, K), torch.finfo(dtype).min)
    return attn_bias