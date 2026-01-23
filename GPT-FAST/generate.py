import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import itertools
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import os, sys
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import torch._inductor.config
import torch._dynamo.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tp import maybe_init_dist
from sentencepiece import SentencePieceProcessor
from kernels.sparse import build_sparse_list_decode, sparse_attention_fwd

@torch.no_grad()
def warmup_triton_sparse_decode(model: Transformer, device: torch.device, *, T: int = 4096, block_seq: int = 256):
    """
    Warm up BOTH:
      - build_sparse_list_decode (index_build)
      - sparse_attention_fwd (sparse_kernel)

    Call after model.setup_caches(...) and model is on `device`.
    """
    attn = model.layers[0].attention
    B = 1
    H = attn.n_head
    Hl = attn.n_local_heads
    D = attn.head_dim
    L = attn.L
    R = attn.R
    Ktotal = int(attn.heavy_const)

    # representative decode length (prefix length)
    T = int(T)
    T = max(T, 256)  # avoid tiny degenerate shapes

    # Build dummy inputs on device
    q_bhd = torch.empty((B, H, D), device=device, dtype=torch.bfloat16)

    # q_probs: [B,H,L,R]
    q_probs = torch.empty((B, H, L, R), device=device, dtype=torch.bfloat16).normal_()

    # k_hard_bhlt: [B,H,L,T] int16 in [0, R-1]
    k_hard_bhlt = torch.randint(
        low=0, high=R, size=(B, H, L, T), device=device, dtype=torch.int16
    )

    # v_norm_bht: [B,H,T] fp16
    v_norm_bht = torch.rand((B, H, T), device=device, dtype=torch.float16)

    # allowed_bht: [B,H,T] bool
    allowed_bht = torch.ones((B, H, T), device=device, dtype=torch.bool)

    # k/v backend: [B,Hl,T,D] bf16
    k_backend = torch.empty((B, Hl, T, D), device=device, dtype=torch.bfloat16)
    v_backend = torch.empty((B, Hl, T, D), device=device, dtype=torch.bfloat16)

    # Use same knobs as your decode
    sink = int(getattr(attn.config, "sink_size", 120))
    window = int(getattr(attn.config, "window_size", 120))
    M = int(attn.heavy_const)

    sink = max(0, min(sink, T))
    window = max(0, min(window, T))
    M = max(0, min(M, T))

    # Run twice to ensure compilation + any autotune paths complete
    for _ in range(2):
        sparse_list, sparse_len = build_sparse_list_decode(
            q_probs,
            k_hard_bhlt,
            v_norm_bht,
            allowed_bht,
            sink=sink,
            window=window,
            M=M,
            KC=8,
            BLOCK_N=512,
            num_warps=8,
            num_stages=2,
        )

        if sparse_list.dtype != torch.int32:
            sparse_list = sparse_list.to(torch.int32)
        if sparse_len.dtype != torch.int32:
            sparse_len = sparse_len.to(torch.int32)

        _ = sparse_attention_fwd(
            q_bhd,
            k_backend,
            v_backend,
            sparse_list,
            sparse_len,
            block_seq=block_seq,
        )

    torch.cuda.synchronize()


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    # assert torch.allclose(logits[0], logits[1], atol=1e-6, rtol=1e-6)
    # idx_next = multinomial_sample_one_no_sync(probs)
    idx_next = torch.argmax(probs, dim=-1, keepdim=True).to(dtype=torch.int) # TODO: change the sampling method
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, decode_type: str, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    if decode_type == "dense":
        logits = model(x, input_pos)
    else:
        logits = model.sparse_forward(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def sparse_decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model.sparse_forward(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    decode_type: str = "dense",   # "dense" | "sparse"
    **sampling_kwargs
):
    new_tokens, new_probs = [], []

    for _ in range(num_new_tokens):
        if decode_type == "dense":
            # Dense path
            with sdpa_kernel([SDPBackend.MATH]):
                next_token, next_prob = decode_one_token(
                    model, cur_token, input_pos, **sampling_kwargs
                )
        else:
            with sdpa_kernel([ SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, ]):
                next_token, next_prob = sparse_decode_one_token(
                    model, cur_token, input_pos, **sampling_kwargs
                )

        input_pos += 1
        next_token = next_token.clone()
        next_prob = next_prob.clone()

        new_tokens.append(next_token)
        new_probs.append(next_prob)
        callback(next_token)

        cur_token = next_token

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, decode_type="dense", **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    decode_type: str = "dense",
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    if is_speculative and decode_type != "dense":
        print("Warning: sparse decode is not supported for speculative decoding; using dense.")
        decode_type = "dense"
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    batch_size = prompt.size(0)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    T_warm = min(model.max_seq_length, prompt.size(1)) 
    print("T_warm:", T_warm)
    warmup_triton_sparse_decode(model, torch.device("cuda"), T=T_warm, block_seq=256)
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty((batch_size, T_new), dtype=dtype, device=device)
    empty[:,:T] = prompt
    seq = empty

    # TODO: All sequences share the same position for now
    input_pos = torch.arange(0, T, device=device)

    # --- Timing: prefill vs decode ---
    # We use CUDA events to avoid CPU-side noise. Synchronize boundaries so the split is clean.
    use_cuda_timing = (device.type == "cuda")
    if use_cuda_timing:
        prefill_start_evt = torch.cuda.Event(enable_timing=True)
        prefill_end_evt = torch.cuda.Event(enable_timing=True)
        decode_start_evt = torch.cuda.Event(enable_timing=True)
        decode_end_evt = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        prefill_start_evt.record()
    else:
        prefill_start_t = time.perf_counter()

    next_token = prefill(model, prompt, input_pos, decode_type, **sampling_kwargs)
    # model.build_tables(prefill_len=prompt.size(1))

    if is_speculative:
        prefill(draft_model, prompt, input_pos, "dense", **sampling_kwargs)

    if use_cuda_timing:
        prefill_end_evt.record()
        torch.cuda.synchronize()
        prefill_time_s = prefill_start_evt.elapsed_time(prefill_end_evt) / 1e3
        decode_start_evt.record()
    else:
        prefill_time_s = time.perf_counter() - prefill_start_t
        decode_start_t = time.perf_counter()
    seq[:,T] = next_token[:,0]

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(
            model,
            next_token,
            input_pos,
            max_new_tokens - 1,
            callback=callback,
            decode_type=decode_type,
            **sampling_kwargs,
        )
        seq[:,T + 1:] = torch.cat(generated_tokens, dim=1)

    if use_cuda_timing:
        decode_end_evt.record()
        torch.cuda.synchronize()
        decode_time_s = decode_start_evt.elapsed_time(decode_end_evt) / 1e3
    else:
        decode_time_s = time.perf_counter() - decode_start_t

    # Decode-only tokens exclude the first token produced by prefill.
    decode_only_tokens = batch_size * max(0, max_new_tokens - 1)

    generate_stats = {
        'accept_counts': accept_counts,
        'prefill_time_s': prefill_time_s,
        'decode_time_s': decode_time_s,
        'decode_only_tokens': decode_only_tokens,
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, batch_size, bos=True, device='cuda'):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    tokens_tensor = torch.tensor(tokens, dtype=torch.int, device=device)
    batch_tokens_tensor = tokens_tensor.unsqueeze(0).repeat(batch_size, 1)
    return batch_tokens_tensor

def _load_model(checkpoint_path, device, precision, use_tp):
    # 1) build on meta
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    # 2) keep your quantization path exactly (meta-safe module surgery)
    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    # 3) materialize meta tensors on the real device (this is the critical fix)
    model = model.to_empty(device=device)

    # 4) load checkpoint (CPU -> assign into already-materialized tensors)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)

    if missing:
        print("Missing keys (expected for new hash buffers):", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected:
        print("Unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")

    # 5) initialize any newly-added hash tensors that aren't in the ckpt
    #    (IMPORTANT: to_empty leaves them uninitialized garbage)
    needs_hash_init = any(("planes" in k) or ("protos_T" in k) for k in missing)
    if needs_hash_init:
        for m in model.modules():
            # if you registered these as buffers:
            if hasattr(m, "planes") and isinstance(m.planes, torch.Tensor):
                m.planes.normal_(mean=0.0, std=0.02)
            if hasattr(m, "protos_T") and isinstance(m.protos_T, torch.Tensor):
                m.protos_T.normal_(mean=0.0, std=0.02)

    # 6) tensor parallel after weights are in place
    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    # 7) cast dtype (DON'T pass device here; it's already on device)
    model = model.to(dtype=precision)

    return model.eval()


B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: str = "Hello, my name is",
    prompt_file: Optional[Path] = None,
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 1000,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    decode_type: str = "dense",
    batch_size: int = 1,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        torch.cuda.set_device(rank)
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    device = 'cuda'
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

    if (prompt_file is not None) and (not interactive):
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    encoded = encode_tokens(tokenizer, prompt, batch_size=batch_size, bos=True, device=device)
    prompt_length = encoded.size(1)

    torch.manual_seed(1234)
    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    if compile:
        if is_speculative and use_tp:
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
        # decode_one_token = torch.compile(decode_one_token, fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if args.compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        torch.cuda.synchronize()
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                decode_type=decode_type,
                temperature=temperature,
                top_k=top_k,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        if not interactive:
            print(tokenizer.decode(y.tolist()))
        else:
            print()
        tokens_generated = batch_size * (y.size(1) - prompt_length)
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        if metrics.get('decode_time_s', None) is not None:
            decode_time_s = metrics['decode_time_s']
            decode_only_tokens = metrics.get('decode_only_tokens', None)
            if decode_only_tokens is None:
                # Fallback if older metrics dict is used
                decode_only_tokens = batch_size * max(0, (y.size(1) - prompt_length - 1))
            if decode_time_s > 0:
                print(f"Decode-only: {decode_time_s:.04f} sec, {decode_only_tokens / decode_time_s:.02f} tokens/sec")
            print(f"Prefill: {metrics.get('prefill_time_s', 0.0):.04f} sec")
    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str, default="Write me a 1000 word story on soccer.", help='Input prompt.')
    parser.add_argument('--prompt_file', type=Path, default=None, help='Path to a text file containing the prompt (avoids argument length limits).')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("/scratch/sj157/Compressed_Retrieval_Attention/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--decode_type', type=str, default="sparse", choices=["dense", "sparse"], help='Decode path to use for generation.')

    args = parser.parse_args()
    main(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        interactive=args.interactive,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
        compile=args.compile,
        compile_prefill=args.compile_prefill,
        profile=args.profile,
        draft_checkpoint_path=args.draft_checkpoint_path,
        speculate_k=args.speculate_k,
        decode_type=args.decode_type,
        batch_size=args.batch_size,
    )
