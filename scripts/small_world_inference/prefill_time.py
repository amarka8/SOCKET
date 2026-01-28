import torch
from transformers import AutoTokenizer
import sys
from pipeline.train_quest.modeling.modeling_llama import LlamaForCausalLM

@torch.inference_mode()
def measure_prefill_ms(model, input_ids, attention_mask=None, warmup=5, iters=20):
    """
    Measures median prefill time (ms): forward pass over prompt with use_cache=False.
    """
    model.eval()

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    # Warmup
    for _ in range(warmup):
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))  # ms

    times.sort()
    return times[len(times)//2]

device = "cuda"

# Load model
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct"
)

# Build a synthetic prompt
B = 1
T = 16384   # prompt length you want to test
input_ids = torch.randint(
    0, model.config.vocab_size,
    (B, T),
    device=device,
    dtype=torch.long,
)
attention_mask = torch.ones((B, T), device=device, dtype=torch.long)
# -------------------------
# BASELINE (no Triton)
# -------------------------
model.set_masker_mode("inference_only")
base_ms = measure_prefill_ms(
    model, input_ids, attention_mask,
    warmup=5, iters=20,
)
print(f"[baseline] prefill = {base_ms:.2f} ms")

# # -------------------------
# # TRITON PATH
# # -------------------------
# model.set_masker_mode("inference_only")  # enables Triton kernels
# fast_ms = measure_prefill_ms(
#     model, input_ids, attention_mask,
#     warmup=5, iters=20,
# )
# print(f"[triton]   prefill = {fast_ms:.2f} ms")

# print(f"speedup = {base_ms / fast_ms:.2f}x")