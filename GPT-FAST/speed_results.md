# SOCKET vs FA2 vs FA3 — decode-throughput speed results

1-layer Llama-3.1-8B, H200, bf16, `torch.compile(fullgraph + reduce-overhead + CUDA graphs)`,
GPT-FAST throughput path, decode-only, warmup-excluded (`SOCKET_DECODE_WARMUP=8`),
`num_samples=5` (median of warm decode-only samples), `max_new_tokens=50`, `batch_size=1`.

SOCKET: rank-1 + per-kv-head scorer + heavy-base dedup; `sink=window=120` (`sink+window=240`),
`SOCKET_L=60`. Heavy budget `HEAVY = round(N/ratio) - 240`; realized prompt_len = N+1 so
realized sparsity hits the target exactly. K=8 uses `SOCKET_R=256`, K=10 uses `SOCKET_R=1024`
(R is structurally pinned to 2**K).

FA2 and FA3 are **dense** baselines: their decode cost is independent of the SOCKET K / sparsity /
heavy-budget config, so the **same FA2/FA3 numbers are reused across all four tables** (measured
once in the K=10 / 33.3x run; FA2 = `SOCKET_FORCE_FA2=1` + `SOCKET_REQUIRE_BACKEND=fa2`,
FA3 = default + `SOCKET_REQUIRE_BACKEND=fa3`, backend-asserted on each).

All tok/s are the median of the 5 warm decode-only samples. Higher is better.

> SOCKET is largely K- and sparsity-insensitive at decode time — its cost is dominated by the
> soft-hash scorer plus fixed per-step overhead rather than the heavy budget — so the four
> tables below are close to one another. In contrast, the SOCKET-over-dense lead grows with
> context: near parity at 18K, ~1.2x at 100K, and ~1.3–1.5x at 140K, because the dense FA2/FA3
> cost scales with the full KV length while SOCKET attends to a fixed heavy budget.

## 1. K=10, L=60, 33.3x

| ctx | SOCKET | FA2 | FA3 | SOCKET/FA2 | SOCKET/FA3 |
|----:|-------:|----:|----:|-----------:|-----------:|
| 18K | 1756.1 | 1715.1 | 1791.7 | 1.02x | 0.98x |
| 36K | 1615.2 | 1534.4 | 1561.5 | 1.05x | 1.03x |
| 72K | 1508.2 | 1286.2 | 1343.3 | 1.17x | 1.12x |
| 100K | 1450.2 | 1211.5 | 1225.7 | 1.20x | 1.18x |
| 140K | 1383.0 | 953.3 | 1067.5 | 1.45x | 1.30x |

## 2. K=10, L=60, 40x

| ctx | SOCKET | FA2 | FA3 | SOCKET/FA2 | SOCKET/FA3 |
|----:|-------:|----:|----:|-----------:|-----------:|
| 18K | 1752.0 | 1715.1 | 1791.7 | 1.02x | 0.98x |
| 36K | 1617.3 | 1534.4 | 1561.5 | 1.05x | 1.04x |
| 72K | 1524.2 | 1286.2 | 1343.3 | 1.19x | 1.13x |
| 100K | 1460.0 | 1211.5 | 1225.7 | 1.21x | 1.19x |
| 140K | 1397.9 | 953.3 | 1067.5 | 1.47x | 1.31x |

## 3. K=8, L=60, 40x

| ctx | SOCKET | FA2 | FA3 | SOCKET/FA2 | SOCKET/FA3 |
|----:|-------:|----:|----:|-----------:|-----------:|
| 18K | 1735.2 | 1715.1 | 1791.7 | 1.01x | 0.97x |
| 36K | 1641.0 | 1534.4 | 1561.5 | 1.07x | 1.05x |
| 72K | 1564.6 | 1286.2 | 1343.3 | 1.22x | 1.16x |
| 100K | 1473.4 | 1211.5 | 1225.7 | 1.22x | 1.20x |
| 140K | 1350.3 | 953.3 | 1067.5 | 1.42x | 1.26x |

## 4. K=8, L=60, 33.3x

| ctx | SOCKET | FA2 | FA3 | SOCKET/FA2 | SOCKET/FA3 |
|----:|-------:|----:|----:|-----------:|-----------:|
| 18K | 1777.0 | 1715.1 | 1791.7 | 1.04x | 0.99x |
| 36K | 1641.4 | 1534.4 | 1561.5 | 1.07x | 1.05x |
| 72K | 1547.0 | 1286.2 | 1343.3 | 1.20x | 1.15x |
| 100K | 1472.1 | 1211.5 | 1225.7 | 1.22x | 1.20x |
| 140K | 1377.1 | 953.3 | 1067.5 | 1.44x | 1.29x |
