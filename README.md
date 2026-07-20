# nncp

Apple Silicon (Metal / MPSGraph) port of [NNCP](https://bellard.org/nncp/) — neural network compression with online learning.

## What it does

NNCP compresses files using a Transformer that **learns during compression**. The same model (starting from identical deterministic weights) is run on both the compressor and decompressor, so no model file needs to be transferred. Every segment the model is trained on the data seen so far, adapting to the input distribution in real time.

This implementation targets the original NNCP `default` profile and runs entirely on Apple Silicon via Metal / MPSGraph.

## Architecture

| Parameter | Value |
|-----------|-------|
| d\_model | 256 |
| n\_layer | 4 |
| n\_head | 8 |
| d\_inner (FFN) | 512 |
| vocab | 256 (bytes) |
| seg\_len | 32 |
| mem\_len | 32 |
| streams | 16 (parallel) |
| Optimizer | RMSProp β₂=0.9999 |
| Learning rate | 1e-4 (linear warmup → decay) |
| Gradient clip | L2 norm 0.1 |
| LayerNorm | Post-LN (LN\_POST) |
| Positional encoding | Transformer-XL relative PE (w\_r / b\_r) |

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13.0+ (Ventura)
- Xcode Command Line Tools

## Build

```bash
mkdir -p build && cd build
cmake ..
make nncp -j$(sysctl -n hw.ncpu)
```

## Usage

```bash
# Compress
./build/nncp compress input.txt output.nncp

# Decompress
./build/nncp decompress output.nncp restored.txt

# Verify roundtrip
diff input.txt restored.txt && echo PASS
```

## Performance

Small-file benchmarks (`default` profile, online learning, deterministic seed):

| Input | Size | Compressed | Ratio |
|-------|------|------|-------|
| Dictionary text | 10 KB | ~6.3 KB | **63.5%** |
| Natural language | 10 KB | ~6.2 KB | **62.4%** |
| C source code | 50 KB | ~21.3 KB | **42.5%** |

enwik8 (`enwik8` profile — 20-layer Transformer, dictionary preprocessing + periodic retraining on recent history, matching the original NNCP recipe): sub-full-scale runs (up to 25 MB) land around **~20%**, approaching the original NNCP's published 14.9% at full 100 MB scale. A full 100 MB run has not yet been benchmarked end-to-end.

### Rough 100 MB timing estimate

Extrapolated from partial-scale measurements and public Apple Silicon GPU/memory-bandwidth specs — **not a direct 100 MB benchmark**, expect ±50% margin:

| Apple Silicon | Est. full 100 MB compress time |
|---|---|
| M2 (base) | ~7–10 days |
| M4 Max | ~3–5 days |

This workload is GPU-bound but only lightly parallel (small batch, online learning), so it doesn't scale linearly with GPU core count across chip tiers.

## How it works

```
compress input.txt output.nncp
  └─ for each byte b in input:
       1. run Transformer forward pass → probability distribution over 256 bytes
       2. arithmetic-encode b using that distribution  → compressed output
       3. train the Transformer on (context → b) via backpropagation
          (same update applied symmetrically on decompress side)
```

16 byte-streams are processed in parallel (batch\_size=16). Each stream trains independently after each segment (32 bytes), ensuring determinism across compress and decompress.

The Transformer uses Transformer-XL style KV-cache memory (mem\_len=32 + seg\_len=32 = 64 total context). Relative positional encoding (w\_r / b\_r) is learned jointly during compression.

## Key files

| File | Role |
|------|------|
| `src/neural/neural_bridge_lossless_cuda.mm` | Compress / decompress main loop |
| `src/neural/mps_transformer_graph.mm` | MPSGraph inference + KV-cache decode |
| `src/neural/online_trainer.mm` | Causal segment training graph (B=1, T=32→64) |
| `src/metal/compute/neural_net.metal` | Metal kernels (KV-cache attention, RMSProp, relative PE) |

## Original NNCP

Based on [NNCP v3 by Fabrice Bellard](https://bellard.org/nncp/) (MIT License).
Original paper: *"Low Complexity Lossless Data Compression with Neural Networks"* (Bellard, 2021).
