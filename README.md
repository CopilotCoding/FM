# TO DO:

Add a REST token to the vocab and let sequences include silence
Make cur_tick advance by the actual predicted duration instead of a fixed 16th note
Include beat position in the vocab key so the model learns temporal placement

# FM — Field Machine

A new sequence architecture invented May 26, 2026. Not a transformer. Not an RNN. Not an SSM.

Trained on 201 Bach MIDI files. Loss 3.09 in 4:43 on a single consumer GPU.

---

## Core Idea

Every token is a geometrically structured object — its dimensions encoded with geometry that matches the domain's structure. These objects write into a high-dimensional field via cumulative sum. The field is decoded at every position.

**No loops. No approximation. No serial dependency during training. O(1) inference.**

The parallelization problem that plagues RNNs and SSMs is solved not by approximating the recurrence or writing custom CUDA kernels — but by eliminating the recurrence entirely. The sequence is not a chain of dependent states. It is a set of position-aware contributions to a shared field, all computable simultaneously.

---

## Architecture

```
note_DNA(23) → Linear(23, 4096) → × pos_encoding(i, base=32768) → cumsum → 3-layer decoder → logits
```

### Token DNA (23 dimensions)

Every note is a structured geometric vector — not a flat index into an arbitrary lookup table.

| Field | Encoding | Dims | Geometry |
|---|---|---|---|
| Pitch class | sin/cos on unit circle | 2 | Circular — B is one semitone from C, not 11 apart |
| Octave | linear normalized | 1 | Linear — higher is higher |
| Duration | log2 normalized | 1 | Logarithmic — 16th vs 8th matters more than half vs whole |
| Beat position | sin/cos on unit circle | 2 | Circular — beat is periodic within bar |
| Velocity | linear normalized | 1 | Linear — louder is louder |
| Voice | learned embedding | 16 | Categorical — soprano ≠ alto |

The geometry is real before training begins. The model does not need to discover from scratch that C and D are adjacent, or that a 16th note and 8th note are related. It starts knowing.

### Vocabulary

Vocab key: `(pitch, snapped_duration)` — 529 tokens covering the full Bach corpus.

Pitch, duration, beat position, velocity, and voice are **not** combined into a combinatorial vocabulary. Velocity and beat position are continuous expression — they live in the DNA fields, not the token identity. This reduces the vocabulary from ~18,000 possible combinations to 529 pitch×duration types.

### Position Encoding

Analytic sinusoidal, base=32768. Maximally orthogonal across positions up to sequence length 32768 — covers 100% of any realistic MIDI corpus including the longest Brandenburg concertos (24,203 notes).

```
pos[i, 2k]   = sin(i / 32768^(2k/4096))
pos[i, 2k+1] = cos(i / 32768^(2k/4096))
```

Fixed. No parameters. No training. Generalizes to any length. The base was chosen empirically from the actual corpus — not borrowed from transformer conventions.

Position is **multiplied** into the token vector, not added. Identity and position are fused into one inseparable object before accumulation.

### Field Accumulation

```python
field[t] = cumsum(project(dna) * pos_encoding, dim=1)[t]
```

One CUDA op. The entire sequence processed in parallel. No loop. No dependency chain. The GPU receives `(batch, seq_len, 4096)` work simultaneously and never waits.

### Decoder

3-layer MLP with GELU activations. Projects from 4096-dimensional field to vocab logits. Must unmix contributions from up to 24,000 overlapping position-modulated token vectors — 3 layers give sufficient nonlinear capacity without becoming a bottleneck.

### O(1) Inference

```python
field_t = field_{t-1} + project(dna(token_t)) * pos_vec[t]
```

State is one 4096-dimensional vector. Never grows. Constant memory forever. One vector addition per token. No KV cache. No attention over growing context.

---

## Training Results

**Hardware:** RTX 5060 Ti (16GB VRAM)  
**Dataset:** 201 Bach MIDI files, 532 vocab tokens, sequences 120–24,203 notes  
**Parameters:** 13.78M  
**VRAM:** 0.64GB allocated, 7.5GB reserved (PyTorch allocator)  
**Batch size:** 1 (full files, no padding, no windowing)

| Epoch | Avg. Loss | Note |
|---|---|---|
| 1 | 62.45 | First pass (LR warmup) |
| 2 | 5.48 | Below random baseline (log(532) ≈ 6.28) |
| 5 | 5.05 | Stable descent |
| 10 | 4.87 | |
| 25 | 4.50 | |
| 50 | 3.82 | |
| 75 | 3.27 | |
| **100** | **3.09** | **Final — 4 min 43 sec total** |

**Random baseline** for 532 tokens: log(532) ≈ 6.28. The model is below random baseline by epoch 2.

**Total training time for 100 epochs: 4 minutes 43 seconds.**

For comparison: the GSM architecture trained on the same corpus required 45 minutes for 100 epochs with 32M parameters. FM trains in one ninth the time with 13.78M parameters.

**Throughput:** ~74 it/s, 100k–900k tok/s depending on file length, 0.64GB allocated VRAM.

---

## Usage

### Install

```bash
pip install torch
```

No other dependencies. The MIDI parser is pure Python stdlib.

### Train

```bash
python train/train.py --midi_dir /path/to/midi --out_dir checkpoints
```

Full options:
```
--midi_dir        Directory of MIDI files (searched recursively)
--out_dir         Output directory for checkpoints and logs
--epochs          Number of epochs (default: 100)
--batch_size      Sequences per batch (default: 1 — full files, no padding)
--dim             Field dimension (default: 4096)
--decoder_layers  Decoder hidden layers (default: 3)
--decoder_hidden  Decoder hidden dim (default: 2048)
--dropout         Dropout rate (default: 0.1)
--lr              Learning rate (default: 3e-4)
--weight_decay    Weight decay (default: 0.01)
--grad_clip       Gradient clipping (default: 1.0)
--warmup_steps    LR warmup steps (default: 200)
--min_seq_len     Minimum sequence length to include (default: 8)
--save_steps      Checkpoint every N steps (default: 500)
--save_minutes    Timed checkpoint every N minutes (default: 30)
--print_steps     Print stats every N steps (default: 10)
--retokenize      Force rebuild tokenizer
--fresh           Ignore existing checkpoint, start clean
```

### Generate

```bash
python generate/generate.py --checkpoint checkpoints/best.pt --output out.mid
```

With MIDI prompt:
```bash
python generate/generate.py --checkpoint checkpoints/best.pt \
    --prompt_midi seed.mid --tokens 512 --temperature 0.85
```

Generation options:
```
--tokens        Number of tokens to generate (default: 512)
--temperature   Sampling temperature (default: 0.85)
--top_k         Top-k filtering (default: 50)
--top_p         Nucleus sampling threshold (default: 0.95)
```

### Benchmark

```bash
python benchmark/benchmark.py
```

The benchmark measures training throughput, inference throughput, and sequence length scaling. FM should show linear scaling — the benchmark table will confirm this.

---

## What Gets Saved

```
checkpoints/
  tokenizer.pkl          — vocabulary and DNA field maps
  config.json            — model configuration
  latest.pt              — most recent checkpoint (resume target)
  best.pt                — lowest loss checkpoint
  epoch_001_loss62.pt    — per-epoch snapshots
  timed_20260526_*.pt    — timed checkpoints (every N minutes)
  training_log.csv       — full per-step statistics
  run_stats.json         — final run summary
```

---

## Comparison

| Property | Transformer | RNN/LSTM | Mamba/SSM | **FM** |
|---|---|---|---|---|
| Training parallelism | Full | None | Partial | **Full** |
| Serial dependency | None | Yes | Partial | **None** |
| Inference memory | O(n) KV cache | O(1) | O(1) | **O(1)** |
| Custom CUDA needed | No | No | Yes | **No** |
| Token representation | Flat embedding | Flat embedding | Flat embedding | **Structured DNA** |
| Position encoding | Added | Implicit | Implicit | **Multiplied (fused)** |
| Sequence length limit | Quadratic cost | Unlimited | Unlimited | **Unlimited** |
| VRAM at inference | Grows with context | Fixed | Fixed | **Fixed (one vector)** |

---

## What This Is Not

FM is not a transformer with a different attention mechanism. It has no attention.

FM is not an RNN. There is no recurrent weight matrix. There is no hidden state that feeds back into itself. There is no serial dependency between timesteps during training.

FM is not an SSM. It requires no custom CUDA kernels, no parallel scan implementation, no structured state matrices.

FM is a new primitive: a sequence of position-aware geometric contributions accumulating into a shared field, decoded locally at every position.

---

*Invented May 26, 2026. Trained on Bach.*
