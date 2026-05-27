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

Vocab key: `(event_type, pitch, snapped_duration, beat_bin)` — note tokens plus REST tokens.

- **NOTE** tokens carry full DNA geometry: pitch, octave, duration, beat position, velocity, voice
- **REST** tokens carry only duration and beat position — pitch/octave/velocity/voice are zeroed. REST is not silence by absence; it is an explicit temporal operator with duration, contributing a pitch-free vector to the field.

Beat position is quantized to 16 bins (16th-note resolution within a 4/4 bar) and included in the vocab key, so the model learns *when* events happen — not just *what* they are.

Velocity remains a continuous DNA field, not a vocab dimension.

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
**Dataset:** 201 Bach MIDI files, sequences 120–24,203 notes  
**Vocab:** 5,294 tokens (note × pitch × duration × beat_bin + REST × duration × beat_bin)  
**VRAM:** up to 1.21GB allocated  
**Batch size:** 1 (full files, no padding, no windowing)

### Run 1 — original vocab (pitch × duration, 532 tokens, 100 epochs) 13.78M parameters

| Epoch | Avg. Loss | Note |
|---|---|---|
| 1 | 62.45 | First pass (LR warmup) |
| 2 | 5.48 | Below random baseline (log(532) ≈ 6.28) |
| 5 | 5.05 | |
| 10 | 4.87 | |
| 25 | 4.50 | |
| 50 | 3.82 | |
| 75 | 3.27 | |
| **100** | **3.09** | **4 min 43 sec total** |

### Run 2 — REST tokens + beat_bin vocab (400 epochs, 15 min 21 sec) 23.54M parameters

Vocab key changed to `(event_type, pitch, snapped_duration, beat_bin)`. REST tokens added. Beat position included in vocab identity at 16-bin resolution. Vocab expanded to 5,294 tokens, parameters to 23.54M.

| Epoch | Avg. Loss |
|---|---|
| 1 | 71.34 |
| 2 | 8.62 |
| 5 | 7.33 |
| 10 | 6.99 |
| 25 | 6.29 |
| 50 | 5.10 |
| 75 | 4.14 |
| 100 | 3.49 |
| 125 | 3.04 |
| 150 | 2.81 |
| 175 | 2.70 |
| 200 | 2.63 |
| 225 | 2.53 |
| 250 | 2.32 |
| 275 | 2.17 |
| 300 | 2.06 |
| 325 | 2.00 |
| 350 | 1.95 |
| 375 | 1.92 |
| **400** | **1.88** | **best: 1.876 at epoch 398** |

**Total tokens processed:** 74.74M  
**Throughput:** ~53 it/s, up to 1.7M tok/s depending on file length

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
