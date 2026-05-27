# FM — Field Machine

PROJECT IS A FAILURE TO LEARN FROM: STILL HAS SAME SONG PRODUCTION ISSUE, WORKING ON PATCH NOW AS NAIVE SEED APPROACH FAILED, PARTIALLY FIXED
STILL HAS SAME SONG PRODUCTION ISSUE, WORKING ON PATCH NOW AS NAIVE SEED APPROACH FAILED, PARTIALLY FIXED

A new sequence architecture invented May 26, 2026. Not a transformer. Not an RNN. Not an SSM.

---

## Core Idea

Every token is a geometrically structured object — its dimensions encoded with geometry that matches the domain's structure. These objects write into a high-dimensional field via cumulative sum. The field is decoded at every position.

**No loops. No approximation. No serial dependency during training. O(1) inference.**

The parallelization problem that plagues RNNs and SSMs is solved not by approximating recurrence or writing custom CUDA kernels — but by removing recurrence entirely. The sequence is not a chain of dependent states. It is a set of position-aware contributions to a shared field, all computable simultaneously.

---

## Architecture

```
note_DNA(23) → Linear(23, 4096) → × pos_encoding(i, base=32768) → cumsum → 3-layer decoder → logits
```

---

### Token DNA (23 dimensions)

Every note is a structured geometric vector — not a flat index into an arbitrary lookup table.

| Field         | Encoding               | Dims | Geometry                                                  |
| ------------- | ---------------------- | ---- | --------------------------------------------------------- |
| Pitch class   | sin/cos on unit circle | 2    | Circular — B is one semitone from C, not 11 apart         |
| Octave        | linear normalized      | 1    | Linear — higher is higher                                 |
| Duration      | log2 normalized        | 1    | Logarithmic — 16th vs 8th matters more than half vs whole |
| Beat position | sin/cos on unit circle | 2    | Circular — beat is periodic within bar                    |
| Velocity      | linear normalized      | 1    | Linear — louder is louder                                 |
| Voice         | learned embedding      | 16   | Categorical — soprano ≠ alto                              |

The geometry is defined before training. The model is not expected to infer these relationships from scratch; it is initialized with them explicitly embedded in representation space.

---

### Vocabulary

Vocab key: `(event_type, pitch, snapped_duration, beat_bin)` — note tokens plus REST tokens.

* **NOTE** tokens carry full DNA geometry: pitch, octave, duration, beat position, velocity, voice
* **REST** tokens carry only duration and beat position — pitch/octave/velocity/voice are zeroed. REST is not absence; it is an explicit temporal operator contributing a pitch-free vector to the field.

Beat position is quantized to 16 bins (16th-note resolution in 4/4), so the model learns when events occur, not only what they are.

Velocity remains continuous within the DNA field, not part of the vocabulary.

---

### Position Encoding

Analytic sinusoidal encoding, base = 32768. Designed to remain orthogonal up to sequence length 32768, covering all sequences in the corpus (including longest Bach Brandenburg-style pieces at ~24k notes).

```
pos[i, 2k]   = sin(i / 32768^(2k/4096))
pos[i, 2k+1] = cos(i / 32768^(2k/4096))
```

No learned parameters. No training required. Deterministic mapping.

Position is **multiplied into the token vector**, not added. Identity and position are fused prior to accumulation.

---

### Field Accumulation

```python
field[t] = cumsum(project(dna) * pos_encoding, dim=1)[t]
```

Single CUDA reduction. Entire sequence processed in parallel. No recurrence, no loop, no sequential dependency during training.

---

### Decoder

3-layer MLP with GELU activations. Projects from 4096-dimensional field to vocabulary logits.

The decoder must disentangle overlapping contributions from thousands of position-modulated token vectors. Depth is intentionally minimal to avoid bottlenecking the representation while still providing nonlinear separation capacity.

---

### O(1) Inference

```python
field_t = field_{t-1} + project(dna(token_t)) * pos_vec[t]
```

State remains a fixed 4096-dimensional vector. Memory does not grow with sequence length. No KV cache. No attention window.

---

### Field Seed (generation-time variation)

At generation time:

```python
field += randn(4096) * seed_strength
```

This perturbs initial field conditions, producing different sampling trajectories from the same prompt. It does not guarantee structural divergence of the global attractor, only local perturbation of the field state.

---

## Training Results

**Hardware:** RTX 5060 Ti (16GB VRAM)
**Batch size:** 1 (full files, no padding, no windowing)

---

### Run 1 — Bach corpus (201 files, 100 epochs, 8 min 11 sec) — 23.55M parameters

| Epoch   | Avg. Loss |
| ------- | --------- |
| 1       | 87.67     |
| 2       | 8.64      |
| 5       | 7.38      |
| 10      | 7.10      |
| 25      | 6.32      |
| 50      | 5.00      |
| 75      | 4.16      |
| **100** | **3.93**  |

Vocab: 5,299 tokens. Random baseline: log(5299) ≈ 8.57 → model drops below baseline by epoch 2.

Total tokens: 38.35M
Throughput: ~78k tok/sec

---

### Observed failure mode (critical)

Despite convergence, the model exhibits a **collapsed generative manifold**:

* Outputs converge to a single dominant musical attractor (“default amalgam song”)
* This attractor persists across:

  * different seeds
  * different temperatures
  * different checkpoints
  * different runs

This is not sampling noise. It is **representation collapse in generation space**.

The model is not learning a distribution over compositions. It is learning a single stable solution with minor stochastic perturbations.

---

### Earlier runs (archived)

Prior to current preprocessing pipeline, FM trained on the same Bach corpus using:

* 532-token vocabulary
* no REST tokens
* no beat_bin

Those runs reached loss **1.88 over 400 epochs in 15 minutes**.

These results are not directly comparable due to:

* vocabulary expansion
* structural token changes
* inclusion of REST operator and temporal encoding changes

---

## Usage

### Install

```bash
pip install torch rich
```

No external dependencies. MIDI parsing is stdlib-only.

---

### Train

```bash
python train/train.py --midi_dir /path/to/midi --out_dir checkpoints
```

First run builds binary dataset cache:

* `tokens.bin`
* `dna.bin`
* `offsets.bin`

Subsequent runs load cached tensors.

Use `--retokenize` to rebuild.

---

### Full options

```
--midi_dir        Directory of MIDI files (recursive scan)
--out_dir         Output directory for checkpoints/logs
--epochs          Number of epochs (default: 100)
--batch_size      Sequences per batch (default: 1 — full files, no padding)
--dim             Field dimension (default: 4096)
--decoder_layers  Decoder depth (default: 3)
--decoder_hidden  Decoder width (default: 2048)
--dropout         Dropout rate (default: 0.1)
--lr              Learning rate (default: 3e-4)
--weight_decay    Weight decay (default: 0.01)
--grad_clip       Gradient clipping (default: 1.0)
--warmup_steps    LR warmup steps (default: 200)
--min_seq_len     Minimum sequence length (default: 8)
--save_steps      Checkpoint interval (default: 500)
--save_minutes    Time-based checkpoint interval (default: 30)
--print_steps     Logging interval (default: 10)
--retokenize      Rebuild tokenizer and dataset cache
--fresh           Ignore existing checkpoints
```

---

### Generate

```bash
python generate/generate.py --checkpoint checkpoints/best.pt --output out.mid
```

With MIDI prompt:

```bash
python generate/generate.py --checkpoint checkpoints/best.pt \
    --prompt_midi seed.mid --tokens 512 --temperature 0.85
```

---

### Generation options

```
--tokens          Number of tokens to generate (default: 512)
--temperature     Sampling temperature (default: 0.85)
--top_k           Top-k filtering (default: 50)
--top_p           Nucleus sampling threshold (default: 0.95)
--seed_strength   Field perturbation scale (default: 0.05, recommended 0.05–0.15)
--seed            RNG seed (default: random per run)
```

---

### Benchmark

```bash
python benchmark/benchmark.py
```

---

## What Gets Saved

```
checkpoints/
  tokenizer.pkl     — vocabulary + DNA field maps
  tokens.bin        — flattened token indices
  dna.bin           — flattened DNA vectors (N × 7)
  offsets.bin       — sequence boundaries
  manifest.json     — dataset metadata
  config.json       — model config
  latest.pt         — latest checkpoint
  best.pt           — best validation loss checkpoint
  training_log.csv  — full training history
  run_stats.json    — summary statistics
```

---

## Comparison

| Property              | Transformer    | RNN/LSTM       | Mamba/SSM      | FM                    |
| --------------------- | -------------- | -------------- | -------------- | --------------------- |
| Training parallelism  | Full           | None           | Partial        | Full                  |
| Serial dependency     | None           | Yes            | Partial        | None                  |
| Inference memory      | O(n) KV cache  | O(1)           | O(1)           | O(1)                  |
| Custom CUDA needed    | No             | No             | Yes            | No                    |
| Token representation  | Flat embedding | Flat embedding | Flat embedding | Structured DNA        |
| Position encoding     | Added          | Implicit       | Implicit       | Multiplied (fused)    |
| Sequence length limit | Quadratic cost | Unlimited      | Unlimited      | Unlimited             |
| VRAM at inference     | Grows          | Fixed          | Fixed          | Fixed (single vector) |

---

## What This Is Not

FM is not a transformer with modified attention. There is no attention mechanism.

FM is not an RNN. There is no recurrent state matrix and no learned temporal feedback loop.

FM is not an SSM. There are no structured state transitions, no scan kernels, no custom CUDA recurrence.

FM is a geometric accumulation system: a sequence of position-weighted token contributions forming a shared field, decoded locally.

---

*Invented May 26, 2026. Trained on Bach.*
