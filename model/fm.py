"""
FM — Field Machine
==================
A sequence architecture built on structured token DNA and cumulative field accumulation.

Core idea:
  Every token is a geometrically structured object encoding what it is AND where it is.
  These objects accumulate into a field via cumsum — one parallel CUDA op.
  The field is decoded into predictions at every position.

  No loops. No approximation. No serial dependency during training.
  O(1) inference — field is a running sum, each new token adds one vector.

Architecture:
  note_DNA(23) → Linear(23, 4096) → * pos_encoding(pos, base=32768) → cumsum → 3-layer decoder → logits

Token types:
  NOTE(pitch, dur, beat_bin) — carries full DNA geometry
  REST(dur, beat_bin)        — pitch_class/octave/velocity/voice zeroed; time passes, no pitch

Token DNA (23 dims, structured geometry):
  pitch_class : sin(2π*pc/12), cos(2π*pc/12)   — circular,  2 dims  (0 for REST)
  octave      : octave / 8                       — linear,    1 dim   (0 for REST)
  duration    : log2(dur + ε) normalized         — log,       1 dim
  beat_pos    : sin(2π*beat_bin/16), cos(...)    — circular,  2 dims  (16-bin quantized)
  velocity    : vel / 127                        — linear,    1 dim   (0 for REST)
  voice       : learned embedding                — learned,  16 dims  (0 for REST)
  ─────────────────────────────────────────────────────────────────
  total fixed  : 7 dims
  total learned: 16 dims
  total        : 23 dims

Position encoding:
  Analytic sinusoidal, base=32768.
  pos_vec[i, 2k]   = sin(i / 32768^(2k/4096))
  pos_vec[i, 2k+1] = cos(i / 32768^(2k/4096))
  Fixed. No parameters. Generalizes to any length. Maximally orthogonal up to 32768.

Field accumulation:
  field[t] = cumsum(project(dna) * pos_vec, dim=1)[t]

Inference O(1):
  field_t = field_{t-1} + project(dna(token_t)) * pos_vec[t]
  State is one 4096-dim vector. Never grows.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ──────────────────────────────────────────────────────────────────

DIM               = 4096
DNA_DIM           = 23
VOICE_DIM         = 16
N_VOICES          = 16
POS_BASE          = 32768.0
N_DECODER_LAYERS  = 3
DECODER_HIDDEN    = 2048   # 4096 * 0.5


# ── Position Encoding ──────────────────────────────────────────────────────────

def build_pos_encoding(seq_len: int, dim: int, base: float,
                       device, dtype) -> torch.Tensor:
    """
    Analytic sinusoidal position encoding matrix: (seq_len, dim).
    pos[i, 2k]   = sin(i / base^(2k/dim))
    pos[i, 2k+1] = cos(i / base^(2k/dim))

    Maximally orthogonal across positions for sequences up to length `base`.
    Fixed forever — no parameters, no training, no gradient.
    Generalizes to any sequence length.
    """
    i    = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    k    = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freq = 1.0 / (base ** (k / dim))                  # (dim/2,)
    ang  = i * freq.unsqueeze(0)                       # (seq_len, dim/2)
    pe   = torch.empty(seq_len, dim, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe.to(dtype)


# ── Token DNA Encoder ──────────────────────────────────────────────────────────

class NoteDNA(nn.Module):
    """
    Encodes a structured MIDI note event into a 23-dimensional DNA vector.

    Every dimension has geometry that matches the musical concept it encodes:
      - Pitch class is circular  → unit circle encoding
      - Octave is linear         → normalized scalar
      - Duration is logarithmic  → log2 normalized scalar
      - Beat position is circular → unit circle encoding
      - Velocity is linear       → normalized scalar
      - Voice is categorical     → small learned embedding

    This is not a lookup table. The geometry is real before training begins.
    """
    def __init__(self, n_voices: int = N_VOICES, voice_dim: int = VOICE_DIM):
        super().__init__()
        self.voice_embed = nn.Embedding(n_voices, voice_dim)
        nn.init.normal_(self.voice_embed.weight, std=0.02)

    def forward(self,
                pitch_class:   torch.Tensor,   # (B, T) int 0-11
                octave:        torch.Tensor,   # (B, T) float 0-1
                log_duration:  torch.Tensor,   # (B, T) float normalized
                beat_sin:      torch.Tensor,   # (B, T) float
                beat_cos:      torch.Tensor,   # (B, T) float
                velocity:      torch.Tensor,   # (B, T) float 0-1
                voice:         torch.Tensor,   # (B, T) int 0-15
                ) -> torch.Tensor:             # (B, T, 23)

        pc_angle = pitch_class.float() * (2.0 * math.pi / 12.0)
        pc_sin   = torch.sin(pc_angle)         # circular pitch class
        pc_cos   = torch.cos(pc_angle)

        fixed = torch.stack([
            pc_sin,
            pc_cos,
            octave.float(),
            log_duration.float(),
            beat_sin.float(),
            beat_cos.float(),
            velocity.float(),
        ], dim=-1)                             # (B, T, 7)

        v_emb = self.voice_embed(voice.long())  # (B, T, 16)
        return torch.cat([fixed, v_emb], dim=-1)  # (B, T, 23)


# ── Field Machine ──────────────────────────────────────────────────────────────

class FM(nn.Module):
    """
    Field Machine.

    Training forward (fully parallel, no loops, no approximation):
      DNA → project → * pos_encoding → cumsum → decode → logits

    Inference (O(1) per token):
      field_t = field_{t-1} + project(dna(token_t)) * pos_vec[t]
      decode(field_t) → next token logits
    """

    def __init__(self,
                 vocab_size:          int,
                 dim:                 int   = DIM,
                 dna_dim:             int   = DNA_DIM,
                 n_voices:            int   = N_VOICES,
                 voice_dim:           int   = VOICE_DIM,
                 pos_base:            float = POS_BASE,
                 n_decoder_layers:    int   = N_DECODER_LAYERS,
                 decoder_hidden:      int   = DECODER_HIDDEN,
                 dropout:             float = 0.1):
        super().__init__()

        self.vocab_size       = vocab_size
        self.dim              = dim
        self.pos_base         = pos_base
        self.dna_dim          = dna_dim
        self.n_decoder_layers = n_decoder_layers

        # DNA encoder
        self.dna  = NoteDNA(n_voices, voice_dim)

        # DNA → field projection + normalization
        self.proj = nn.Sequential(
            nn.Linear(dna_dim, dim),
            nn.LayerNorm(dim),
        )

        # Field → logits decoder (3 hidden layers, GELU)
        dec_layers = []
        in_dim = dim
        for i in range(n_decoder_layers):
            is_last = (i == n_decoder_layers - 1)
            out_dim = vocab_size if is_last else decoder_hidden
            dec_layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                dec_layers.append(nn.GELU())
                dec_layers.append(nn.Dropout(dropout))
            in_dim = decoder_hidden
        self.decoder = nn.Sequential(*dec_layers)

        self.dropout = nn.Dropout(dropout)

        # Position encoding cache — built on first forward, reused
        self._pe_cache        = None
        self._pe_cache_len    = 0
        self._pe_cache_device = None

        # Vocabulary → DNA fields map, populated by tokenizer for generation
        self.idx_to_fields = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pos(self, seq_len: int, device, dtype) -> torch.Tensor:
        """Return position encoding (seq_len, dim), building/extending cache as needed."""
        if (self._pe_cache is None or
                seq_len > self._pe_cache_len or
                self._pe_cache.device != device):
            build_len = max(seq_len, 1024)
            self._pe_cache        = build_pos_encoding(
                build_len, self.dim, self.pos_base, device, dtype)
            self._pe_cache_len    = build_len
            self._pe_cache_device = device
        return self._pe_cache[:seq_len].to(dtype)

    # ── Training forward ───────────────────────────────────────────────────────

    def forward(self,
                pitch_class:  torch.Tensor,
                octave:       torch.Tensor,
                log_duration: torch.Tensor,
                beat_sin:     torch.Tensor,
                beat_cos:     torch.Tensor,
                velocity:     torch.Tensor,
                voice:        torch.Tensor,
                ) -> torch.Tensor:
        """
        All inputs: (B, T)
        Returns:    (B, T, vocab_size)

        Fully parallel. Zero loops. Zero approximation. Zero serial dependency.
        """
        B, T   = pitch_class.shape
        device = pitch_class.device
        dtype  = next(self.parameters()).dtype

        # 1. DNA: (B, T, 23)
        dna = self.dropout(
            self.dna(pitch_class, octave, log_duration,
                     beat_sin, beat_cos, velocity, voice)
        )

        # 2. Project to field dim: (B, T, 4096)
        projected = self.proj(dna)

        # 3. Elementwise multiply by position encoding: (B, T, 4096)
        pe        = self._pos(T, device, projected.dtype)
        modulated = projected * pe.unsqueeze(0)

        # 4. Cumulative field — one CUDA op: (B, T, 4096)
        field = torch.cumsum(modulated, dim=1)

        # 5. Decode all positions in one batched pass: (B, T, vocab)
        logits = self.decoder(
            field.reshape(B * T, self.dim)
        ).reshape(B, T, self.vocab_size)

        return logits

    # ── O(1) Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self,
                 prompt_fields:  dict,
                 max_new_tokens: int   = 512,
                 temperature:    float = 1.0,
                 top_k:          int   = 50,
                 top_p:          float = 0.95) -> list:
        """
        O(1) per token inference via running field accumulation.

        prompt_fields: dict of {field_name: (1, T) tensor}
        Returns: list of generated vocab indices.

        State = one 4096-dim vector. Never grows. Memory is constant forever.
        """
        self.eval()
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        def _to(x):
            return x.to(device)

        pc  = _to(prompt_fields['pitch_class'])
        oc  = _to(prompt_fields['octave'])
        dur = _to(prompt_fields['log_duration'])
        bs  = _to(prompt_fields['beat_sin'])
        bc  = _to(prompt_fields['beat_cos'])
        vel = _to(prompt_fields['velocity'])
        voi = _to(prompt_fields['voice'])

        T_prompt = pc.shape[1]
        pe       = self._pos(T_prompt + max_new_tokens, device, dtype)

        # Build initial field from prompt
        dna       = self.dna(pc, oc, dur, bs, bc, vel, voi)   # (1, T, 23)
        projected = self.proj(dna)                              # (1, T, 4096)
        modulated = projected * pe[:T_prompt].unsqueeze(0)      # (1, T, 4096)
        field     = modulated.sum(dim=1).squeeze(0)             # (4096,) running sum

        generated = []
        t = T_prompt

        for step in range(max_new_tokens):
            # Decode current field state
            logits = self.decoder(field.unsqueeze(0)) / max(temperature, 1e-8)  # (1, vocab)

            # Top-k
            if top_k > 0:
                kk = min(top_k, logits.shape[-1])
                top_vals, _ = torch.topk(logits, kk)
                logits[logits < top_vals[:, -1:]] = float('-inf')

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs  = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            # O(1) field update
            token_fields = self._token_to_fields(next_token, device, dtype)
            if token_fields is None:
                break

            dna_vec  = self.dna(**token_fields).squeeze(0).squeeze(0)  # (23,)
            proj_vec = self.proj(dna_vec.unsqueeze(0).unsqueeze(0)
                                 ).squeeze(0).squeeze(0)                # (4096,)
            field    = field + proj_vec * pe[t]                         # O(1)
            t       += 1

        return generated

    def _token_to_fields(self, token_idx: int, device, dtype):
        """Convert vocab index to DNA field tensors for generation. Returns None on EOS/unknown."""
        if self.idx_to_fields is None:
            return None
        if token_idx < 0 or token_idx >= len(self.idx_to_fields):
            return None
        fields = self.idx_to_fields[token_idx]
        if fields is None:
            return None
        return {k: torch.tensor([[v]], device=device) for k, v in fields.items()}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, dna_dim={self.dna_dim}, "
                f"vocab={self.vocab_size}, pos_base={self.pos_base}, "
                f"params={self.count_parameters():,}")
