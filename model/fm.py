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
  note_DNA(23) → Linear(23, 4096) → * pos_encoding(pos, base=32768) → * seed_gain → cumsum → 3-layer decoder → logits

Seed conditioning:
  seed ∈ R^256 → seed_proj → sigmoid * 3.0 → gain ∈ (0,1)^4096
  gain multiplied into every token's field contribution — structurally load-bearing.
  Per-layer decoder seeds: each decoder layer gated by a separate seed projection.
  Contrastive loss: different seeds on same sequence → different fields (forced separation).
  Diversity loss: gain masks from different seeds penalized for similarity.
  Fixed seed assignment: each training sequence assigned a consistent seed across epochs.
  Seed annealing: seed std grows from 0.3→1.0 over training for coarse→fine curriculum.
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
DECODER_HIDDEN    = 2048
SEED_DIM          = 256
SEED_SIGMOID_SCALE = 3.0


# ── Position Encoding ──────────────────────────────────────────────────────────

def build_pos_encoding(seq_len: int, dim: int, base: float,
                       device, dtype) -> torch.Tensor:
    i    = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    k    = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freq = 1.0 / (base ** (k / dim))
    ang  = i * freq.unsqueeze(0)
    pe   = torch.empty(seq_len, dim, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe.to(dtype)


# ── Token DNA Encoder ──────────────────────────────────────────────────────────

class NoteDNA(nn.Module):
    def __init__(self, n_voices: int = N_VOICES, voice_dim: int = VOICE_DIM):
        super().__init__()
        self.voice_embed = nn.Embedding(n_voices, voice_dim)
        nn.init.normal_(self.voice_embed.weight, std=0.02)

    def forward(self, pitch_class, octave, log_duration,
                beat_sin, beat_cos, velocity, voice):
        pc_angle = pitch_class.float() * (2.0 * math.pi / 12.0)
        fixed = torch.stack([
            torch.sin(pc_angle), torch.cos(pc_angle),
            octave.float(), log_duration.float(),
            beat_sin.float(), beat_cos.float(),
            velocity.float(),
        ], dim=-1)
        v_emb = self.voice_embed(voice.long())
        return torch.cat([fixed, v_emb], dim=-1)


# ── Field Machine ──────────────────────────────────────────────────────────────

class FM(nn.Module):
    def __init__(self,
                 vocab_size:       int,
                 dim:              int   = DIM,
                 dna_dim:          int   = DNA_DIM,
                 n_voices:         int   = N_VOICES,
                 voice_dim:        int   = VOICE_DIM,
                 pos_base:         float = POS_BASE,
                 n_decoder_layers: int   = N_DECODER_LAYERS,
                 decoder_hidden:   int   = DECODER_HIDDEN,
                 dropout:          float = 0.1,
                 seed_dim:         int   = SEED_DIM):
        super().__init__()

        self.vocab_size       = vocab_size
        self.dim              = dim
        self.pos_base         = pos_base
        self.dna_dim          = dna_dim
        self.n_decoder_layers = n_decoder_layers
        self.seed_dim         = seed_dim

        # DNA encoder
        self.dna  = NoteDNA(n_voices, voice_dim)

        # DNA → field projection
        self.proj = nn.Sequential(
            nn.Linear(dna_dim, dim),
            nn.LayerNorm(dim),
        )

        # Field → logits decoder (standard MLP, no per-layer seed gating)
        # Seed conditioning happens at field level only — cleaner signal
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
        self.dropout     = nn.Dropout(dropout)

        # Field-level seed gain — gates field accumulation per dimension
        self.seed_proj = nn.Linear(seed_dim, dim)

        self._pe_cache        = None
        self._pe_cache_len    = 0
        self._pe_cache_device = None
        self.idx_to_fields    = None

        self._init_weights()
        # Reinitialize seed_proj with larger weights — ensures gain masks are far from
        # uniform 0.5 even early in training. Without this, near-zero projection → sigmoid
        # near 0.5 everywhere → seed has no effect.
        nn.init.normal_(self.seed_proj.weight, std=0.5)
        nn.init.zeros_(self.seed_proj.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pos(self, seq_len, device, dtype):
        if (self._pe_cache is None or seq_len > self._pe_cache_len or
                self._pe_cache.device != device):
            build_len = max(seq_len, 1024)
            self._pe_cache        = build_pos_encoding(build_len, self.dim, self.pos_base, device, dtype)
            self._pe_cache_len    = build_len
            self._pe_cache_device = device
        return self._pe_cache[:seq_len].to(dtype)

    def _seed_gain(self, seed, dtype):
        """Compute field-level gain mask from seed. (B, dim) ∈ (0,1)"""
        return torch.sigmoid(self.seed_proj(seed.to(dtype)) * SEED_SIGMOID_SCALE)

    def _decode(self, field_flat):
        """Standard decoder — no per-layer seed gating."""
        return self.decoder(field_flat)

    # ── Training forward ───────────────────────────────────────────────────────

    def forward(self,
                pitch_class:  torch.Tensor,
                octave:       torch.Tensor,
                log_duration: torch.Tensor,
                beat_sin:     torch.Tensor,
                beat_cos:     torch.Tensor,
                velocity:     torch.Tensor,
                voice:        torch.Tensor,
                seed:         torch.Tensor = None,
                return_field: bool         = False,
                ) -> torch.Tensor:
        """
        All inputs: (B, T)
        seed: (B, seed_dim)
        return_field: if True returns (logits, field_final) for contrastive loss
        Returns: (B, T, vocab_size) or ((B, T, vocab_size), (B, dim))
        """
        B, T   = pitch_class.shape
        device = pitch_class.device
        dtype  = next(self.parameters()).dtype

        # DNA → project
        dna       = self.dropout(self.dna(pitch_class, octave, log_duration,
                                          beat_sin, beat_cos, velocity, voice))
        projected = self.proj(dna)                        # (B, T, dim)

        # Position encoding
        pe        = self._pos(T, device, projected.dtype)
        modulated = projected * pe.unsqueeze(0)           # (B, T, dim)

        # Field-level seed gain — gates entire field accumulation
        if seed is not None:
            gain      = self._seed_gain(seed, dtype)      # (B, dim)
            modulated = modulated * gain.unsqueeze(1)      # (B, T, dim)

        # Cumulative field
        field = torch.cumsum(modulated, dim=1)            # (B, T, dim)

        # Decode
        logits = self._decode(field.reshape(B * T, self.dim)).reshape(B, T, self.vocab_size)

        if return_field:
            field_final = field[:, -1, :]
            return logits, field_final

        return logits

    # ── O(1) Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self,
                 prompt_fields:  dict,
                 max_new_tokens: int   = 512,
                 temperature:    float = 1.0,
                 top_k:          int   = 50,
                 top_p:          float = 0.95,
                 seed:           int   = None,
                 seed_vec:       torch.Tensor = None,
                 seed_b:         int   = None,
                 alpha:          float = None) -> list:
        """
        O(1) inference.
        seed: integer → reproducible generation
        seed_vec: raw tensor seed (overrides seed integer)
        seed_b + alpha: interpolate between seed and seed_b (lerp in seed space)
        """
        self.eval()
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        def _to(x): return x.to(device)

        pc  = _to(prompt_fields['pitch_class'])
        oc  = _to(prompt_fields['octave'])
        dur = _to(prompt_fields['log_duration'])
        bs  = _to(prompt_fields['beat_sin'])
        bc  = _to(prompt_fields['beat_cos'])
        vel = _to(prompt_fields['velocity'])
        voi = _to(prompt_fields['voice'])

        T_prompt = pc.shape[1]
        pe       = self._pos(T_prompt + max_new_tokens, device, dtype)

        # Build seed vector
        if seed_vec is not None:
            sv = seed_vec.to(device=device, dtype=dtype)
        else:
            rng = torch.Generator(device=device)
            if seed is not None:
                rng.manual_seed(seed)
            sv = torch.randn(1, self.seed_dim, generator=rng, device=device, dtype=dtype)

        # Seed interpolation
        if seed_b is not None and alpha is not None:
            rng2 = torch.Generator(device=device)
            rng2.manual_seed(seed_b)
            sv2 = torch.randn(1, self.seed_dim, generator=rng2, device=device, dtype=dtype)
            sv  = alpha * sv + (1.0 - alpha) * sv2

        gain       = self._seed_gain(sv, dtype).squeeze(0)        # (dim,)

        # Build initial field
        dna       = self.dna(pc, oc, dur, bs, bc, vel, voi)
        projected = self.proj(dna)
        modulated = projected * pe[:T_prompt].unsqueeze(0) * gain.unsqueeze(0).unsqueeze(0)
        field     = modulated.sum(dim=1).squeeze(0)                # (dim,)

        generated = []
        t = T_prompt

        for _ in range(max_new_tokens):
            logits = self._decode(field.unsqueeze(0)).squeeze(0) / max(temperature, 1e-8)

            if top_k > 0:
                kk = min(top_k, logits.shape[-1])
                top_vals, _ = torch.topk(logits, kk)
                logits[logits < top_vals[-1]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs   = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0]  = False
                sorted_logits[remove_mask] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            token_fields = self._token_to_fields(next_token, device, dtype)
            if token_fields is None:
                break

            dna_vec  = self.dna(**token_fields).squeeze(0).squeeze(0)
            proj_vec = self.proj(dna_vec.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            field    = field + proj_vec * pe[t] * gain
            t       += 1

        return generated

    def _token_to_fields(self, token_idx, device, dtype):
        if self.idx_to_fields is None: return None
        if token_idx < 0 or token_idx >= len(self.idx_to_fields): return None
        fields = self.idx_to_fields[token_idx]
        if fields is None: return None
        return {k: torch.tensor([[v]], device=device) for k, v in fields.items()}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self):
        return (f"dim={self.dim}, dna_dim={self.dna_dim}, "
                f"vocab={self.vocab_size}, pos_base={self.pos_base}, "
                f"params={self.count_parameters():,}")
