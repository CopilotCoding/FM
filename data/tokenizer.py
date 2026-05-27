"""
FM Tokenizer
============
Converts raw MIDI files into structured DNA token fields.

Vocab key: (pitch, snapped_duration)
  - pitch         : full MIDI pitch 0-127
  - snapped_duration: nearest musical duration from [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

Everything else — octave, pitch class, beat position, velocity, voice —
is continuous DNA, not vocab identity. These are expressive details, not
token types. Voice was mistakenly in the vocab key; it's already a DNA field.

Result: ~529 token types instead of 17978. Decoder is 100x smaller.

Each token carries 7 DNA fields for the model:
  pitch_class  : int   0-11
  octave       : float 0-1
  log_duration : float normalized log2 duration
  beat_sin     : float sin(2π * beat_position)
  beat_cos     : float cos(2π * beat_position)
  velocity     : float 0-1
  voice        : int   0-15
"""

import os
import math
import struct
import pickle
from pathlib import Path
from typing import List, Dict, Optional


# ── Musical duration grid ──────────────────────────────────────────────────────

MUSICAL_DURS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

def snap_duration(dur_beats: float) -> float:
    """Snap to nearest musical duration value."""
    return min(MUSICAL_DURS, key=lambda x: abs(x - dur_beats))

def dur_to_log_norm(dur_q: float) -> float:
    """log2(dur) normalized to [-1, 1] over [0.25, 4.0] range."""
    log_raw = math.log2(dur_q + 1e-4)
    # log2(0.25) = -2, log2(4.0) = 2
    norm = (log_raw - (-2.0)) / (2.0 - (-2.0)) * 2.0 - 1.0
    return max(-1.0, min(1.0, norm))


# ── MIDI Parser ────────────────────────────────────────────────────────────────

def parse_midi(path: str):
    """
    Parse MIDI file. Returns (ticks_per_beat, notes).
    notes: list of (abs_tick, pitch, velocity, duration_ticks, channel)
    Sorted by (abs_tick, pitch) — deterministic polyphony ordering.
    """
    with open(path, 'rb') as f:
        data = f.read()

    pos = 0
    if data[pos:pos+4] != b'MThd':
        raise ValueError(f"Not a MIDI file: {path}")
    pos += 4
    pos += 4
    fmt      = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
    n_tracks = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
    tpb      = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2

    all_raw = []

    for _ in range(n_tracks):
        if pos + 8 > len(data): break
        if data[pos:pos+4] != b'MTrk': break
        pos += 4
        tlen       = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
        track_data = data[pos:pos+tlen]; pos += tlen

        tp = 0; rs = None; abs_tick = 0

        while tp < len(track_data):
            delta = 0
            while True:
                b = track_data[tp]; tp += 1
                delta = (delta << 7) | (b & 0x7f)
                if not (b & 0x80): break
            abs_tick += delta
            if tp >= len(track_data): break

            status = track_data[tp]
            if status & 0x80: rs = status; tp += 1
            else:             status = rs
            if status is None: break

            msg_type = (status & 0xf0) >> 4
            ch       = status & 0x0f

            if msg_type == 0x9:
                if tp + 1 < len(track_data):
                    pitch = track_data[tp]; vel = track_data[tp+1]; tp += 2
                    if vel > 0: all_raw.append(('on',  abs_tick, ch, pitch, vel))
                    else:       all_raw.append(('off', abs_tick, ch, pitch, 0))
            elif msg_type == 0x8:
                if tp + 1 < len(track_data):
                    pitch = track_data[tp]; tp += 2
                    all_raw.append(('off', abs_tick, ch, pitch, 0))
            elif msg_type in [0xa, 0xb, 0xe]: tp += 2
            elif msg_type in [0xc, 0xd]:      tp += 1
            elif status == 0xff:
                meta_type = track_data[tp]; tp += 1
                meta_len  = track_data[tp]; tp += 1
                tp += meta_len
            elif status in [0xf0, 0xf7]:
                while tp < len(track_data) and track_data[tp] != 0xf7: tp += 1
                tp += 1

    # Match note-on to note-off for durations
    active = {}
    notes  = []
    for ev in sorted(all_raw, key=lambda e: e[1]):
        kind, tick, ch, pitch, vel = ev
        key = (ch, pitch)
        if kind == 'on':
            active[key] = (tick, vel)
        elif kind == 'off' and key in active:
            start, v = active.pop(key)
            dur = max(tick - start, 1)
            notes.append((start, pitch, v, dur, ch))

    if all_raw:
        last_tick = max(e[1] for e in all_raw)
        for (ch, pitch), (start, v) in active.items():
            notes.append((start, pitch, v, max(last_tick - start, 1), ch))

    notes.sort(key=lambda n: (n[0], n[1]))
    return tpb, notes


# ── Field computation ──────────────────────────────────────────────────────────

def note_to_token(pitch: int, vel: int, dur_ticks: int, channel: int,
                  abs_tick: int, tpb: int) -> dict:
    """
    Convert a raw note to:
      'key'    : (pitch, dur_snapped) — vocab identity
      'fields' : dict of 7 DNA float/int values
    """
    dur_beats   = dur_ticks / tpb
    dur_snapped = snap_duration(dur_beats)
    log_dur     = dur_to_log_norm(dur_snapped)

    pitch_class = pitch % 12
    octave      = min(pitch // 12, 8) / 8.0

    # Beat position within bar (assume 4/4, good enough for quantization)
    beat_in_bar = (abs_tick % (tpb * 4)) / tpb
    beat_frac   = beat_in_bar - int(beat_in_bar)
    # Quantize beat to nearest 16th
    beat_q      = round(beat_frac * 4) / 4 % 1.0
    beat_angle  = 2.0 * math.pi * beat_q

    return {
        'key': (pitch, dur_snapped),
        'fields': {
            'pitch_class':  pitch_class,
            'octave':       octave,
            'log_duration': log_dur,
            'beat_sin':     math.sin(beat_angle),
            'beat_cos':     math.cos(beat_angle),
            'velocity':     min(vel, 127) / 127.0,
            'voice':        min(channel, 15),
        }
    }


# ── Tokenizer ──────────────────────────────────────────────────────────────────

class FMTokenizer:
    """
    Vocab: (pitch, snapped_duration) tuples observed in corpus.
    Special: PAD=0, BOS=1, EOS=2. Note tokens start at index 3.
    idx_to_fields: exact DNA floats per vocab index, used for generation.
    """

    PAD     = 0
    BOS     = 1
    EOS     = 2
    SPECIAL = 3

    def __init__(self):
        self.key_to_idx:    Dict[tuple, int]       = {}
        self.idx_to_key:    Dict[int, tuple]       = {}
        self.idx_to_fields: List[Optional[dict]]   = []
        self._built = False

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_fields)

    def build(self, midi_dir: str, verbose: bool = True) -> 'FMTokenizer':
        files = [str(p) for p in Path(midi_dir).rglob('*.mid')]
        if not files:
            raise FileNotFoundError(f"No .mid files in {midi_dir}")

        if verbose:
            print(f"  Building vocabulary from {len(files)} MIDI files...")

        seen_keys = set()
        # Also store representative fields per key for generation
        key_fields: Dict[tuple, dict] = {}
        errors = 0

        for path in files:
            try:
                tpb, notes = parse_midi(path)
                for (abs_tick, pitch, vel, dur_ticks, ch) in notes:
                    t = note_to_token(pitch, vel, dur_ticks, ch, abs_tick, tpb)
                    k = t['key']
                    seen_keys.add(k)
                    if k not in key_fields:
                        key_fields[k] = t['fields']
            except Exception:
                errors += 1

        # Build vocab
        self.idx_to_fields = [None, None, None]  # PAD, BOS, EOS
        for key in sorted(seen_keys):
            idx = len(self.idx_to_fields)
            self.key_to_idx[key] = idx
            self.idx_to_key[idx] = key
            self.idx_to_fields.append(key_fields[key])

        self._built = True

        if verbose:
            print(f"  Vocabulary: {self.vocab_size} tokens "
                  f"({self.vocab_size - self.SPECIAL} note types + 3 special)")
            if errors:
                print(f"  Warning: {errors} files failed to parse")
        return self

    def tokenize_file(self, path: str) -> Optional[Dict[str, list]]:
        assert self._built, "Call build() first"
        try:
            tpb, notes = parse_midi(path)
        except Exception:
            return None
        if not notes:
            return None

        out = {k: [] for k in [
            'pitch_class', 'octave', 'log_duration',
            'beat_sin', 'beat_cos', 'velocity', 'voice', 'token_idx'
        ]}

        for (abs_tick, pitch, vel, dur_ticks, ch) in notes:
            t   = note_to_token(pitch, vel, dur_ticks, ch, abs_tick, tpb)
            idx = self.key_to_idx.get(t['key'], self.PAD)
            f   = t['fields']
            out['pitch_class'].append(f['pitch_class'])
            out['octave'].append(f['octave'])
            out['log_duration'].append(f['log_duration'])
            out['beat_sin'].append(f['beat_sin'])
            out['beat_cos'].append(f['beat_cos'])
            out['velocity'].append(f['velocity'])
            out['voice'].append(f['voice'])
            out['token_idx'].append(idx)

        return out

    def tokenize_corpus(self, midi_dir: str,
                        verbose: bool = True) -> List[Dict[str, list]]:
        files    = [str(p) for p in Path(midi_dir).rglob('*.mid')]
        seqs     = []
        errors   = 0
        for i, path in enumerate(files):
            r = self.tokenize_file(path)
            if r is not None: seqs.append(r)
            else: errors += 1
            if verbose and (i + 1) % 50 == 0:
                print(f"  Tokenized {i+1}/{len(files)} files...")
        if verbose:
            lengths = [len(s['token_idx']) for s in seqs]
            print(f"  Tokenized {len(seqs)} files ({errors} errors)")
            print(f"  Lengths: min={min(lengths)} max={max(lengths)} "
                  f"mean={sum(lengths)//len(lengths)}")
        return seqs

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'key_to_idx':    self.key_to_idx,
                'idx_to_key':    self.idx_to_key,
                'idx_to_fields': self.idx_to_fields,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'FMTokenizer':
        tok = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tok.key_to_idx    = data['key_to_idx']
        tok.idx_to_key    = data['idx_to_key']
        tok.idx_to_fields = data['idx_to_fields']
        tok._built        = True
        return tok

    def stats(self) -> dict:
        return {
            'vocab_size':  self.vocab_size,
            'note_types':  self.vocab_size - self.SPECIAL,
        }
