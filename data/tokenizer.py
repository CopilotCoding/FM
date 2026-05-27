"""
FM Tokenizer
============
Converts raw MIDI files into structured DNA token fields.

Vocab key: (event_type, pitch_or_none, snapped_duration, beat_bin)
  event_type      : 'note' or 'rest'
  pitch           : MIDI pitch 0-127 (note tokens only, None for rest)
  snapped_duration: nearest value from MUSICAL_DURS
  beat_bin        : quantized beat position, 0-15 (16 bins per bar)

REST tokens carry only duration and beat_bin. Pitch/octave/pitch_class
fields are zeroed.

DNA fields (unchanged dimensionality — 23 dims):
  pitch_class  : int   0-11  (0 for REST)
  octave       : float 0-1   (0 for REST)
  log_duration : float normalized log2 duration
  beat_sin     : float sin(2π * beat_bin/16)
  beat_cos     : float cos(2π * beat_bin/16)
  velocity     : float 0-1   (0 for REST)
  voice        : int   0-15  (0 for REST)
"""

import os
import math
import struct
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ── Musical duration grid ──────────────────────────────────────────────────────

MUSICAL_DURS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
N_BEAT_BINS  = 16

def snap_duration(dur_beats: float) -> float:
    return min(MUSICAL_DURS, key=lambda x: abs(x - dur_beats))

def dur_to_log_norm(dur_q: float) -> float:
    log_raw = math.log2(dur_q + 1e-4)
    norm = (log_raw - (-2.0)) / (2.0 - (-2.0)) * 2.0 - 1.0
    return max(-1.0, min(1.0, norm))

def quantize_beat(abs_tick: int, tpb: int) -> int:
    """Return beat bin 0-15 (16 bins per 4/4 bar = 16th-note resolution)."""
    ticks_per_bar  = tpb * 4
    tick_in_bar    = abs_tick % ticks_per_bar
    bin_size       = ticks_per_bar / N_BEAT_BINS
    return int(tick_in_bar / bin_size) % N_BEAT_BINS

def beat_bin_to_sincos(beat_bin: int) -> Tuple[float, float]:
    angle = 2.0 * math.pi * beat_bin / N_BEAT_BINS
    return math.sin(angle), math.cos(angle)


# ── MIDI Parser ────────────────────────────────────────────────────────────────

def parse_midi(path: str):
    """
    Parse MIDI file. Returns (ticks_per_beat, notes).
    notes: list of (abs_tick, pitch, velocity, duration_ticks, channel)
    Sorted by (abs_tick, pitch).
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
    dur_beats   = dur_ticks / tpb
    dur_snapped = snap_duration(dur_beats)
    log_dur     = dur_to_log_norm(dur_snapped)
    beat_bin    = quantize_beat(abs_tick, tpb)
    bs, bc      = beat_bin_to_sincos(beat_bin)
    pitch_class = pitch % 12
    octave      = min(pitch // 12, 8) / 8.0

    return {
        'key': ('note', pitch, dur_snapped, beat_bin),
        'fields': {
            'pitch_class':  pitch_class,
            'octave':       octave,
            'log_duration': log_dur,
            'beat_sin':     bs,
            'beat_cos':     bc,
            'velocity':     min(vel, 127) / 127.0,
            'voice':        min(channel, 15),
        }
    }

def rest_to_token(dur_beats: float, abs_tick: int, tpb: int) -> dict:
    dur_snapped = snap_duration(dur_beats)
    log_dur     = dur_to_log_norm(dur_snapped)
    beat_bin    = quantize_beat(abs_tick, tpb)
    bs, bc      = beat_bin_to_sincos(beat_bin)

    return {
        'key': ('rest', None, dur_snapped, beat_bin),
        'fields': {
            'pitch_class':  0,
            'octave':       0.0,
            'log_duration': log_dur,
            'beat_sin':     bs,
            'beat_cos':     bc,
            'velocity':     0.0,
            'voice':        0,
        }
    }


def extract_rests(notes: list, tpb: int) -> list:
    """
    Interleave REST tokens into a sorted note list wherever there is a gap
    between consecutive note onsets greater than one 16th note (tpb/4 ticks).
    Returns a new list of dicts: {'type': 'note'|'rest', 'abs_tick': int, ...}
    """
    if not notes:
        return []

    min_gap = tpb / 4  # 16th note minimum gap to insert a rest

    result = []
    for i, n in enumerate(notes):
        abs_tick, pitch, vel, dur_ticks, ch = n
        result.append({'type': 'note', 'abs_tick': abs_tick,
                       'pitch': pitch, 'vel': vel,
                       'dur_ticks': dur_ticks, 'ch': ch})

        if i + 1 < len(notes):
            next_tick = notes[i + 1][0]
            gap_ticks = next_tick - abs_tick
            if gap_ticks > min_gap:
                gap_beats = gap_ticks / tpb
                gap_snapped = snap_duration(gap_beats)
                result.append({'type': 'rest', 'abs_tick': abs_tick + dur_ticks,
                               'dur_beats': gap_snapped})

    return result


def _single_pass_chunk(paths: list) -> tuple:
    """
    Single pass: build vocab keys AND tokenize sequences simultaneously.
    Returns (key_fields dict, sequences list) for this chunk.
    """
    import numpy as np
    key_fields = {}
    sequences  = []

    for path in paths:
        try:
            tpb, notes = parse_midi(path)
        except Exception:
            continue
        if not notes:
            continue

        notes_arr = np.array(notes, dtype=np.int64)
        abs_ticks  = notes_arr[:, 0]
        pitches    = notes_arr[:, 1]
        vels       = notes_arr[:, 2]
        dur_ticks  = notes_arr[:, 3]
        channels   = notes_arr[:, 4]

        dur_beats    = dur_ticks / tpb
        dur_arr      = np.array(MUSICAL_DURS)
        snapped_idx  = np.argmin(np.abs(dur_beats[:, None] - dur_arr[None, :]), axis=1)
        snapped_durs = dur_arr[snapped_idx]

        log_durs = np.log2(snapped_durs + 1e-4)
        log_durs = np.clip((log_durs - (-2.0)) / (2.0 - (-2.0)) * 2.0 - 1.0, -1.0, 1.0)

        ticks_per_bar = tpb * 4
        tick_in_bar   = abs_ticks % ticks_per_bar
        bin_size      = ticks_per_bar / N_BEAT_BINS
        beat_bins     = (tick_in_bar / bin_size).astype(np.int64) % N_BEAT_BINS
        angles        = 2.0 * np.pi * beat_bins / N_BEAT_BINS
        beat_sins     = np.sin(angles)
        beat_coss     = np.cos(angles)

        pitch_classes = pitches % 12
        octaves       = np.clip(pitches // 12, 0, 8) / 8.0
        velocities    = np.clip(vels, 0, 127) / 127.0
        voices        = np.clip(channels, 0, 15)

        out_pc, out_oct, out_ld, out_bs, out_bc = [], [], [], [], []
        out_vel, out_voice, out_keys = [], [], []

        for i in range(len(notes)):
            nkey = ('note', int(pitches[i]), float(snapped_durs[i]), int(beat_bins[i]))
            if nkey not in key_fields:
                key_fields[nkey] = {
                    'pitch_class':  int(pitch_classes[i]),
                    'octave':       float(octaves[i]),
                    'log_duration': float(log_durs[i]),
                    'beat_sin':     float(beat_sins[i]),
                    'beat_cos':     float(beat_coss[i]),
                    'velocity':     float(velocities[i]),
                    'voice':        int(voices[i]),
                }
            out_pc.append(int(pitch_classes[i]))
            out_oct.append(float(octaves[i]))
            out_ld.append(float(log_durs[i]))
            out_bs.append(float(beat_sins[i]))
            out_bc.append(float(beat_coss[i]))
            out_vel.append(float(velocities[i]))
            out_voice.append(int(voices[i]))
            out_keys.append(nkey)

            # Rest between notes
            if i + 1 < len(notes):
                gap = int(notes[i+1][0]) - int(notes[i][0])
                if gap > tpb / 4:
                    gb = gap / tpb
                    gs = min(MUSICAL_DURS, key=lambda x: abs(x - gb))
                    gl = float(np.clip((np.log2(gs + 1e-4) - (-2.0)) / (2.0 - (-2.0)) * 2.0 - 1.0, -1.0, 1.0))
                    rt = int(notes[i][0]) + int(notes[i][3])
                    rb = int((rt % ticks_per_bar) / bin_size) % N_BEAT_BINS
                    ra = 2.0 * np.pi * rb / N_BEAT_BINS
                    rbs, rbc = float(np.sin(ra)), float(np.cos(ra))
                    rkey = ('rest', None, gs, rb)
                    if rkey not in key_fields:
                        key_fields[rkey] = {
                            'pitch_class':  0,
                            'octave':       0.0,
                            'log_duration': gl,
                            'beat_sin':     rbs,
                            'beat_cos':     rbc,
                            'velocity':     0.0,
                            'voice':        0,
                        }
                    out_pc.append(0); out_oct.append(0.0); out_ld.append(gl)
                    out_bs.append(rbs); out_bc.append(rbc)
                    out_vel.append(0.0); out_voice.append(0)
                    out_keys.append(rkey)

        if out_keys:
            sequences.append({
                'pitch_class':  out_pc,
                'octave':       out_oct,
                'log_duration': out_ld,
                'beat_sin':     out_bs,
                'beat_cos':     out_bc,
                'velocity':     out_vel,
                'voice':        out_voice,
                'keys':         out_keys,   # resolved to indices after vocab is built
            })

    return key_fields, sequences
    """Parse a chunk of MIDI files, return combined key->fields dict."""
    result = {}
    for path in paths:
        try:
            tpb, notes = parse_midi(path)
            events = extract_rests(notes, tpb)
            for ev in events:
                if ev['type'] == 'note':
                    t = note_to_token(ev['pitch'], ev['vel'],
                                      ev['dur_ticks'], ev['ch'],
                                      ev['abs_tick'], tpb)
                else:
                    t = rest_to_token(ev['dur_beats'], ev['abs_tick'], tpb)
                k = t['key']
                if k not in result:
                    result[k] = t['fields']
        except Exception:
            pass
    return result


def _tokenize_file_chunk(args) -> list:
    """Tokenize a chunk of files. Returns list of (result_or_None) per file."""
    paths, key_to_idx, pad_idx = args
    out = []
    for path in paths:
        try:
            tpb, notes = parse_midi(path)
            events = extract_rests(notes, tpb)
            seq = {k: [] for k in [
                'pitch_class', 'octave', 'log_duration',
                'beat_sin', 'beat_cos', 'velocity', 'voice', 'token_idx'
            ]}
            for ev in events:
                if ev['type'] == 'note':
                    t = note_to_token(ev['pitch'], ev['vel'],
                                      ev['dur_ticks'], ev['ch'],
                                      ev['abs_tick'], tpb)
                else:
                    t = rest_to_token(ev['dur_beats'], ev['abs_tick'], tpb)
                idx = key_to_idx.get(t['key'], pad_idx)
                f   = t['fields']
                seq['pitch_class'].append(f['pitch_class'])
                seq['octave'].append(f['octave'])
                seq['log_duration'].append(f['log_duration'])
                seq['beat_sin'].append(f['beat_sin'])
                seq['beat_cos'].append(f['beat_cos'])
                seq['velocity'].append(f['velocity'])
                seq['voice'].append(f['voice'])
                seq['token_idx'].append(idx)
            out.append(seq if seq['token_idx'] else None)
        except Exception:
            out.append(None)
    return out


# keep old names as aliases so existing imports don't break
def tokenize_file_vectorized(path: str, key_to_idx: dict, pad_idx: int) -> Optional[Dict[str, list]]:
    """
    Vectorized tokenization — processes all notes as numpy arrays.
    5-10x faster than per-note Python loops.
    """
    import numpy as np
    try:
        tpb, notes = parse_midi(path)
    except Exception:
        return None
    if not notes:
        return None

    notes_arr = np.array(notes, dtype=np.int64)  # (N, 5): abs_tick, pitch, vel, dur_ticks, ch
    abs_ticks  = notes_arr[:, 0]
    pitches    = notes_arr[:, 1]
    vels       = notes_arr[:, 2]
    dur_ticks  = notes_arr[:, 3]
    channels   = notes_arr[:, 4]

    # Duration in beats → snap to nearest MUSICAL_DUR
    dur_beats = dur_ticks / tpb
    dur_arr   = np.array(MUSICAL_DURS)
    # argmin of abs difference for each note
    snapped_idx  = np.argmin(np.abs(dur_beats[:, None] - dur_arr[None, :]), axis=1)
    snapped_durs = dur_arr[snapped_idx]

    # Log duration normalized
    log_durs = np.log2(snapped_durs + 1e-4)
    log_durs = (log_durs - (-2.0)) / (2.0 - (-2.0)) * 2.0 - 1.0
    log_durs = np.clip(log_durs, -1.0, 1.0)

    # Beat bins
    ticks_per_bar = tpb * 4
    tick_in_bar   = abs_ticks % ticks_per_bar
    bin_size      = ticks_per_bar / N_BEAT_BINS
    beat_bins     = (tick_in_bar / bin_size).astype(np.int64) % N_BEAT_BINS
    angles        = 2.0 * np.pi * beat_bins / N_BEAT_BINS
    beat_sins     = np.sin(angles)
    beat_coss     = np.cos(angles)

    # Pitch fields
    pitch_classes = pitches % 12
    octaves       = np.clip(pitches // 12, 0, 8) / 8.0
    velocities    = np.clip(vels, 0, 127) / 127.0
    voices        = np.clip(channels, 0, 15)

    # Build token indices
    token_indices = []
    for i in range(len(notes)):
        key = ('note', int(pitches[i]), float(snapped_durs[i]), int(beat_bins[i]))
        token_indices.append(key_to_idx.get(key, pad_idx))

    # Insert rests between notes where gap > 16th note
    min_gap = tpb / 4
    out_pc, out_oct, out_ld, out_bs, out_bc, out_vel, out_voice, out_idx = [], [], [], [], [], [], [], []

    for i in range(len(notes)):
        out_pc.append(int(pitch_classes[i]))
        out_oct.append(float(octaves[i]))
        out_ld.append(float(log_durs[i]))
        out_bs.append(float(beat_sins[i]))
        out_bc.append(float(beat_coss[i]))
        out_vel.append(float(velocities[i]))
        out_voice.append(int(voices[i]))
        out_idx.append(token_indices[i])

        if i + 1 < len(notes):
            gap = notes[i+1][0] - notes[i][0]
            if gap > min_gap:
                gap_beats   = gap / tpb
                gap_snapped = min(MUSICAL_DURS, key=lambda x: abs(x - gap_beats))
                gap_log     = float(np.clip((np.log2(gap_snapped + 1e-4) - (-2.0)) / (2.0 - (-2.0)) * 2.0 - 1.0, -1.0, 1.0))
                rest_tick   = notes[i][0] + notes[i][3]
                rb          = int((rest_tick % ticks_per_bar) / bin_size) % N_BEAT_BINS
                rbs, rbc    = float(np.sin(2 * np.pi * rb / N_BEAT_BINS)), float(np.cos(2 * np.pi * rb / N_BEAT_BINS))
                rkey        = ('rest', None, gap_snapped, rb)
                out_pc.append(0); out_oct.append(0.0); out_ld.append(gap_log)
                out_bs.append(rbs); out_bc.append(rbc)
                out_vel.append(0.0); out_voice.append(0)
                out_idx.append(key_to_idx.get(rkey, pad_idx))

    if not out_idx:
        return None

    return {
        'pitch_class':  out_pc,
        'octave':       out_oct,
        'log_duration': out_ld,
        'beat_sin':     out_bs,
        'beat_cos':     out_bc,
        'velocity':     out_vel,
        'voice':        out_voice,
        'token_idx':    out_idx,
    }

def _tokenize_one_file(args):
    path, key_to_idx, pad_idx = args
    results = _tokenize_file_chunk(([path], key_to_idx, pad_idx))
    return results[0] if results else None


# ── Tokenizer ──────────────────────────────────────────────────────────────────

class FMTokenizer:
    """
    Vocab: (event_type, pitch_or_none, snapped_duration, beat_bin) tuples.
    Special: PAD=0, BOS=1, EOS=2. Tokens start at index 3.
    REST tokens have pitch_or_none=None, pitch_class/octave/velocity/voice=0.
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

    def _build_with_progress_parallel(self, midi_dir: str, files: list,
                                       progress, task, workers: int = 16) -> 'FMTokenizer':
        from concurrent.futures import ThreadPoolExecutor, as_completed

        seen_keys = set()
        key_fields: Dict[tuple, dict] = {}

        def parse_one(path):
            try:
                tpb, notes = parse_midi(path)
                events = extract_rests(notes, tpb)
                result = {}
                for ev in events:
                    if ev['type'] == 'note':
                        t = note_to_token(ev['pitch'], ev['vel'],
                                          ev['dur_ticks'], ev['ch'],
                                          ev['abs_tick'], tpb)
                    else:
                        t = rest_to_token(ev['dur_beats'], ev['abs_tick'], tpb)
                    k = t['key']
                    if k not in result:
                        result[k] = t['fields']
                return result
            except Exception:
                return {}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(parse_one, f): f for f in files}
            for fut in as_completed(futures):
                r = fut.result()
                for k, f in r.items():
                    seen_keys.add(k)
                    if k not in key_fields:
                        key_fields[k] = f
                progress.advance(task)

        self.idx_to_fields = [None, None, None]
        for key in sorted(seen_keys, key=lambda k: (k[0], k[2] or 0.0, k[1] or 0, k[3])):
            idx = len(self.idx_to_fields)
            self.key_to_idx[key] = idx
            self.idx_to_key[idx] = key
            self.idx_to_fields.append(key_fields[key])
        self._built = True
        return self

    def _build_with_progress(self, midi_dir: str, progress, task) -> 'FMTokenizer':
        """Build vocabulary with an external Rich progress task."""
        files = [str(p) for p in Path(midi_dir).rglob('*.mid')]
        seen_keys = set()
        key_fields: Dict[tuple, dict] = {}
        errors = 0
        for path in files:
            try:
                tpb, notes = parse_midi(path)
                events = extract_rests(notes, tpb)
                for ev in events:
                    if ev['type'] == 'note':
                        t = note_to_token(ev['pitch'], ev['vel'],
                                          ev['dur_ticks'], ev['ch'],
                                          ev['abs_tick'], tpb)
                    else:
                        t = rest_to_token(ev['dur_beats'], ev['abs_tick'], tpb)
                    k = t['key']
                    seen_keys.add(k)
                    if k not in key_fields:
                        key_fields[k] = t['fields']
            except Exception:
                errors += 1
            progress.advance(task)

        self.idx_to_fields = [None, None, None]
        for key in sorted(seen_keys, key=lambda k: (k[0], k[2] or 0.0, k[1] or 0, k[3])):
            idx = len(self.idx_to_fields)
            self.key_to_idx[key] = idx
            self.idx_to_key[idx] = key
            self.idx_to_fields.append(key_fields[key])
        self._built = True
        return self

    def build(self, midi_dir: str, verbose: bool = True) -> 'FMTokenizer':
        files = [str(p) for p in Path(midi_dir).rglob('*.mid')]
        if not files:
            raise FileNotFoundError(f"No .mid files in {midi_dir}")

        if verbose:
            print(f"  Building vocabulary from {len(files)} MIDI files...")

        seen_keys = set()
        key_fields: Dict[tuple, dict] = {}
        errors = 0

        for path in files:
            try:
                tpb, notes = parse_midi(path)
                events = extract_rests(notes, tpb)
                for ev in events:
                    if ev['type'] == 'note':
                        t = note_to_token(ev['pitch'], ev['vel'],
                                          ev['dur_ticks'], ev['ch'],
                                          ev['abs_tick'], tpb)
                    else:
                        t = rest_to_token(ev['dur_beats'], ev['abs_tick'], tpb)
                    k = t['key']
                    seen_keys.add(k)
                    if k not in key_fields:
                        key_fields[k] = t['fields']
            except Exception:
                errors += 1

        self.idx_to_fields = [None, None, None]  # PAD, BOS, EOS
        for key in sorted(seen_keys, key=lambda k: (k[0], k[2] or 0.0, k[1] or 0, k[3])):
            idx = len(self.idx_to_fields)
            self.key_to_idx[key] = idx
            self.idx_to_key[idx] = key
            self.idx_to_fields.append(key_fields[key])

        self._built = True

        n_rest = sum(1 for k in seen_keys if k[0] == 'rest')
        n_note = sum(1 for k in seen_keys if k[0] == 'note')

        if verbose:
            print(f"  Vocabulary: {self.vocab_size} tokens "
                  f"({n_note} note types + {n_rest} rest types + 3 special)")
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

        events = extract_rests(notes, tpb)

        out = {k: [] for k in [
            'pitch_class', 'octave', 'log_duration',
            'beat_sin', 'beat_cos', 'velocity', 'voice', 'token_idx'
        ]}

        for ev in events:
            if ev['type'] == 'note':
                t = note_to_token(ev['pitch'], ev['vel'],
                                  ev['dur_ticks'], ev['ch'],
                                  ev['abs_tick'], tpb)
            else:
                t = rest_to_token(ev['dur_beats'], ev['abs_tick'], tpb)

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
        files  = [str(p) for p in Path(midi_dir).rglob('*.mid')]
        seqs   = []
        errors = 0
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

    def is_rest(self, idx: int) -> bool:
        key = self.idx_to_key.get(idx)
        return key is not None and key[0] == 'rest'

    def get_duration_beats(self, idx: int) -> float:
        """Return snapped duration in beats for a given token index."""
        key = self.idx_to_key.get(idx)
        if key is None:
            return 0.25
        return key[2]  # snapped_duration

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
        n_rest = sum(1 for k in self.idx_to_key.values() if k[0] == 'rest')
        n_note = sum(1 for k in self.idx_to_key.values() if k[0] == 'note')
        return {
            'vocab_size': self.vocab_size,
            'note_types': n_note,
            'rest_types': n_rest,
        }
