"""
preprocess.py — two-phase corpus preprocessing.

Phase 1 (parallel):   scan all files, collect vocab keys only (tiny return values).
Phase 2 (streaming):  single-threaded, re-parse each file, stream directly to binary.

Phase 1 uses mp.Pool with chunk_size=500. Workers return only key->fields dicts
(kilobytes, not megabytes). No sequence data crosses process boundaries.

Phase 2 never accumulates all sequences in RAM. Writes tokens.bin, dna.bin,
offsets.bin incrementally with buffered I/O.

Output in out_dir:
  tokenizer.pkl   vocabulary
  tokens.bin      flat int32, all token indices
  dna.bin         flat float32, shape (total_tokens, 7)
  offsets.bin     flat int64, shape (n_sequences, 2) = (start, length)
  manifest.json   metadata
"""

import math
import json
import struct
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

MUSICAL_DURS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
N_BEAT_BINS  = 16
_LOG_MIN     = math.log2(0.25 + 1e-4)
_LOG_RANGE   = math.log2(4.0  + 1e-4) - _LOG_MIN
DNA_FIELDS   = ['pitch_class', 'octave', 'log_duration', 'beat_sin', 'beat_cos', 'velocity', 'voice']
N_DNA        = 7

def _snap(db):
    return min(MUSICAL_DURS, key=lambda x: abs(x - db))

def _log_norm(d):
    return max(-1.0, min(1.0, (math.log2(d + 1e-4) - _LOG_MIN) / _LOG_RANGE * 2.0 - 1.0))

def _beat_bin(abs_tick, tpb):
    tpbar = tpb * 4
    return int((abs_tick % tpbar) / (tpbar / N_BEAT_BINS)) % N_BEAT_BINS

def _sincos(bb):
    a = 2.0 * math.pi * bb / N_BEAT_BINS
    return math.sin(a), math.cos(a)

# ── MIDI parser ───────────────────────────────────────────────────────────────

def _parse_midi_bytes(data: bytes):
    if data[:4] != b'MThd' or len(data) < 14:
        return None, None
    tpb = struct.unpack('>H', data[12:14])[0]
    if tpb <= 0:
        return None, None

    pos = 14; active = {}; notes = []
    while pos + 8 <= len(data):
        if data[pos:pos+4] != b'MTrk': break
        tlen = struct.unpack('>I', data[pos+4:pos+8])[0]
        pos += 8; end = pos + tlen; tp = pos; abs_tick = 0; rs = 0
        while tp < end:
            delta = 0
            while tp < end:
                b = data[tp]; tp += 1; delta = (delta << 7) | (b & 0x7f)
                if not (b & 0x80): break
            abs_tick += delta
            if tp >= end: break
            status = data[tp]
            if status & 0x80: rs = status; tp += 1
            else: status = rs
            mt = (status >> 4) & 0xf; ch = status & 0xf
            if mt == 0x9 and tp + 1 < end:
                p = data[tp]; v = data[tp+1]; tp += 2
                if v > 0: active[(ch,p)] = (abs_tick, v)
                else:
                    k = (ch,p)
                    if k in active:
                        s,vel = active.pop(k); notes.append((s,p,vel,max(abs_tick-s,1),ch))
            elif mt == 0x8 and tp + 1 < end:
                p = data[tp]; tp += 2; k = (ch,p)
                if k in active:
                    s,vel = active.pop(k); notes.append((s,p,vel,max(abs_tick-s,1),ch))
            elif mt in (0xa,0xb,0xe): tp += 2
            elif mt in (0xc,0xd): tp += 1
            elif status == 0xff:
                if tp < end: tp += 1
                if tp < end: ml = data[tp]; tp += 1; tp += ml
            elif status in (0xf0,0xf7):
                while tp < end and data[tp] != 0xf7: tp += 1
                tp += 1
        pos = end
    if active:
        last = max((n[0] for n in notes), default=0)
        for (ch,p),(s,vel) in active.items():
            notes.append((s,p,vel,max(last-s,1),ch))
    notes.sort(key=lambda n: (n[0],n[1]))
    return tpb, notes


# ── Phase 1 worker: returns ONLY vocab keys, no sequence data ─────────────────

def _vocab_chunk(paths: List[str]) -> dict:
    """Parse files, return only key->fields dict. Tiny return value."""
    key_fields = {}
    for path in paths:
        try:
            with open(path, 'rb') as f:
                data = f.read()
            tpb, notes = _parse_midi_bytes(data)
            if not notes: continue
            min_gap = tpb / 4
            for i, (at,p,v,dt,ch) in enumerate(notes):
                sd = _snap(dt/tpb); bb = _beat_bin(at, tpb); bs,bc = _sincos(bb)
                nkey = ('note', p, sd, bb)
                if nkey not in key_fields:
                    key_fields[nkey] = {
                        'pitch_class': p%12, 'octave': min(p//12,8)/8.0,
                        'log_duration': _log_norm(sd),
                        'beat_sin': bs, 'beat_cos': bc,
                        'velocity': min(v,127)/127.0, 'voice': min(ch,15)
                    }
                if i+1 < len(notes):
                    gap = notes[i+1][0] - at
                    if gap > min_gap:
                        gsd = _snap(gap/tpb); rb = _beat_bin(at+dt, tpb); rbs,rbc = _sincos(rb)
                        rkey = ('rest', None, gsd, rb)
                        if rkey not in key_fields:
                            key_fields[rkey] = {
                                'pitch_class': 0, 'octave': 0.0,
                                'log_duration': _log_norm(gsd),
                                'beat_sin': rbs, 'beat_cos': rbc,
                                'velocity': 0.0, 'voice': 0
                            }
        except Exception:
            pass
    return key_fields


# ── Phase 2: single-threaded streaming tokenize → binary ─────────────────────

def _tokenize_chunk_to_bin(args):
    """
    Worker: parse a chunk of files, write tokens+dna directly to temp files.
    Returns (tokens_tmp_path, dna_tmp_path, offsets_list, n_tokens).
    Nothing large comes back to main process.
    """
    import os, uuid
    paths, key_to_idx, pad_idx, min_seq_len, tmp_dir = args

    uid        = uuid.uuid4().hex
    tokens_tmp = os.path.join(tmp_dir, f"tok_{uid}.bin")
    dna_tmp    = os.path.join(tmp_dir, f"dna_{uid}.bin")

    BUF     = 50_000
    tok_buf = np.empty(BUF, dtype=np.int32)
    dna_buf = np.empty((BUF, N_DNA), dtype=np.float32)
    buf_pos = 0
    cursor  = 0
    offsets = []

    with open(tokens_tmp, 'wb') as ft, open(dna_tmp, 'wb') as fd:
        def flush():
            nonlocal buf_pos
            if buf_pos == 0: return
            ft.write(tok_buf[:buf_pos].tobytes())
            fd.write(dna_buf[:buf_pos].tobytes())
            buf_pos = 0

        for path in paths:
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                tpb, notes = _parse_midi_bytes(data)
                if not notes: continue

                min_gap   = tpb / 4
                seq_start = cursor
                seq_len   = 0

                for i, (at,p,v,dt,ch) in enumerate(notes):
                    sd = _snap(dt/tpb); bb = _beat_bin(at, tpb); bs,bc = _sincos(bb)
                    nkey = ('note', p, sd, bb)
                    idx  = key_to_idx.get(nkey, pad_idx)
                    if buf_pos >= BUF: flush()
                    tok_buf[buf_pos]    = idx
                    dna_buf[buf_pos, 0] = p%12
                    dna_buf[buf_pos, 1] = min(p//12,8)/8.0
                    dna_buf[buf_pos, 2] = _log_norm(sd)
                    dna_buf[buf_pos, 3] = bs
                    dna_buf[buf_pos, 4] = bc
                    dna_buf[buf_pos, 5] = min(v,127)/127.0
                    dna_buf[buf_pos, 6] = min(ch,15)
                    buf_pos += 1; cursor += 1; seq_len += 1

                    if i+1 < len(notes):
                        gap = notes[i+1][0] - at
                        if gap > min_gap:
                            gsd = _snap(gap/tpb); rb = _beat_bin(at+dt,tpb); rbs,rbc = _sincos(rb)
                            rkey = ('rest', None, gsd, rb)
                            ridx = key_to_idx.get(rkey, pad_idx)
                            if buf_pos >= BUF: flush()
                            tok_buf[buf_pos]    = ridx
                            dna_buf[buf_pos, 0] = 0;   dna_buf[buf_pos, 1] = 0.0
                            dna_buf[buf_pos, 2] = _log_norm(gsd)
                            dna_buf[buf_pos, 3] = rbs;  dna_buf[buf_pos, 4] = rbc
                            dna_buf[buf_pos, 5] = 0.0;  dna_buf[buf_pos, 6] = 0
                            buf_pos += 1; cursor += 1; seq_len += 1

                if seq_len >= min_seq_len:
                    offsets.append((seq_start, seq_len))
            except Exception:
                pass

        flush()

    return tokens_tmp, dna_tmp, offsets, cursor


# ── Public API ────────────────────────────────────────────────────────────────

def build_cache(midi_dir: str, out_dir: str, min_seq_len: int = 8,
                workers: int = None, chunk_size: int = 500,
                progress_callback=None):
    import tempfile, os, shutil
    from data.tokenizer import FMTokenizer

    midi_files = [str(p) for p in Path(midi_dir).rglob('*.mid')]
    if not midi_files:
        raise FileNotFoundError(f"No .mid files in {midi_dir}")
    if workers is None:
        workers = min(mp.cpu_count(), 16)

    chunks = [midi_files[i:i+chunk_size] for i in range(0, len(midi_files), chunk_size)]
    all_kf = {}

    # Phase 1: parallel vocab scan — tiny return values
    with mp.Pool(processes=workers) as pool:
        for kf in pool.imap_unordered(_vocab_chunk, chunks):
            for k, v in kf.items():
                if k not in all_kf:
                    all_kf[k] = v
            if progress_callback:
                progress_callback(chunk_size)

    # Build tokenizer
    tok = FMTokenizer()
    tok.idx_to_fields = [None, None, None]
    for key in sorted(all_kf, key=lambda k: (k[0], k[2] or 0.0, k[1] or 0, k[3])):
        idx = len(tok.idx_to_fields)
        tok.key_to_idx[key] = idx
        tok.idx_to_key[idx] = key
        tok.idx_to_fields.append(all_kf[key])
    tok._built = True

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = str(Path(out_dir) / '_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Phase 2: parallel tokenize → each worker writes its own temp binary
    chunk_args = [
        (chunk, tok.key_to_idx, FMTokenizer.PAD, min_seq_len, tmp_dir)
        for chunk in chunks
    ]

    results = []
    with mp.Pool(processes=workers) as pool:
        for r in pool.imap_unordered(_tokenize_chunk_to_bin, chunk_args):
            results.append(r)
            if progress_callback:
                progress_callback(chunk_size)

    # Concatenate temp files into final binaries
    tokens_path  = str(Path(out_dir) / 'tokens.bin')
    dna_path     = str(Path(out_dir) / 'dna.bin')
    offsets_path = str(Path(out_dir) / 'offsets.bin')

    all_offsets = []
    global_cursor = 0

    with open(tokens_path, 'wb') as ft, open(dna_path, 'wb') as fd:
        for tok_tmp, dna_tmp, local_offsets, n_tokens in results:
            # Adjust offsets by global cursor
            for start, length in local_offsets:
                all_offsets.append((global_cursor + start, length))
            global_cursor += n_tokens
            # Stream temp files into final
            with open(tok_tmp, 'rb') as f:
                shutil.copyfileobj(f, ft)
            with open(dna_tmp, 'rb') as f:
                shutil.copyfileobj(f, fd)
            os.unlink(tok_tmp)
            os.unlink(dna_tmp)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    off_arr = np.array(all_offsets, dtype=np.int64)
    off_arr.tofile(offsets_path)

    meta = {
        'n_sequences':  len(all_offsets),
        'total_tokens': global_cursor,
        'n_dna_fields': N_DNA,
        'dna_fields':   DNA_FIELDS,
        'tokens_dtype': 'int32',
        'dna_dtype':    'float32',
        'offsets_dtype':'int64',
    }
    with open(str(Path(out_dir) / 'manifest.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return tok, out_dir
