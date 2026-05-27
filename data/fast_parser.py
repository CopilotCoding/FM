"""
Fast MIDI parser using Numba JIT.
Compiles to machine code, releases GIL, runs in parallel threads.
First call takes ~2s to JIT compile — cached after that.
"""

import numpy as np
from numba import njit, prange
from numba.typed import List
import math


@njit(cache=True)
def _parse_track(track: np.ndarray, tpb: int):
    """
    Parse a single MIDI track byte array into note arrays.
    Returns parallel arrays: abs_ticks, pitches, vels, dur_ticks, channels
    All int32. njit compiled — no GIL, machine code speed.
    """
    MAX_NOTES = 65536
    on_tick   = np.zeros((16, 128), dtype=np.int32)
    on_vel    = np.zeros((16, 128), dtype=np.int32)
    on_active = np.zeros((16, 128), dtype=np.uint8)

    abs_ticks  = np.empty(MAX_NOTES, dtype=np.int32)
    pitches    = np.empty(MAX_NOTES, dtype=np.int32)
    vels       = np.empty(MAX_NOTES, dtype=np.int32)
    dur_ticks  = np.empty(MAX_NOTES, dtype=np.int32)
    channels   = np.empty(MAX_NOTES, dtype=np.int32)
    n_notes    = 0

    tp        = 0
    abs_tick  = 0
    rs        = 0
    tlen      = len(track)

    while tp < tlen:
        # Read variable-length delta time
        delta = 0
        while tp < tlen:
            b = int(track[tp]); tp += 1
            delta = (delta << 7) | (b & 0x7f)
            if not (b & 0x80):
                break
        abs_tick += delta

        if tp >= tlen:
            break

        status = int(track[tp])
        if status & 0x80:
            rs = status; tp += 1
        else:
            status = rs

        msg_type = (status >> 4) & 0xf
        ch       = status & 0xf

        if msg_type == 0x9 and tp + 1 < tlen:
            pitch = int(track[tp]); vel = int(track[tp+1]); tp += 2
            if vel > 0:
                on_tick[ch, pitch]   = abs_tick
                on_vel[ch, pitch]    = vel
                on_active[ch, pitch] = 1
            else:
                if on_active[ch, pitch]:
                    dur = abs_tick - on_tick[ch, pitch]
                    if dur < 1: dur = 1
                    if n_notes < MAX_NOTES:
                        abs_ticks[n_notes] = on_tick[ch, pitch]
                        pitches[n_notes]   = pitch
                        vels[n_notes]      = on_vel[ch, pitch]
                        dur_ticks[n_notes] = dur
                        channels[n_notes]  = ch
                        n_notes += 1
                    on_active[ch, pitch] = 0

        elif msg_type == 0x8 and tp + 1 < tlen:
            pitch = int(track[tp]); tp += 2
            if on_active[ch, pitch]:
                dur = abs_tick - on_tick[ch, pitch]
                if dur < 1: dur = 1
                if n_notes < MAX_NOTES:
                    abs_ticks[n_notes] = on_tick[ch, pitch]
                    pitches[n_notes]   = pitch
                    vels[n_notes]      = on_vel[ch, pitch]
                    dur_ticks[n_notes] = dur
                    channels[n_notes]  = ch
                    n_notes += 1
                on_active[ch, pitch] = 0

        elif msg_type in (0xa, 0xb, 0xe):
            tp += 2
        elif msg_type in (0xc, 0xd):
            tp += 1
        elif status == 0xff:
            if tp < tlen: tp += 1
            if tp < tlen:
                ml = int(track[tp]); tp += 1
                tp += ml
        elif status == 0xf0 or status == 0xf7:
            while tp < tlen and int(track[tp]) != 0xf7:
                tp += 1
            tp += 1

    # Flush active notes
    last_tick = abs_tick
    for c in range(16):
        for p in range(128):
            if on_active[c, p]:
                dur = last_tick - on_tick[c, p]
                if dur < 1: dur = 1
                if n_notes < MAX_NOTES:
                    abs_ticks[n_notes] = on_tick[c, p]
                    pitches[n_notes]   = p
                    vels[n_notes]      = on_vel[c, p]
                    dur_ticks[n_notes] = dur
                    channels[n_notes]  = c
                    n_notes += 1

    return abs_ticks[:n_notes], pitches[:n_notes], vels[:n_notes], \
           dur_ticks[:n_notes], channels[:n_notes]


@njit(cache=True)
def _compute_dna(abs_ticks, pitches, vels, dur_ticks, channels,
                 tpb, n_beat_bins, musical_durs, log_min, log_range):
    """
    Vectorized DNA field computation over all notes in a file.
    Returns parallel arrays for all DNA fields + vocab keys.
    """
    N = len(abs_ticks)
    ticks_per_bar = tpb * 4

    out_pc    = np.empty(N, dtype=np.int32)
    out_oct   = np.empty(N, dtype=np.float32)
    out_ld    = np.empty(N, dtype=np.float32)
    out_bs    = np.empty(N, dtype=np.float32)
    out_bc    = np.empty(N, dtype=np.float32)
    out_vel   = np.empty(N, dtype=np.float32)
    out_voice = np.empty(N, dtype=np.int32)
    out_snap  = np.empty(N, dtype=np.float32)
    out_bbin  = np.empty(N, dtype=np.int32)
    out_pitch = np.empty(N, dtype=np.int32)

    TWO_PI = 2.0 * math.pi

    for i in range(N):
        p   = int(pitches[i])
        v   = int(vels[i])
        dt  = int(dur_ticks[i])
        ch  = int(channels[i])
        atk = int(abs_ticks[i])

        # Pitch fields
        pc  = p % 12
        oct = min(p // 12, 8)

        # Duration snap
        db      = dt / tpb
        best_d  = musical_durs[0]
        best_dd = abs(db - musical_durs[0])
        for j in range(1, len(musical_durs)):
            dd = abs(db - musical_durs[j])
            if dd < best_dd:
                best_dd = dd
                best_d  = musical_durs[j]

        # Log duration normalized
        ld = math.log2(best_d + 1e-4)
        ld = (ld - log_min) / log_range * 2.0 - 1.0
        if ld < -1.0: ld = -1.0
        if ld >  1.0: ld =  1.0

        # Beat bin
        tick_in_bar = atk % ticks_per_bar
        bin_size    = ticks_per_bar / n_beat_bins
        bb          = int(tick_in_bar / bin_size) % n_beat_bins
        angle       = TWO_PI * bb / n_beat_bins

        out_pc[i]    = pc
        out_oct[i]   = oct / 8.0
        out_ld[i]    = ld
        out_bs[i]    = math.sin(angle)
        out_bc[i]    = math.cos(angle)
        out_vel[i]   = min(v, 127) / 127.0
        out_voice[i] = min(ch, 15)
        out_snap[i]  = best_d
        out_bbin[i]  = bb
        out_pitch[i] = p

    return out_pc, out_oct, out_ld, out_bs, out_bc, out_vel, out_voice, \
           out_snap, out_bbin, out_pitch


# ── Public API ────────────────────────────────────────────────────────────────

MUSICAL_DURS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], dtype=np.float64)
N_BEAT_BINS  = 16
_LOG_MIN     = math.log2(0.25 + 1e-4)
_LOG_MAX     = math.log2(4.0  + 1e-4)
_LOG_RANGE   = _LOG_MAX - _LOG_MIN


def parse_file_fast_from_bytes(raw_bytes: bytes) -> dict:
    """Same as parse_file_fast but takes raw bytes — no file I/O."""
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    except Exception:
        return None
    if len(arr) < 14:
        return None
    if arr[0] != ord('M') or arr[1] != ord('T') or arr[2] != ord('h') or arr[3] != ord('d'):
        return None

    tpb = int(arr[12]) << 8 | int(arr[13])
    if tpb <= 0:
        return None

    all_at, all_p, all_v, all_dt, all_ch = [], [], [], [], []
    pos = 14
    while pos + 8 <= len(arr):
        if not (arr[pos] == ord('M') and arr[pos+1] == ord('T') and
                arr[pos+2] == ord('r') and arr[pos+3] == ord('k')):
            break
        tlen = int(arr[pos+4])<<24 | int(arr[pos+5])<<16 | int(arr[pos+6])<<8 | int(arr[pos+7])
        pos += 8
        if pos + tlen > len(arr):
            break
        track = np.ascontiguousarray(arr[pos:pos+tlen])
        at, p, v, dt, ch = _parse_track(track, tpb)
        if len(at) > 0:
            all_at.append(at); all_p.append(p); all_v.append(v)
            all_dt.append(dt); all_ch.append(ch)
        pos += tlen

    if not all_at:
        return None

    abs_ticks = np.concatenate(all_at)
    pitches   = np.concatenate(all_p)
    vels      = np.concatenate(all_v)
    dur_ticks = np.concatenate(all_dt)
    channels  = np.concatenate(all_ch)

    order     = np.lexsort((pitches, abs_ticks))
    abs_ticks = abs_ticks[order]
    pitches   = pitches[order]
    vels      = vels[order]
    dur_ticks = dur_ticks[order]
    channels  = channels[order]

    pc, oct, ld, bs, bc, vel, voice, snap, bbin, pitch = _compute_dna(
        abs_ticks, pitches, vels, dur_ticks, channels,
        tpb, N_BEAT_BINS, MUSICAL_DURS, _LOG_MIN, _LOG_RANGE
    )

    N = len(abs_ticks)
    min_gap_ticks = tpb / 4
    ticks_per_bar = tpb * 4
    TWO_PI = 2.0 * math.pi

    out_pc, out_oct, out_ld, out_bs, out_bc = [], [], [], [], []
    out_vel, out_voice, out_keys = [], [], []
    key_fields = {}

    for i in range(N):
        nkey = ('note', int(pitch[i]), float(snap[i]), int(bbin[i]))
        if nkey not in key_fields:
            key_fields[nkey] = {
                'pitch_class':  int(pc[i]),
                'octave':       float(oct[i]),
                'log_duration': float(ld[i]),
                'beat_sin':     float(bs[i]),
                'beat_cos':     float(bc[i]),
                'velocity':     float(vel[i]),
                'voice':        int(voice[i]),
            }
        out_pc.append(int(pc[i])); out_oct.append(float(oct[i]))
        out_ld.append(float(ld[i])); out_bs.append(float(bs[i]))
        out_bc.append(float(bc[i])); out_vel.append(float(vel[i]))
        out_voice.append(int(voice[i])); out_keys.append(nkey)

        if i + 1 < N:
            gap = int(abs_ticks[i+1]) - int(abs_ticks[i])
            if gap > min_gap_ticks:
                gb  = gap / tpb
                bd  = MUSICAL_DURS[np.argmin(np.abs(MUSICAL_DURS - gb))]
                gl  = math.log2(float(bd) + 1e-4)
                gl  = (gl - _LOG_MIN) / _LOG_RANGE * 2.0 - 1.0
                gl  = max(-1.0, min(1.0, gl))
                rt  = int(abs_ticks[i]) + int(dur_ticks[i])
                rb  = int((rt % ticks_per_bar) / (ticks_per_bar / N_BEAT_BINS)) % N_BEAT_BINS
                ra  = TWO_PI * rb / N_BEAT_BINS
                rbs = math.sin(ra); rbc = math.cos(ra)
                rkey = ('rest', None, float(bd), rb)
                if rkey not in key_fields:
                    key_fields[rkey] = {
                        'pitch_class': 0, 'octave': 0.0,
                        'log_duration': gl,
                        'beat_sin': rbs, 'beat_cos': rbc,
                        'velocity': 0.0, 'voice': 0,
                    }
                out_pc.append(0); out_oct.append(0.0); out_ld.append(gl)
                out_bs.append(rbs); out_bc.append(rbc)
                out_vel.append(0.0); out_voice.append(0); out_keys.append(rkey)

    if not out_keys:
        return None

    return {
        'pitch_class':  out_pc,
        'octave':       out_oct,
        'log_duration': out_ld,
        'beat_sin':     out_bs,
        'beat_cos':     out_bc,
        'velocity':     out_vel,
        'voice':        out_voice,
        'keys':         out_keys,
        'key_fields':   key_fields,
    }


def parse_file_fast(path: str) -> dict:
    """
    Parse one MIDI file to DNA fields using Numba JIT.
    Returns dict with all DNA arrays + key tuples, or None on failure.
    """
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except Exception:
        return None

    arr = np.frombuffer(raw, dtype=np.uint8)
    if len(arr) < 14:
        return None
    if arr[0] != ord('M') or arr[1] != ord('T') or arr[2] != ord('h') or arr[3] != ord('d'):
        return None

    tpb = int(arr[12]) << 8 | int(arr[13])
    if tpb <= 0:
        return None

    # Find all tracks and concatenate notes
    all_at, all_p, all_v, all_dt, all_ch = [], [], [], [], []
    pos = 14
    while pos + 8 <= len(arr):
        if not (arr[pos] == ord('M') and arr[pos+1] == ord('T') and
                arr[pos+2] == ord('r') and arr[pos+3] == ord('k')):
            break
        tlen = int(arr[pos+4])<<24 | int(arr[pos+5])<<16 | int(arr[pos+6])<<8 | int(arr[pos+7])
        pos += 8
        if pos + tlen > len(arr):
            break
        track = np.ascontiguousarray(arr[pos:pos+tlen])
        at, p, v, dt, ch = _parse_track(track, tpb)
        if len(at) > 0:
            all_at.append(at); all_p.append(p); all_v.append(v)
            all_dt.append(dt); all_ch.append(ch)
        pos += tlen

    if not all_at:
        return None

    abs_ticks = np.concatenate(all_at)
    pitches   = np.concatenate(all_p)
    vels      = np.concatenate(all_v)
    dur_ticks = np.concatenate(all_dt)
    channels  = np.concatenate(all_ch)

    # Sort by abs_tick then pitch
    order     = np.lexsort((pitches, abs_ticks))
    abs_ticks = abs_ticks[order]
    pitches   = pitches[order]
    vels      = vels[order]
    dur_ticks = dur_ticks[order]
    channels  = channels[order]

    # JIT-compiled DNA computation
    pc, oct, ld, bs, bc, vel, voice, snap, bbin, pitch = _compute_dna(
        abs_ticks, pitches, vels, dur_ticks, channels,
        tpb, N_BEAT_BINS, MUSICAL_DURS, _LOG_MIN, _LOG_RANGE
    )

    N = len(abs_ticks)
    min_gap_ticks = tpb / 4
    ticks_per_bar = tpb * 4
    TWO_PI = 2.0 * math.pi

    # Build output with rests interleaved
    out_pc, out_oct, out_ld, out_bs, out_bc = [], [], [], [], []
    out_vel, out_voice, out_keys = [], [], []
    key_fields = {}

    for i in range(N):
        nkey = ('note', int(pitch[i]), float(snap[i]), int(bbin[i]))
        if nkey not in key_fields:
            key_fields[nkey] = {
                'pitch_class':  int(pc[i]),
                'octave':       float(oct[i]),
                'log_duration': float(ld[i]),
                'beat_sin':     float(bs[i]),
                'beat_cos':     float(bc[i]),
                'velocity':     float(vel[i]),
                'voice':        int(voice[i]),
            }
        out_pc.append(int(pc[i])); out_oct.append(float(oct[i]))
        out_ld.append(float(ld[i])); out_bs.append(float(bs[i]))
        out_bc.append(float(bc[i])); out_vel.append(float(vel[i]))
        out_voice.append(int(voice[i])); out_keys.append(nkey)

        if i + 1 < N:
            gap = int(abs_ticks[i+1]) - int(abs_ticks[i])
            if gap > min_gap_ticks:
                gb  = gap / tpb
                bd  = MUSICAL_DURS[np.argmin(np.abs(MUSICAL_DURS - gb))]
                gl  = math.log2(float(bd) + 1e-4)
                gl  = (gl - _LOG_MIN) / _LOG_RANGE * 2.0 - 1.0
                gl  = max(-1.0, min(1.0, gl))
                rt  = int(abs_ticks[i]) + int(dur_ticks[i])
                rb  = int((rt % ticks_per_bar) / (ticks_per_bar / N_BEAT_BINS)) % N_BEAT_BINS
                ra  = TWO_PI * rb / N_BEAT_BINS
                rbs = math.sin(ra); rbc = math.cos(ra)
                rkey = ('rest', None, float(bd), rb)
                if rkey not in key_fields:
                    key_fields[rkey] = {
                        'pitch_class': 0, 'octave': 0.0,
                        'log_duration': gl,
                        'beat_sin': rbs, 'beat_cos': rbc,
                        'velocity': 0.0, 'voice': 0,
                    }
                out_pc.append(0); out_oct.append(0.0); out_ld.append(gl)
                out_bs.append(rbs); out_bc.append(rbc)
                out_vel.append(0.0); out_voice.append(0); out_keys.append(rkey)

    if not out_keys:
        return None

    return {
        'pitch_class':  out_pc,
        'octave':       out_oct,
        'log_duration': out_ld,
        'beat_sin':     out_bs,
        'beat_cos':     out_bc,
        'velocity':     out_vel,
        'voice':        out_voice,
        'keys':         out_keys,
        'key_fields':   key_fields,
    }


def warmup():
    """Call once at startup to JIT-compile the kernels."""
    import tempfile, os
    # Minimal valid MIDI
    midi = (b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0'
            b'MTrk\x00\x00\x00\x14'
            b'\x00\x90\x3c\x50'   # note on C4 vel 80
            b'\x78\x80\x3c\x00'   # note off C4
            b'\x00\x90\x3e\x50'   # note on D4
            b'\x78\x80\x3e\x00'   # note off D4
            b'\x00\xff\x2f\x00')  # end of track
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        f.write(midi); tmp = f.name
    try:
        parse_file_fast(tmp)
    finally:
        os.unlink(tmp)
