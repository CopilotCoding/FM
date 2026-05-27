"""
FM Generation
=============
O(1) per token inference. Generates MIDI from a trained FM model.
Optionally seeds from a real MIDI file prompt.
"""

import sys
import math
import struct
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.tokenizer import FMTokenizer, parse_midi
from model.fm       import FM


# ── MIDI Writer ───────────────────────────────────────────────────────────────

def write_variable_length(value: int) -> bytes:
    result = []
    result.append(value & 0x7f)
    value >>= 7
    while value:
        result.append((value & 0x7f) | 0x80)
        value >>= 7
    return bytes(reversed(result))

def fields_to_midi_note(fields: dict, tpb: int = 480):
    """Convert DNA fields back to MIDI note parameters."""
    pc       = int(fields['pitch_class'])
    oct_int  = max(0, min(8, round(fields['octave'] * 8)))
    pitch    = min(127, max(0, oct_int * 12 + pc))

    # Reverse log_duration normalization
    log_dur_norm = fields['log_duration']
    log_dur_raw  = (log_dur_norm + 1.0) / 2.0 * 5.0 + (-2.0)
    dur_beats    = max(0.25, min(8.0, 2 ** log_dur_raw))
    dur_ticks    = int(dur_beats * tpb)

    vel   = max(1, min(127, int(fields['velocity'] * 127)))
    ch    = int(fields['voice']) % 16

    return pitch, vel, dur_ticks, ch

def tokens_to_midi(token_indices: list, tokenizer: FMTokenizer,
                   tpb: int = 480, out_path: str = 'generated.mid'):
    """Write generated token sequence to a MIDI file."""
    events = []
    cur_tick = 0

    for idx in token_indices:
        if idx < FMTokenizer.SPECIAL:
            continue
        fields = tokenizer.idx_to_fields[idx]
        if fields is None:
            continue

        # Advance by actual predicted duration (beats -> ticks)
        dur_beats = tokenizer.get_duration_beats(idx)
        dur_ticks_advance = max(int(dur_beats * tpb), 1)

        # REST token: advance time, emit no note
        if tokenizer.is_rest(idx):
            cur_tick += dur_ticks_advance
            continue

        pitch, vel, dur_ticks, ch = fields_to_midi_note(fields, tpb)
        events.append(('on',  cur_tick,              ch, pitch, vel))
        events.append(('off', cur_tick + dur_ticks,  ch, pitch, 0))
        cur_tick += dur_ticks_advance

    events.sort(key=lambda e: e[1])

    # Build MIDI bytes — single track format 0
    track_bytes = bytearray()
    prev_tick   = 0

    # Tempo: 120 BPM = 500000 us/beat
    track_bytes += write_variable_length(0)
    track_bytes += b'\xff\x51\x03\x07\xa1\x20'

    for ev in events:
        kind, tick, ch, pitch, vel = ev
        delta = tick - prev_tick
        prev_tick = tick
        track_bytes += write_variable_length(delta)
        if kind == 'on':
            track_bytes += bytes([0x90 | ch, pitch, vel])
        else:
            track_bytes += bytes([0x80 | ch, pitch, 0])

    # End of track
    track_bytes += b'\x00\xff\x2f\x00'

    # MIDI header
    header = b'MThd'
    header += struct.pack('>I', 6)
    header += struct.pack('>HHH', 0, 1, tpb)

    track_header = b'MTrk'
    track_header += struct.pack('>I', len(track_bytes))

    with open(out_path, 'wb') as f:
        f.write(header + track_header + track_bytes)

    print(f"  Wrote {len(token_indices)} tokens → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if (torch.cuda.is_available() and
                                 torch.cuda.is_bf16_supported()) else torch.float32

    # Load tokenizer
    tok_path = Path(args.checkpoint).parent / 'tokenizer.pkl'
    if not tok_path.exists():
        tok_path = Path(args.tokenizer)
    print(f"Loading tokenizer from {tok_path}")
    tok = FMTokenizer.load(str(tok_path))

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt   = torch.load(args.checkpoint, map_location=device)
    config = ckpt['config']

    model = FM(
        vocab_size       = config['vocab_size'],
        dim              = config['dim'],
        n_decoder_layers = config.get('decoder_layers', 3),
        decoder_hidden   = config.get('decoder_hidden', 2048),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.idx_to_fields = tok.idx_to_fields
    model.eval()

    print(f"  Model: {model.count_parameters():,} parameters")
    print(f"  Vocab: {config['vocab_size']} tokens")

    # Build prompt
    if args.prompt_midi:
        print(f"  Seeding from: {args.prompt_midi}")
        fields_seq = tok.tokenize_file(args.prompt_midi)
        if fields_seq is None:
            print("  Failed to tokenize prompt, using empty prompt")
            fields_seq = None

    if not args.prompt_midi or fields_seq is None:
        # Empty prompt — start from zero field
        fields_seq = {
            'pitch_class':  [0],
            'octave':       [0.5],
            'log_duration': [0.0],
            'beat_sin':     [0.0],
            'beat_cos':     [1.0],
            'velocity':     [0.5],
            'voice':        [0],
        }

    # Convert to tensors
    prompt = {}
    for k, v in fields_seq.items():
        if k == 'token_idx':
            continue
        t = torch.tensor([v], dtype=torch.float32)
        if k in {'pitch_class', 'voice'}:
            t = t.long()
        prompt[k] = t

    interp_str = f" | lerp seed_b={args.seed_b} alpha={args.alpha}" if args.seed_b is not None else ""
    print(f"\nGenerating {args.tokens} tokens...")
    print(f"  Temperature: {args.temperature} | top_k: {args.top_k} | top_p: {args.top_p} | seed: {args.seed}{interp_str}")

    with torch.amp.autocast('cuda', dtype=dtype, enabled=(device.type=='cuda')):
        generated = model.generate(
            prompt_fields  = prompt,
            max_new_tokens = args.tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
            seed           = args.seed,
            seed_b         = args.seed_b,
            alpha          = args.alpha,
        )

    print(f"  Generated {len(generated)} tokens")

    out_path = args.output
    tokens_to_midi(generated, tok, tpb=480, out_path=out_path)
    print(f"\nDone. Output: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FM Generation')
    parser.add_argument('--checkpoint',   required=True,              help='Path to .pt checkpoint')
    parser.add_argument('--tokenizer',    default='',                 help='Path to tokenizer.pkl (auto-detected if empty)')
    parser.add_argument('--output',       default='generated.mid',    help='Output MIDI path')
    parser.add_argument('--prompt_midi',  default='',                 help='Seed MIDI file (optional)')
    parser.add_argument('--tokens',       type=int,   default=512,    help='Tokens to generate')
    parser.add_argument('--temperature',  type=float, default=0.85,   help='Sampling temperature')
    parser.add_argument('--top_k',          type=int,   default=50)
    parser.add_argument('--top_p',          type=float, default=0.95)
    parser.add_argument('--seed',    type=int,   default=None, help='Seed integer (None=random)')
    parser.add_argument('--seed_b',  type=int,   default=None, help='Second seed for interpolation')
    parser.add_argument('--alpha',   type=float, default=0.5,  help='Interpolation weight: alpha*seed + (1-alpha)*seed_b')
    args = parser.parse_args()
    generate(args)
