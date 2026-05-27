"""
FM Benchmark
============
Full throughput, memory, and correctness benchmarking.
Compares FM training speed, inference speed, memory usage.
"""

import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.fm import FM


def fmt_num(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}k"
    return str(int(n))

def gpu_mem():
    if not torch.cuda.is_available(): return {}
    d = torch.cuda.current_device()
    return {
        'alloc_gb':    round(torch.cuda.memory_allocated(d) / 1e9, 3),
        'reserved_gb': round(torch.cuda.memory_reserved(d) / 1e9, 3),
        'total_gb':    round(torch.cuda.get_device_properties(d).total_memory / 1e9, 3),
    }

def make_batch(batch_size, seq_len, vocab_size, device):
    """Generate a random batch of DNA fields."""
    return {
        'pitch_class':  torch.randint(0, 12,    (batch_size, seq_len), device=device),
        'octave':       torch.rand(batch_size, seq_len, device=device),
        'log_duration': torch.rand(batch_size, seq_len, device=device) * 2 - 1,
        'beat_sin':     torch.rand(batch_size, seq_len, device=device) * 2 - 1,
        'beat_cos':     torch.rand(batch_size, seq_len, device=device) * 2 - 1,
        'velocity':     torch.rand(batch_size, seq_len, device=device),
        'voice':        torch.randint(0, 16,    (batch_size, seq_len), device=device),
        'target':       torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
    }

def benchmark_training(model, vocab_size, device, dtype,
                        batch_size, seq_len, warmup, iters):
    """Benchmark training forward+backward throughput."""
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(warmup):
        batch = make_batch(batch_size, seq_len, vocab_size, device)
        with torch.amp.autocast('cuda', dtype=dtype, enabled=(device.type=='cuda')):
            logits = model(**{k: v for k, v in batch.items() if k != 'target'})
            loss   = criterion(logits.reshape(-1, vocab_size), batch['target'].reshape(-1))
        loss.backward()
        if device.type == 'cuda': torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None

    t0 = time.perf_counter()
    for _ in range(iters):
        batch = make_batch(batch_size, seq_len, vocab_size, device)
        with torch.amp.autocast('cuda', dtype=dtype, enabled=(device.type=='cuda')):
            logits = model(**{k: v for k, v in batch.items() if k != 'target'})
            loss   = criterion(logits.reshape(-1, vocab_size), batch['target'].reshape(-1))
        loss.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    it_per_sec  = iters / elapsed
    tok_per_sec = iters * batch_size * seq_len / elapsed
    ms_per_iter = elapsed / iters * 1000

    peak_mem = {}
    if device.type == 'cuda':
        peak_mem['peak_gb'] = round(
            torch.cuda.max_memory_allocated() / 1e9, 3)

    return it_per_sec, tok_per_sec, ms_per_iter, peak_mem

def benchmark_inference(model, vocab_size, device, dtype,
                        batch_size, seq_len, warmup, iters):
    """Benchmark inference (forward only) throughput."""
    model.eval()

    for _ in range(warmup):
        batch = make_batch(batch_size, seq_len, vocab_size, device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype, enabled=(device.type=='cuda')):
                _ = model(**{k: v for k, v in batch.items() if k != 'target'})
        if device.type == 'cuda': torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        batch = make_batch(batch_size, seq_len, vocab_size, device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype, enabled=(device.type=='cuda')):
                _ = model(**{k: v for k, v in batch.items() if k != 'target'})
        if device.type == 'cuda': torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    it_per_sec  = iters / elapsed
    tok_per_sec = iters * batch_size * seq_len / elapsed
    ms_per_iter = elapsed / iters * 1000
    return it_per_sec, tok_per_sec, ms_per_iter

def benchmark_seq_scaling(model, vocab_size, device, dtype, batch_size=4):
    """Test how throughput scales with sequence length — should be nearly linear (no quadratic)."""
    model.eval()
    results = []
    for seq_len in [64, 128, 256, 512, 1024, 2048, 4096]:
        try:
            batch = make_batch(batch_size, seq_len, vocab_size, device)
            # Warmup
            for _ in range(2):
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype, enabled=(device.type=='cuda')):
                        _ = model(**{k: v for k, v in batch.items() if k != 'target'})
                if device.type == 'cuda': torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(5):
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype, enabled=(device.type=='cuda')):
                        _ = model(**{k: v for k, v in batch.items() if k != 'target'})
                if device.type == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            ms = elapsed / 5 * 1000
            tps = 5 * batch_size * seq_len / elapsed
            mem = gpu_mem()
            results.append((seq_len, ms, tps, mem.get('alloc_gb', 0)))
        except torch.cuda.OutOfMemoryError:
            results.append((seq_len, -1, -1, -1))
            break
    return results


def run_benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = (torch.bfloat16 if (torch.cuda.is_available() and
               torch.cuda.is_bf16_supported()) else torch.float32)

    vocab_size = args.vocab_size
    bar = '=' * 72

    print(f"\n{bar}")
    print(f"  FM Benchmark")
    print(f"  Device: {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Dtype:  {str(dtype).split('.')[-1]}")
    print(bar)

    model = FM(
        vocab_size       = vocab_size,
        dim              = args.dim,
        n_decoder_layers = args.decoder_layers,
        decoder_hidden   = args.decoder_hidden,
    ).to(device)

    print(f"\n  Parameters:  {fmt_num(model.count_parameters())}")
    print(f"  Field dim:   {args.dim}")
    print(f"  Decoder:     {args.decoder_layers} layers × {args.decoder_hidden}")
    print(f"  Vocab:       {vocab_size}")

    m = gpu_mem()
    if m:
        print(f"  VRAM (model load): {m['alloc_gb']:.3f}GB / {m['total_gb']:.1f}GB")

    # ── Training benchmark ────────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  TRAINING  (batch={args.batch_size}, seq={args.seq_len}, "
          f"{args.warmup} warmup, {args.iters} iters)")
    print(bar)

    it_s, tok_s, ms, peak = benchmark_training(
        model, vocab_size, device, dtype,
        args.batch_size, args.seq_len, args.warmup, args.iters)
    print(f"  {it_s:.2f} it/s  |  {tok_s/1000:.1f}k tok/s  |  {ms:.1f} ms/iter")
    if peak: print(f"  Peak VRAM: {peak['peak_gb']:.3f}GB")

    # ── Inference benchmark ───────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  INFERENCE  (batch={args.batch_size}, seq={args.seq_len}, "
          f"{args.warmup} warmup, {args.iters} iters)")
    print(bar)

    it_s, tok_s, ms = benchmark_inference(
        model, vocab_size, device, dtype,
        args.batch_size, args.seq_len, args.warmup, args.iters)
    print(f"  {it_s:.2f} it/s  |  {tok_s/1000:.1f}k tok/s  |  {ms:.1f} ms/iter")

    # ── Sequence length scaling ───────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  SEQUENCE LENGTH SCALING  (batch={args.scale_batch})")
    print(f"  FM is O(n) — expect linear scaling, not quadratic")
    print(bar)
    print(f"  {'seq_len':>8} | {'ms/iter':>10} | {'tok/s':>12} | {'VRAM':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")

    results = benchmark_seq_scaling(model, vocab_size, device, dtype, args.scale_batch)
    prev_ms = None
    for seq_len, ms, tps, vram in results:
        if ms < 0:
            print(f"  {seq_len:>8} | {'OOM':>10} | {'':>12} | {'':>8}")
            continue
        scale = f"({ms/prev_ms:.2f}x)" if prev_ms else ""
        print(f"  {seq_len:>8} | {ms:>8.1f}ms {scale:>6} | {tps/1000:>10.1f}k | {vram:>6.3f}GB")
        prev_ms = ms

    print(f"\n{bar}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FM Benchmark')
    parser.add_argument('--vocab_size',     type=int,   default=512)
    parser.add_argument('--dim',            type=int,   default=4096)
    parser.add_argument('--decoder_layers', type=int,   default=3)
    parser.add_argument('--decoder_hidden', type=int,   default=2048)
    parser.add_argument('--batch_size',     type=int,   default=8)
    parser.add_argument('--seq_len',        type=int,   default=512)
    parser.add_argument('--scale_batch',    type=int,   default=2)
    parser.add_argument('--warmup',         type=int,   default=3)
    parser.add_argument('--iters',          type=int,   default=20)
    args = parser.parse_args()
    run_benchmark(args)
