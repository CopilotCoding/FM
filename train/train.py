"""
FM Training Script
==================
Full statistics, full debugging, full checkpointing, full benchmarking.

Features:
  - bf16/fp16 mixed precision auto-detected
  - Full GPU stats every N steps (VRAM alloc/reserved/total, utilization)
  - Rolling loss window (smoothed loss)
  - Throughput: tokens/sec, notes/sec, iterations/sec
  - ETA per epoch and full run
  - Step checkpoints (every N steps)
  - Timed checkpoints (every N minutes)
  - Best checkpoint tracking
  - Full CSV log every step
  - JSON run stats on completion
  - Gradient norm tracking
  - Per-epoch summary
  - Resume from checkpoint
  - Warmup steps
  - Gradient clipping
  - Weight decay
  - Cosine LR schedule with warmup
"""

import os
import csv
import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.tokenizer import FMTokenizer
from data.dataset   import FMDataset, collate_fn, FIELD_NAMES
from model.fm       import FM


# ── Utilities ──────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def fmt_num(n: int) -> str:
    if n >= 1_000_000_000: return f"{n/1e9:.2f}B"
    if n >= 1_000_000:     return f"{n/1e6:.2f}M"
    if n >= 1_000:         return f"{n/1e3:.1f}k"
    return str(n)

def gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    d        = torch.cuda.current_device()
    alloc    = torch.cuda.memory_allocated(d)
    reserved = torch.cuda.memory_reserved(d)
    total    = torch.cuda.get_device_properties(d).total_memory
    try:
        util = torch.cuda.utilization(d)
    except Exception:
        util = -1
    return {
        'vram_alloc_gb':    round(alloc    / 1e9, 3),
        'vram_reserved_gb': round(reserved / 1e9, 3),
        'vram_total_gb':    round(total    / 1e9, 3),
        'gpu_util_pct':     util,
    }

def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    step, loss, config):
    torch.save({
        'epoch':     epoch,
        'step':      step,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'loss':      loss,
        'config':    config,
        'timestamp': datetime.now().isoformat(),
    }, path)

def write_csv(csv_path: Path, row: dict):
    exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def batch_to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = None

    use_amp    = dtype is not None
    use_scaler = dtype == torch.float16
    scaler     = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok_path = Path(args.out_dir) / 'tokenizer.pkl'
    if tok_path.exists() and not args.retokenize:
        print(f"Loading tokenizer from {tok_path}")
        tok = FMTokenizer.load(str(tok_path))
    else:
        print("Building tokenizer...")
        tok = FMTokenizer().build(args.midi_dir, verbose=True)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        tok.save(str(tok_path))
        print(f"Tokenizer saved to {tok_path}")

    print(f"Tokenizer: {tok.vocab_size} tokens")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Tokenizing corpus...")
    sequences = tok.tokenize_corpus(args.midi_dir, verbose=True)

    dataset = FMDataset(sequences, min_len=args.min_seq_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    print(f"Dataset: {len(dataset)} sequences | {len(loader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FM(
        vocab_size       = tok.vocab_size,
        dim              = args.dim,
        n_decoder_layers = args.decoder_layers,
        decoder_hidden   = args.decoder_hidden,
        dropout          = args.dropout,
    ).to(device)

    # Attach idx_to_fields for generation
    model.idx_to_fields = tok.idx_to_fields

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = args.epochs * len(loader)
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    import math
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ── Output paths ──────────────────────────────────────────────────────────
    out       = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / 'latest.pt'
    best_path = out / 'best.pt'
    csv_path  = out / 'training_log.csv'
    stats_path= out / 'run_stats.json'

    config = {
        'vocab_size':      tok.vocab_size,
        'dim':             args.dim,
        'decoder_layers':  args.decoder_layers,
        'decoder_hidden':  args.decoder_hidden,
        'dropout':         args.dropout,
        'parameters':      model.count_parameters(),
        'pos_base':        32768,
    }
    with open(out / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    best_loss    = float('inf')

    if ckpt_path.exists() and not args.fresh:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler'):
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        global_step = ckpt.get('step', 0)
        best_loss   = ckpt.get('loss', float('inf'))
        print(f"  Resumed: epoch {start_epoch}, step {global_step}, loss {ckpt['loss']:.4f}")

    # ── Header ────────────────────────────────────────────────────────────────
    bar = '=' * 72
    g   = gpu_stats()
    print(f"\n{bar}")
    print(f"  FM — Field Machine Training  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(bar)
    print(f"  Device:          {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Precision:       {str(dtype).split('.')[-1] if dtype else 'fp32'}")
    print(f"  Parameters:      {fmt_num(model.count_parameters())}")
    print(f"  Vocab size:      {tok.vocab_size}")
    print(f"  Field dim:       {args.dim}")
    print(f"  Decoder:         {args.decoder_layers} layers, hidden={args.decoder_hidden}")
    print(f"  Dataset:         {len(dataset)} sequences")
    print(f"  Batch size:      {args.batch_size}  |  {len(loader)} batches/epoch")
    print(f"  Epochs:          {args.epochs}  |  Total steps: {total_steps}")
    print(f"  LR:              {args.lr:.1e} (warmup {warmup_steps} steps, cosine decay)")
    print(f"  Weight decay:    {args.weight_decay}")
    print(f"  Grad clip:       {args.grad_clip}")
    if g:
        print(f"  VRAM:            {g['vram_alloc_gb']:.2f}GB alloc / {g['vram_total_gb']:.1f}GB total")
    print(bar)
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    run_start        = time.time()
    last_timed_save  = time.time()
    epoch_times      = []
    tokens_total     = 0
    recent_losses    = deque(maxlen=100)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss        = 0.0
        epoch_start       = time.time()
        steps_this_epoch  = 0
        epoch_tokens      = 0
        epoch_grad_norms  = []

        for step, batch in enumerate(loader):
            step_start = time.time()
            batch      = batch_to_device(batch, device)

            with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
                logits = model(
                    pitch_class  = batch['pitch_class'],
                    octave       = batch['octave'],
                    log_duration = batch['log_duration'],
                    beat_sin     = batch['beat_sin'],
                    beat_cos     = batch['beat_cos'],
                    velocity     = batch['velocity'],
                    voice        = batch['voice'],
                )
                B, T, V = logits.shape
                loss = criterion(
                    logits.reshape(B * T, V),
                    batch['target'].reshape(B * T),
                )

            optimizer.zero_grad()
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()

            step_time    = time.time() - step_start
            loss_val     = loss.item()
            # Count non-padding tokens
            tokens_step  = int((batch['target'] != -100).sum().item())
            tok_per_sec  = tokens_step / max(step_time, 1e-6)

            total_loss        += loss_val
            global_step       += 1
            steps_this_epoch  += 1
            epoch_tokens      += tokens_step
            tokens_total      += tokens_step
            recent_losses.append(loss_val)
            epoch_grad_norms.append(float(grad_norm))

            avg_loss    = total_loss / steps_this_epoch
            smooth_loss = sum(recent_losses) / len(recent_losses)
            lr_now      = scheduler.get_last_lr()[0]
            elapsed     = time.time() - epoch_start
            rate        = steps_this_epoch / max(elapsed, 1e-6)
            eta_epoch   = (len(loader) - step - 1) / max(rate, 1e-6)
            run_elapsed = time.time() - run_start
            eta_run     = eta_epoch + (elapsed / steps_this_epoch * len(loader)) * (args.epochs - epoch - 1)

            # ── Print ──────────────────────────────────────────────────────
            if global_step % args.print_steps == 0:
                g = gpu_stats()
                vram = f" | vram {g['vram_alloc_gb']:.2f}/{g['vram_total_gb']:.1f}GB" if g else ''
                gn   = f" | gnorm {grad_norm:.3f}"
                print(
                    f"  step {global_step:>7} | ep {epoch+1}/{args.epochs} "
                    f"| loss {avg_loss:.4f} (↑smooth {smooth_loss:.4f}) "
                    f"| lr {lr_now:.2e}"
                    f"{gn}"
                    f" | {rate:.2f}it/s | {tok_per_sec/1000:.1f}k tok/s"
                    f" | elapsed {fmt_time(run_elapsed)}"
                    f" | ETA ep {fmt_time(eta_epoch)} | ETA run {fmt_time(eta_run)}"
                    f"{vram}"
                )

            # ── CSV log ───────────────────────────────────────────────────
            g = gpu_stats()
            write_csv(csv_path, {
                'step':             global_step,
                'epoch':            epoch + 1,
                'loss':             round(loss_val, 6),
                'avg_loss':         round(avg_loss, 6),
                'smooth_loss':      round(smooth_loss, 6),
                'grad_norm':        round(float(grad_norm), 4),
                'lr':               round(lr_now, 8),
                'it_per_sec':       round(rate, 3),
                'tok_per_sec':      round(tok_per_sec, 0),
                'tokens_total':     tokens_total,
                'vram_alloc_gb':    g.get('vram_alloc_gb', ''),
                'vram_reserved_gb': g.get('vram_reserved_gb', ''),
                'gpu_util_pct':     g.get('gpu_util_pct', ''),
                'elapsed_sec':      round(run_elapsed, 1),
                'timestamp':        datetime.now().isoformat(),
            })

            # ── Step checkpoint ───────────────────────────────────────────
            if global_step % args.save_steps == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                epoch, global_step, avg_loss, config)
                print(f"  >>> [step ckpt] step {global_step} | loss {avg_loss:.4f}")

            # ── Timed checkpoint ──────────────────────────────────────────
            if (time.time() - last_timed_save) >= args.save_minutes * 60:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                tp = out / f'timed_{ts}_step{global_step}.pt'
                save_checkpoint(tp, model, optimizer, scheduler,
                                epoch, global_step, avg_loss, config)
                last_timed_save = time.time()
                print(f"  >>> [timed ckpt] {tp.name} | loss {avg_loss:.4f}")

        # ── End of epoch ──────────────────────────────────────────────────────
        avg_loss    = total_loss / len(loader)
        epoch_time  = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_gnorm   = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            save_checkpoint(best_path, model, optimizer, scheduler,
                            epoch + 1, global_step, avg_loss, config)
            best_tag = '  ★ new best'
        else:
            best_tag = ''

        epochs_left   = args.epochs - epoch - 1
        avg_epoch_t   = sum(epoch_times[-3:]) / len(epoch_times[-3:])
        eta_finish    = datetime.now() + timedelta(seconds=avg_epoch_t * epochs_left)
        epoch_tok_ps  = epoch_tokens / max(epoch_time, 1e-6)

        print(f"\n  {'='*68}")
        print(f"  Epoch {epoch+1}/{args.epochs} complete{best_tag}")
        print(f"    Loss:          {avg_loss:.4f}  (best: {best_loss:.4f})")
        print(f"    Avg grad norm: {avg_gnorm:.4f}")
        print(f"    Epoch time:    {fmt_time(epoch_time)}")
        print(f"    Total elapsed: {fmt_time(time.time() - run_start)}")
        print(f"    Tokens seen:   {fmt_num(tokens_total)}  ({fmt_num(epoch_tokens)} this epoch)")
        print(f"    Tok/sec:       {epoch_tok_ps/1000:.1f}k (epoch avg)")
        print(f"    ETA finish:    {eta_finish.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  {'='*68}\n")

        save_checkpoint(ckpt_path, model, optimizer, scheduler,
                        epoch + 1, global_step, avg_loss, config)
        torch.save({
            'epoch':  epoch,
            'model':  model.state_dict(),
            'config': config,
            'loss':   avg_loss,
        }, out / f'epoch_{epoch+1:03d}_loss{avg_loss:.4f}.pt')

    # ── Final stats ───────────────────────────────────────────────────────────
    total_time = time.time() - run_start
    final = {
        'total_time_sec':  round(total_time, 1),
        'total_tokens':    tokens_total,
        'avg_tok_per_sec': round(tokens_total / total_time, 0),
        'best_loss':       round(best_loss, 6),
        'epochs':          args.epochs,
        'parameters':      model.count_parameters(),
        'dim':             args.dim,
        'vocab_size':      tok.vocab_size,
        'completed':       datetime.now().isoformat(),
    }
    with open(stats_path, 'w') as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  Training complete in {fmt_time(total_time)}")
    print(f"  Total tokens:   {fmt_num(tokens_total)}")
    print(f"  Avg throughput: {tokens_total/total_time/1000:.1f}k tok/sec")
    print(f"  Best loss:      {best_loss:.4f}")
    print(f"  Stats:          {stats_path}")
    print(f"  Log:            {csv_path}")
    print(f"{'='*72}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import math

    parser = argparse.ArgumentParser(description='FM — Field Machine Training')
    parser.add_argument('--midi_dir',       required=True,              help='Directory of MIDI files')
    parser.add_argument('--out_dir',        default='checkpoints',      help='Output directory')
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=1,      help='Sequences per batch (default=1: full files, no padding, no OOM)')
    parser.add_argument('--dim',            type=int,   default=4096,   help='Field dimension')
    parser.add_argument('--decoder_layers', type=int,   default=3,      help='Decoder hidden layers')
    parser.add_argument('--decoder_hidden', type=int,   default=2048,   help='Decoder hidden dim')
    parser.add_argument('--dropout',        type=float, default=0.1)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--weight_decay',   type=float, default=0.01)
    parser.add_argument('--grad_clip',      type=float, default=1.0)
    parser.add_argument('--warmup_steps',   type=int,   default=200)
    parser.add_argument('--min_seq_len',    type=int,   default=8,      help='Minimum sequence length')
    parser.add_argument('--workers',        type=int,   default=0)
    parser.add_argument('--save_steps',     type=int,   default=500)
    parser.add_argument('--save_minutes',   type=int,   default=30)
    parser.add_argument('--print_steps',    type=int,   default=10)
    parser.add_argument('--retokenize',     action='store_true',        help='Force re-tokenize corpus')
    parser.add_argument('--fresh',          action='store_true',        help='Ignore existing checkpoint')
    args = parser.parse_args()
    train(args)
