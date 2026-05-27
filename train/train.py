"""
FM Training Script
==================
Full statistics, full debugging, full checkpointing, full benchmarking.

Features:
  - Rich progress bars and live display
  - bf16/fp16 mixed precision auto-detected
  - Full GPU stats (VRAM alloc/reserved/total, utilization)
  - Rolling loss window (smoothed loss)
  - Throughput: tokens/sec, iterations/sec
  - ETA per epoch and full run
  - Step checkpoints (every N steps)
  - Timed checkpoints (every N minutes)
  - Best checkpoint tracking
  - Full CSV log every step
  - JSON run stats on completion
  - Gradient norm tracking
  - Per-epoch summary
  - Resume from checkpoint
  - Warmup + cosine LR schedule
  - Gradient clipping / weight decay
"""

import os
import csv
import json
import sys
import math
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn,
    TaskProgressColumn,
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.tokenizer import FMTokenizer
from data.preprocess import build_cache
from data.dataset   import FMDataset, collate_fn, FIELD_NAMES
from model.fm       import FM

console = Console()


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


# ── Live stats table ──────────────────────────────────────────────────────────

def make_stats_table(epoch, total_epochs, step, global_step,
                     loss, smooth_loss, best_loss, lr,
                     grad_norm, it_per_sec, tok_per_sec,
                     tokens_total, elapsed, eta_epoch, eta_run, g) -> Table:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column(style="dim", width=18)
    t.add_column(width=22)
    t.add_column(style="dim", width=14)
    t.add_column(width=20)

    vram_str = (f"{g['vram_alloc_gb']:.2f} / {g['vram_total_gb']:.1f} GB"
                if g else "—")
    util_str = f"{g['gpu_util_pct']}%" if g and g['gpu_util_pct'] >= 0 else "—"

    t.add_row("epoch",       f"[bold]{epoch}[/] / {total_epochs}",
              "step",        f"{global_step:,}")
    t.add_row("loss",        f"[bold yellow]{loss:.4f}[/]",
              "smooth",      f"{smooth_loss:.4f}")
    t.add_row("best loss",   f"[bold green]{best_loss:.4f}[/]",
              "lr",          f"{lr:.2e}")
    t.add_row("grad norm",   f"{grad_norm:.3f}",
              "it/s",        f"{it_per_sec:.1f}")
    t.add_row("tok/s",       fmt_num(int(tok_per_sec)),
              "tokens",      fmt_num(tokens_total))
    t.add_row("VRAM",        vram_str,
              "GPU util",    util_str)
    t.add_row("elapsed",     fmt_time(elapsed),
              "ETA epoch",   fmt_time(eta_epoch))
    t.add_row("ETA run",     fmt_time(eta_run), "", "")
    return t


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

    # ── Corpus: load manifest or preprocess ─────────────────────────────────
    import json as _json

    tok_path      = Path(args.out_dir) / 'tokenizer.pkl'
    manifest_path = Path(args.out_dir) / 'manifest.json'

    if tok_path.exists() and manifest_path.exists() and not args.retokenize:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console, transient=True) as p:
            p.add_task("Loading tokenizer...", total=None)
            tok = FMTokenizer.load(str(tok_path))
        with open(str(manifest_path)) as f:
            meta = _json.load(f)
        console.print(f"[green]✓[/] Tokenizer: [bold]{tok.vocab_size}[/] tokens | Corpus: [bold]{meta['n_sequences']}[/] sequences (bin)")
    else:
        midi_files = list(Path(args.midi_dir).rglob('*.mid'))
        n_workers  = min(args.workers or os.cpu_count(), 16)
        chunk_size = 500

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as p:
            task = p.add_task("Preprocessing corpus...", total=len(midi_files) * 2)
            tok, _ = build_cache(
                args.midi_dir, args.out_dir,
                min_seq_len=args.min_seq_len,
                workers=n_workers,
                chunk_size=chunk_size,
                progress_callback=lambda n: p.advance(task, advance=n),
            )

        tok.save(str(tok_path))
        with open(str(manifest_path)) as f:
            meta = _json.load(f)
        console.print(f"[green]✓[/] Vocab: [bold]{tok.vocab_size}[/] tokens | Corpus: [bold]{meta['n_sequences']}[/] sequences packed")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as p:
        p.add_task("Loading binaries into RAM...", total=None)

    dataset = FMDataset(args.out_dir, min_len=args.min_seq_len)
    console.print(f"[green]✓[/] Dataset: [bold]{len(dataset)}[/] sequences in pinned RAM")
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    console.print(f"[green]✓[/] Dataset: [bold]{len(dataset)}[/] sequences | [bold]{len(loader)}[/] batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FM(
        vocab_size       = tok.vocab_size,
        dim              = args.dim,
        n_decoder_layers = args.decoder_layers,
        decoder_hidden   = args.decoder_hidden,
        dropout          = args.dropout,
    ).to(device)
    model.idx_to_fields = tok.idx_to_fields

    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = args.epochs * len(loader)
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    out        = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path  = out / 'latest.pt'
    best_path  = out / 'best.pt'
    csv_path   = out / 'training_log.csv'
    stats_path = out / 'run_stats.json'

    config = {
        'vocab_size':     tok.vocab_size,
        'dim':            args.dim,
        'decoder_layers': args.decoder_layers,
        'decoder_hidden': args.decoder_hidden,
        'dropout':        args.dropout,
        'parameters':     model.count_parameters(),
        'pos_base':       32768,
    }
    with open(out / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_loss   = float('inf')

    if ckpt_path.exists() and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler'):
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        global_step = ckpt.get('step', 0)
        best_loss   = ckpt.get('loss', float('inf'))
        console.print(f"[green]✓[/] Resumed: epoch {start_epoch}, step {global_step}, loss {ckpt['loss']:.4f}")

    # ── Header panel ──────────────────────────────────────────────────────────
    g = gpu_stats()
    info = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    info.add_column(style="dim", width=18)
    info.add_column(width=30)
    info.add_column(style="dim", width=18)
    info.add_column(width=20)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    prec     = str(dtype).split('.')[-1] if dtype else 'fp32'
    info.add_row("device",     f"{device}  ({gpu_name})", "precision", prec)
    info.add_row("parameters", fmt_num(model.count_parameters()), "vocab size", str(tok.vocab_size))
    info.add_row("field dim",  str(args.dim), "decoder", f"{args.decoder_layers} × {args.decoder_hidden}")
    info.add_row("dataset",    f"{len(dataset)} sequences", "batch size", str(args.batch_size))
    info.add_row("epochs",     f"{args.epochs}  ({total_steps:,} steps)", "LR", f"{args.lr:.1e}")
    if g:
        info.add_row("VRAM",   f"{g['vram_alloc_gb']:.2f} / {g['vram_total_gb']:.1f} GB", "", "")
    console.print(Panel(info, title=f"[bold]FM — Field Machine Training[/]  {datetime.now():%Y-%m-%d %H:%M:%S}", border_style="blue"))

    # ── Training loop ─────────────────────────────────────────────────────────
    run_start       = time.time()
    last_timed_save = time.time()
    epoch_times     = []
    tokens_total    = 0
    recent_losses   = deque(maxlen=100)

    # Two-level progress: epochs (outer) + steps (inner)
    epoch_progress = Progress(
        TextColumn("[bold blue]Epoch {task.completed}/{task.total}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    step_progress = Progress(
        TextColumn("  [dim]step {task.fields[step]:>7}[/]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("[yellow]{task.fields[loss]:.4f}[/]"),
        TextColumn("[dim]{task.fields[its]:.1f} it/s[/]"),
        TimeRemainingColumn(),
        console=console,
    )

    epoch_task = epoch_progress.add_task("epochs", total=args.epochs - start_epoch)

    with Live(console=console, refresh_per_second=4) as live:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss       = 0.0
            epoch_start      = time.time()
            steps_this_epoch = 0
            epoch_tokens     = 0
            epoch_grad_norms = []
            loss_val         = 0.0
            grad_norm        = 0.0
            it_per_sec       = 0.0
            tok_per_sec_val  = 0.0
            smooth_loss      = 0.0

            step_task = step_progress.add_task(
                "steps", total=len(loader),
                step=global_step, loss=0.0, its=0.0
            )

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

                step_time       = time.time() - step_start
                loss_val        = loss.item()
                tokens_step     = int((batch['target'] != -100).sum().item())
                tok_per_sec_val = tokens_step / max(step_time, 1e-6)

                total_loss       += loss_val
                global_step      += 1
                steps_this_epoch += 1
                epoch_tokens     += tokens_step
                tokens_total     += tokens_step
                recent_losses.append(loss_val)
                epoch_grad_norms.append(float(grad_norm))

                avg_loss    = total_loss / steps_this_epoch
                smooth_loss = sum(recent_losses) / len(recent_losses)
                lr_now      = scheduler.get_last_lr()[0]
                elapsed_ep  = time.time() - epoch_start
                it_per_sec  = steps_this_epoch / max(elapsed_ep, 1e-6)
                eta_epoch   = (len(loader) - step - 1) / max(it_per_sec, 1e-6)
                run_elapsed = time.time() - run_start
                eta_run     = eta_epoch + (elapsed_ep / steps_this_epoch * len(loader)) * (args.epochs - epoch - 1)

                step_progress.update(
                    step_task,
                    advance=1,
                    step=global_step,
                    loss=avg_loss,
                    its=it_per_sec,
                )

                # Live stats panel every print_steps
                if global_step % args.print_steps == 0:
                    g = gpu_stats()
                    stats = make_stats_table(
                        epoch+1, args.epochs, step+1, global_step,
                        avg_loss, smooth_loss, best_loss, lr_now,
                        float(grad_norm), it_per_sec, tok_per_sec_val,
                        tokens_total, run_elapsed, eta_epoch, eta_run, g
                    )
                    live.update(Panel(
                        stats,
                        title=f"[bold]FM Training[/]",
                        border_style="blue",
                    ))

                # CSV log
                g = gpu_stats()
                write_csv(csv_path, {
                    'step':             global_step,
                    'epoch':            epoch + 1,
                    'loss':             round(loss_val, 6),
                    'avg_loss':         round(avg_loss, 6),
                    'smooth_loss':      round(smooth_loss, 6),
                    'grad_norm':        round(float(grad_norm), 4),
                    'lr':               round(lr_now, 8),
                    'it_per_sec':       round(it_per_sec, 3),
                    'tok_per_sec':      round(tok_per_sec_val, 0),
                    'tokens_total':     tokens_total,
                    'vram_alloc_gb':    g.get('vram_alloc_gb', ''),
                    'vram_reserved_gb': g.get('vram_reserved_gb', ''),
                    'gpu_util_pct':     g.get('gpu_util_pct', ''),
                    'elapsed_sec':      round(run_elapsed, 1),
                    'timestamp':        datetime.now().isoformat(),
                })

                # Step checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                    epoch, global_step, avg_loss, config)

                # Timed checkpoint
                if (time.time() - last_timed_save) >= args.save_minutes * 60:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    tp = out / f'timed_{ts}_step{global_step}.pt'
                    save_checkpoint(tp, model, optimizer, scheduler,
                                    epoch, global_step, avg_loss, config)
                    last_timed_save = time.time()

            # End of epoch
            step_progress.remove_task(step_task)

            avg_loss   = total_loss / len(loader)
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            avg_gnorm  = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0

            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                save_checkpoint(best_path, model, optimizer, scheduler,
                                epoch + 1, global_step, avg_loss, config)

            epoch_progress.advance(epoch_task)

            epochs_left  = args.epochs - epoch - 1
            avg_epoch_t  = sum(epoch_times[-3:]) / len(epoch_times[-3:])
            eta_finish   = datetime.now() + timedelta(seconds=avg_epoch_t * epochs_left)
            epoch_tok_ps = epoch_tokens / max(epoch_time, 1e-6)
            best_tag     = "  [bold green]★ new best[/]" if is_best else ""

            summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            summary.add_column(style="dim", width=18)
            summary.add_column(width=30)
            summary.add_row("loss",        f"[bold]{avg_loss:.4f}[/]  (best: [green]{best_loss:.4f}[/])")
            summary.add_row("grad norm",   f"{avg_gnorm:.4f}")
            summary.add_row("epoch time",  fmt_time(epoch_time))
            summary.add_row("total elapsed", fmt_time(time.time() - run_start))
            summary.add_row("tokens",      f"{fmt_num(tokens_total)}  ({fmt_num(epoch_tokens)} this epoch)")
            summary.add_row("tok/s",       f"{epoch_tok_ps/1000:.1f}k")
            summary.add_row("ETA finish",  eta_finish.strftime('%Y-%m-%d %H:%M:%S'))

            live.update(Panel(
                summary,
                title=f"[bold]Epoch {epoch+1}/{args.epochs} complete[/]{best_tag}",
                border_style="green" if is_best else "dim",
            ))

            save_checkpoint(ckpt_path, model, optimizer, scheduler,
                            epoch + 1, global_step, avg_loss, config)
            torch.save({
                'epoch':  epoch,
                'model':  model.state_dict(),
                'config': config,
                'loss':   avg_loss,
            }, out / f'epoch_{epoch+1:03d}_loss{avg_loss:.4f}.pt')

    # Final stats
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

    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    summary.add_column(style="dim", width=20)
    summary.add_column()
    summary.add_row("training time",  fmt_time(total_time))
    summary.add_row("total tokens",   fmt_num(tokens_total))
    summary.add_row("avg throughput", f"{tokens_total/total_time/1000:.1f}k tok/sec")
    summary.add_row("best loss",      f"[bold green]{best_loss:.4f}[/]")
    summary.add_row("stats",          str(stats_path))
    summary.add_row("log",            str(csv_path))
    console.print(Panel(summary, title="[bold green]Training complete[/]", border_style="green"))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mp.freeze_support()
    parser = argparse.ArgumentParser(description='FM — Field Machine Training')
    parser.add_argument('--midi_dir',       required=True)
    parser.add_argument('--out_dir',        default='checkpoints')
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=1)
    parser.add_argument('--dim',            type=int,   default=4096)
    parser.add_argument('--decoder_layers', type=int,   default=3)
    parser.add_argument('--decoder_hidden', type=int,   default=2048)
    parser.add_argument('--dropout',        type=float, default=0.1)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--weight_decay',   type=float, default=0.01)
    parser.add_argument('--grad_clip',      type=float, default=1.0)
    parser.add_argument('--warmup_steps',   type=int,   default=200)
    parser.add_argument('--min_seq_len',    type=int,   default=8)
    parser.add_argument('--workers',        type=int,   default=0)
    parser.add_argument('--save_steps',     type=int,   default=500)
    parser.add_argument('--save_minutes',   type=int,   default=30)
    parser.add_argument('--print_steps',    type=int,   default=10)
    parser.add_argument('--retokenize',     action='store_true')
    parser.add_argument('--fresh',          action='store_true')
    args = parser.parse_args()
    train(args)
