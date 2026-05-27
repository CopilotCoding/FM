"""
FM Dataset
==========
Full-file sequences. No padding. No windowing. No information loss.

Default batch size is 1 — one complete file per forward pass.
The cumsum within a single sequence is already fully parallel on GPU.
Batching multiple sequences together requires padding to the longest,
which causes OOM on long files. Batch=1 eliminates this entirely.

If batch_size > 1 is used, sequences are padded to longest in batch.
This is supported but not recommended for corpora with variable-length files.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict


FIELD_NAMES = [
    'pitch_class', 'octave', 'log_duration',
    'beat_sin', 'beat_cos', 'velocity', 'voice'
]

INT_FIELDS   = {'pitch_class', 'voice'}
FLOAT_FIELDS = {'octave', 'log_duration', 'beat_sin', 'beat_cos', 'velocity'}


class FMDataset(Dataset):
    def __init__(self, sequences: List[Dict], min_len: int = 8):
        self.sequences = [s for s in sequences
                          if len(s['token_idx']) >= min_len]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        item = {}
        for field in FIELD_NAMES:
            vals = seq[field][:-1]   # input: all but last
            if field in INT_FIELDS:
                item[field] = torch.tensor(vals, dtype=torch.long)
            else:
                item[field] = torch.tensor(vals, dtype=torch.float32)

        item['target'] = torch.tensor(
            seq['token_idx'][1:], dtype=torch.long)   # target: all but first
        return item


def collate_fn(batch):
    """
    Pad variable-length sequences to longest in batch.
    Only relevant when batch_size > 1.
    """
    if len(batch) == 1:
        return {k: v.unsqueeze(0) for k, v in batch[0].items()}

    max_len = max(item['target'].shape[0] for item in batch)
    out = {field: [] for field in FIELD_NAMES}
    out['target'] = []

    for item in batch:
        T   = item['target'].shape[0]
        pad = max_len - T

        for field in FIELD_NAMES:
            t = item[field]
            if pad > 0:
                pad_val = 0 if field in INT_FIELDS else 0.0
                t = torch.cat([t, torch.full((pad,), pad_val, dtype=t.dtype)])
            out[field].append(t)

        tgt = item['target']
        if pad > 0:
            tgt = torch.cat([tgt, torch.full((pad,), -100, dtype=torch.long)])
        out['target'].append(tgt)

    return {k: torch.stack(v) for k, v in out.items()}
