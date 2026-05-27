"""
FM Dataset
==========
Loads flat binary files into pinned RAM once at startup.
Every __getitem__ is an instant tensor slice — zero disk I/O during training.

Binary layout (written by preprocess.py):
  tokens.bin   — flat int32 array, all token indices concatenated
  dna.bin      — flat float32 array, shape (total_tokens, 7) — 7 DNA fields
  offsets.bin  — int64 array, shape (n_sequences, 2) — (start, length) per sequence
  manifest.json — metadata
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

DNA_FIELDS = ['pitch_class', 'octave', 'log_duration', 'beat_sin', 'beat_cos', 'velocity', 'voice']
INT_DNA    = {0}   # pitch_class index in DNA_FIELDS
N_DNA      = 7
FIELD_NAMES = DNA_FIELDS  # alias for train.py compatibility


class FMDataset(Dataset):
    def __init__(self, data_dir_or_sequences, min_len: int = 8):
        # Legacy: list of dicts in RAM
        if isinstance(data_dir_or_sequences, list):
            self._legacy_init(data_dir_or_sequences, min_len)
            return

        data_dir = Path(data_dir_or_sequences)
        tokens_bin  = data_dir / 'tokens.bin'
        dna_bin     = data_dir / 'dna.bin'
        offsets_bin = data_dir / 'offsets.bin'

        with open(str(data_dir / 'manifest.json')) as f:
            meta = json.load(f)

        total = meta['total_tokens']

        # Use memmap — don't load entire dataset into RAM
        offsets_raw = np.fromfile(str(offsets_bin), dtype=np.int64).reshape(-1, 2)
        self.tokens  = np.memmap(str(tokens_bin),  dtype=np.int32,   mode='r')
        self.dna     = np.memmap(str(dna_bin),     dtype=np.float32, mode='r').reshape(total, N_DNA)
        self.offsets = [(int(offsets_raw[i, 0]), int(offsets_raw[i, 1]))
                        for i in range(len(offsets_raw))
                        if int(offsets_raw[i, 1]) >= min_len]
        self._mode   = 'bin'

    def _legacy_init(self, sequences, min_len):
        self.sequences = [s for s in sequences if len(s['token_idx']) >= min_len]
        self._mode     = 'legacy'

    def __len__(self):
        if self._mode == 'bin':
            return len(self.offsets)
        return len(self.sequences)

    def __getitem__(self, idx):
        if self._mode == 'bin':
            return self._getitem_bin(idx)
        return self._getitem_legacy(idx)

    def _getitem_bin(self, idx):
        start, length = self.offsets[idx]
        toks = torch.from_numpy(np.array(self.tokens[start:start+length], dtype=np.int64))
        dna  = torch.from_numpy(np.array(self.dna[start:start+length],    dtype=np.float32))

        item = {}
        for i, field in enumerate(DNA_FIELDS):
            vals = dna[:-1, i]
            if i in INT_DNA:
                item[field] = vals.long()
            else:
                item[field] = vals.float()

        item['voice']  = dna[:-1, 6].long()
        item['target'] = toks[1:]
        return item

    def _getitem_legacy(self, idx):
        seq  = self.sequences[idx]
        item = {}
        int_fields = {'pitch_class', 'voice'}
        for field in DNA_FIELDS:
            vals = seq[field][:-1]
            if field in int_fields:
                item[field] = torch.tensor(vals, dtype=torch.long)
            else:
                item[field] = torch.tensor(vals, dtype=torch.float32)
        item['target'] = torch.tensor(seq['token_idx'][1:], dtype=torch.long)
        return item


def collate_fn(batch):
    if len(batch) == 1:
        return {k: v.unsqueeze(0) for k, v in batch[0].items()}

    max_len = max(item['target'].shape[0] for item in batch)
    out = {field: [] for field in DNA_FIELDS}
    out['target'] = []

    int_fields = {'pitch_class', 'voice'}
    for item in batch:
        T   = item['target'].shape[0]
        pad = max_len - T
        for field in DNA_FIELDS:
            t = item[field]
            if pad > 0:
                pad_val = 0 if field in int_fields else 0.0
                t = torch.cat([t, torch.full((pad,), pad_val, dtype=t.dtype)])
            out[field].append(t)
        tgt = item['target']
        if pad > 0:
            tgt = torch.cat([tgt, torch.full((pad,), -100, dtype=torch.long)])
        out['target'].append(tgt)

    return {k: torch.stack(v) for k, v in out.items()}
