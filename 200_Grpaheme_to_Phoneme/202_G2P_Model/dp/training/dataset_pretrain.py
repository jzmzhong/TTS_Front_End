from pathlib import Path
from random import Random
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from dp.utils.io import unpickle_binary


class PhonemizerDatasetPretrain(Dataset):

    def __init__(self,
                 items: List[Tuple[int, List[int], List[int]]],
                 unk_idx, non_special_indices, mask_ratio, mask_UNK, mask_random, mask_original) -> None:
        super().__init__()
        self.items = items
        self.unk_idx = unk_idx
        self.non_special_indices = non_special_indices
        self.mask_ratio = mask_ratio
        self.mask_UNK_ratio = mask_UNK
        self.mask_random_ratio = mask_random
        self.mask_original_ratio = mask_original

        # Implement Masking
        # text: [<LANG>, <Gi> * char_repeats, <END>]
        if self.mask_ratio == "EVALUATE":
            new_items = []
            for language, text, phonemes in items:
                text = torch.tensor(text, dtype=torch.long)
                phonemes = torch.tensor(phonemes, dtype=torch.long)
                char_len = text.size(0) - 2
                texts = torch.cat([text] * char_len, dim=-1).reshape(char_len, char_len + 2)
                mask = torch.cat((torch.zeros(char_len, 1), torch.ones(char_len).diag(), torch.zeros(char_len, 1)), dim=-1)
                new_texts = texts * (1 - mask) + mask * unk_idx
                for new_text in new_texts:
                    new_items.append((language, new_text.long(), phonemes))
            self.items = new_items
        else:
            new_items = []
            for language, text, phonemes in items:
                text = torch.tensor(text, dtype=torch.long)
                phonemes = torch.tensor(phonemes, dtype=torch.long)
                char_len = text.size(0) - 2
                mask_rand = torch.rand(char_len)
                mask_UNK = torch.lt(mask_rand, torch.ones(char_len) * self.mask_UNK_ratio * self.mask_ratio).int()
                mask_random = torch.lt(mask_rand, torch.ones(char_len) * (self.mask_UNK_ratio + self.mask_random_ratio) * self.mask_ratio).int() - mask_UNK
                mask_original = torch.lt(mask_rand, torch.ones(char_len) * (self.mask_UNK_ratio + self.mask_random_ratio + self.mask_original_ratio) * self.mask_ratio).int() - mask_UNK - mask_random
                new_text = (1 - mask_UNK - mask_random) * text[1:-1] + mask_UNK * unk_idx
                rand_token_indices = (torch.rand(char_len) * len(non_special_indices)).trunc().int()
                rand_token_indices = torch.tensor([self.non_special_indices[i] for i in rand_token_indices])
                new_text += mask_random * rand_token_indices
                new_text = torch.cat((text[:1], new_text, text[-1:]), dim=0)
                new_items.append((language, new_text.long(), phonemes))
            self.items = new_items

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        language, text, phonemes = item
        return {'item_id': index, 'text': text,
                'phonemes': phonemes, 'language': language,
                'text_len': text.size(0), 'phonemes_len': phonemes.size(0),
                'start_index': phonemes[0]}

    def __len__(self):
        return len(self.items)


# From https://github.com/fatchord/WaveRNN/blob/master/utils/dataset.py
class BinnedLengthSampler(Sampler):

    def __init__(self, phoneme_lens: List[int], batch_size: int, bin_size: int, seed=42) -> None:
        _, self.idx = torch.sort(torch.tensor(phoneme_lens))
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.random = Random(seed)
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        idx = self.idx.numpy()
        bins = []
        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            self.random.shuffle(this_bin)
            bins += [this_bin]
        self.random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)
        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            self.random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])
        return iter(torch.Tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)


def collate_dataset(batch: List[dict]) -> Dict[str, torch.Tensor]:
    lang = [b['language'] for b in batch]
    lang = torch.tensor(lang).long()
    text = [b['text'] for b in batch]
    text = pad_sequence(text, batch_first=True, padding_value=0)
    text_len = torch.tensor([b['text_len'] for b in batch]).long()
    phonemes = [b['phonemes'] for b in batch]
    phonemes = pad_sequence(phonemes, batch_first=True, padding_value=0)
    phonemes_len = torch.tensor([b['phonemes_len'] for b in batch]).long()
    item_ids = [b['item_id'] for b in batch]
    item_ids = torch.tensor(item_ids).long()
    start_index = [b['start_index'] for b in batch]
    start_index = torch.tensor(start_index).long()
    return {'text': text, 'phonemes': phonemes, 'text_len': text_len,
            'phonemes_len': phonemes_len, 'item_id': item_ids, 'language': lang,
            'start_index': start_index}


def new_dataloader_pretrain(dataset_file: Path,
                   batch_size=32,
                   drop_last=False,
                   use_binning=True,
                   unk_idx=3,
                   non_special_indices={},
                   mask_ratio=0.2,
                   mask_UNK=0.8,
                   mask_random=0.1,
                   mask_original=0.1) -> DataLoader:
    dataset = unpickle_binary(dataset_file)
    phonemizer_dataset = PhonemizerDatasetPretrain(dataset,
                                    unk_idx=unk_idx,
                                    non_special_indices=non_special_indices,
                                    mask_ratio=mask_ratio,
                                    mask_UNK=mask_UNK,
                                    mask_random=mask_random,
                                    mask_original=mask_original)
    phoneme_lens = [len(p) for _, _, p in phonemizer_dataset.items]
    if use_binning:
        sampler = BinnedLengthSampler(phoneme_lens=phoneme_lens,
                                      batch_size=batch_size,
                                      bin_size=batch_size*3)
    else:
        sampler = None
    return DataLoader(phonemizer_dataset,
                      collate_fn=collate_dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      num_workers=0,
                      shuffle=False,
                      drop_last=drop_last,
                      pin_memory=True)