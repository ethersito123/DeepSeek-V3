# data.py
import torch, random
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, path_txt, tokenizer, seq_len=1024, repeat=1):
        with open(path_txt, "r", encoding="utf-8") as f:
            text = f.read()
        self.tok = tokenizer
        self.ids = self.tok.encode(text)
        self.seq_len = seq_len
        self.repeat = repeat
    def __len__(self):
        return self.repeat * max(1, len(self.ids)//self.seq_len - 1)
    def __getitem__(self, idx):
        start = random.randint(0, max(0, len(self.ids)-self.seq_len-2))
        x = self.ids[start:start+self.seq_len]
        # opcional: añade BOS al inicio
        if self.tok.bos_id != -1:
            x = [self.tok.bos_id] + x
            x = x[:self.seq_len]
        return torch.tensor(x, dtype=torch.long)

def collate(batch, pad_id=0):
    B = len(batch)
    T = max(x.size(0) for x in batch)
    out = torch.full((B, T), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, :x.size(0)] = x
    labels = out.clone()
    labels[:, :-1] = out[:, 1:]
    labels[:, -1] = pad_id
    return out, labels
