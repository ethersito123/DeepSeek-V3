# llm_io.py
import sentencepiece as spm
import torch

class SPTokenizer:
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        # ids útiles
        self.pad_id  = self.sp.pad_id() if self.sp.pad_id() != -1 else 0
        self.unk_id  = self.sp.unk_id()
        self.bos_id  = self.sp.bos_id()
        self.eos_id  = self.sp.eos_id()
    def encode(self, text: str, add_bos=False, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos and self.bos_id != -1:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id != -1:
            ids = ids + [self.eos_id]
        return ids
    def decode(self, ids):
        # filtra pads
        ids = [int(i) for i in ids if i != self.pad_id]
        return self.sp.decode(ids)

def train_sentencepiece(corpus_txt: str, model_prefix="tokenizer_finanzas", vocab_size=102400):
    spm.SentencePieceTrainer.Train(
        input=f"--input={corpus_txt} "
              f"--model_prefix={model_prefix} "
              f"--vocab_size={vocab_size} "
              f"--character_coverage=0.9995 --model_type=bpe "
              f"--byte_fallback=true --unk_piece=<unk> "
              f"--bos_piece=<s> --eos_piece=</s> --pad_piece=<pad>"
    )

def load_checkpoint(model: torch.nn.Module, path: str, strict: bool = False):
    sd = torch.load(path, map_location="cuda")
    # soporta { 'model': state_dict } o state_dict directo
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=strict)
    return model
