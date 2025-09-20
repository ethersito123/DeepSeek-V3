# infer.py
import torch
from core_model import ModelArgs, Transformer
from llm_io import load_checkpoint, SPTokenizer
from sampling import generate

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")

args = ModelArgs(dtype="bf16")
model = Transformer(args).cuda()
load_checkpoint(model, "pretrain_step_010000.pt", strict=False)
tok = SPTokenizer("tokenizer_finanzas.model")

print(generate(model, tok, "Idea de estrategia factor value en mercados emergentes:"))
