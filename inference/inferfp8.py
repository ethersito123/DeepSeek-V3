# infer_fp8.py
import torch
from core_model import ModelArgs, Transformer
from llm_io import SPTokenizer
from sampling import generate

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")

args = ModelArgs(dtype="fp8")
model = Transformer(args).cuda()
sd = torch.load("pretrain_step_010000_fp8.pt", map_location="cuda")
model.load_state_dict(sd, strict=True)

tok = SPTokenizer("tokenizer_finanzas.model")
print(generate(model, tok, "Propuesta de estrategia intradía con microestructura:"))
