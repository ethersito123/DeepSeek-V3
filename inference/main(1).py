import torch
from core_model import ModelArgs, Transformer
from llm_io import load_checkpoint, SPTokenizer
from sampling import generate, reset_caches
from finance_agent import FinanceAgent

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(0)

# 1) Modelo
args = ModelArgs()          # para FP8: args.dtype="fp8"
model = Transformer(args).cuda()
# model = load_checkpoint(model, "ruta/finetune_finanzas.pt", strict=False)

# 2) Tokenizer real
# tok = SPTokenizer("ruta/tokenizer_finanzas.model")
# Demo temporal si aún no tienes SPM:
class MockTok:
    bos_id, eos_id, pad_id = 1, 2, 0
    def encode(self, s): return [self.bos_id] + [min(ord(c),255)+3 for c in s]
    def decode(self, ids): return "".join(chr(max(i-3,0)) for i in ids if i>=3)
tok = MockTok()

# 3) Inferencia básica
reset_caches(model)
txt = generate(model, tok, "Dame un resumen de mercado para hoy:", max_new_tokens=128, top_p=0.9, temperature=0.7)
print(">>", txt[:400])

# 4) Agente financiero
agent = FinanceAgent(model, tok)
report = agent.research_and_strategy(["AAPL","MSFT","TLT"])
print("Aprobadas:", report["approved"])
print("Crítica:", report["critique"][:400])
