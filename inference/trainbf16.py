# train_bf16.py
import torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from llm_io import SPTokenizer, load_checkpoint
from data import TextDataset, collate
from core_model import ModelArgs, Transformer  # copia tu código del modelo como core_model.py

@torch.no_grad()
def reset_caches(model: torch.nn.Module):
    # Pone a cero los caches registrados como buffers en cada MLA
    for m in model.modules():
        if hasattr(m, "k_cache"):   m.k_cache.zero_()
        if hasattr(m, "v_cache"):   m.v_cache.zero_()
        if hasattr(m, "kv_cache"):  m.kv_cache.zero_()
        if hasattr(m, "pe_cache"):  m.pe_cache.zero_()

def init_weights(m):
    if isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def forward_all_steps_streaming(model, tokens):
    """
    tokens: (B,T) -> logits_full: (B,T,V)
    ejecuta prefill+streaming con start_pos creciente y caches internas
    """
    B, T = tokens.size()
    V = None
    logits_steps = []
    start_pos = 0
    # prefill primer token
    l = model(tokens[:, 0:1], start_pos=start_pos)
    logits_steps.append(l)  # (B,V)
    start_pos += 1
    # resto de pasos
    for t in range(1, T):
        l = model(tokens[:, t:t+1], start_pos=start_pos)  # usa cache
        logits_steps.append(l)
        start_pos += 1
        if V is None:
            V = l.size(-1)
    # apila a (B,T,V)
    return torch.stack(logits_steps, dim=1)  # (B,T,V)

def train():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    tok = SPTokenizer("tokenizer_finanzas.model")
    args = ModelArgs()  # dtype="bf16" por defecto
    model = Transformer(args).cuda()
    model.apply(init_weights)
    model.train()

    ds = TextDataset("corpus_finanzas.txt", tok, seq_len=1024, repeat=2)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2,
                    collate_fn=lambda b: collate(b, pad_id=tok.pad_id))

    opt = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    step, save_every = 0, 1000
    for epoch in range(10_000):  # o los que quieras
        for tokens, labels in dl:
            tokens = tokens.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            reset_caches(model)  # importante: caches vacíos por batch
            logits = forward_all_steps_streaming(model, tokens)  # (B,T,V)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=tok.pad_id
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            if step % 50 == 0:
                print(f"[step {step}] loss={loss.item():.4f}")
            if step % save_every == 0:
                torch.save({"model": model.state_dict(), "step": step}, f"pretrain_step_{step:06d}.pt")
                print("Guardado: pretrain_step_%06d.pt" % step)

if __name__ == "__main__":
    train()
