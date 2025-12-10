import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

vision_attn_weights = []
text_attn_weights = []


def clear_attn_buffers():
    vision_attn_weights.clear()
    text_attn_weights.clear()


def _qkv_hook(mod, args, out):
    mod._cached_qkv_linear_out = out.detach()


def _vision_attn_hook(mod, args, kwargs, out):
    """
    Hook for Qwen3-VL VisionAttention.
    """
    with torch.no_grad():

        hidden_states = args[0]
        cu_seqlens = kwargs["cu_seqlens"]
        cos, sin = kwargs["position_embeddings"]

        qkv_lin = getattr(mod, "_cached_qkv_linear_out", None)
        if qkv_lin is None:
            qkv_lin = mod.qkv(hidden_states)
        qkv_lin = qkv_lin.to(hidden_states.dtype)

        L = hidden_states.shape[0]
        H = mod.num_heads
        D = mod.head_dim

        q, k, v = qkv_lin.view(L, 3, H, D).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(q, lengths, dim=2)
        k_splits = torch.split(k, lengths, dim=2)
        v_splits = torch.split(v, lengths, dim=2)

        layer_chunks = []
        scale = D**-0.5
        for q_i, k_i in zip(q_splits, k_splits):
            scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale
            probs = torch.softmax(scores, dim=-1)
            layer_chunks.append(probs.detach().cpu())
        vision_attn_weights.append(layer_chunks)


def _text_attn_hook(mod, args, kwargs, out):
    _, attn_weights = out
    if attn_weights is not None:
        text_attn_weights.append(attn_weights.detach().cpu())


def _internvl_vision_attn_hook(mod, args, kwargs, out):
    if isinstance(out, tuple) and len(out) == 2:
        _, attn_weights = out
        if attn_weights is not None:
            vision_attn_weights.append([attn_weights.detach().cpu()])
    else:
        vision_attn_weights.append([None])


def attach_internvl_vision_hooks(model):
    hooks = []
    stacks = []
    print("Attaching InternVL vision attention hooks...")
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder"):
        stacks.append(model.vision_model.encoder.layer)
    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "encoder"):
        stacks.append(model.vision_tower.encoder.layer)

    for layers in stacks:
        for blk in layers:
            attn = getattr(blk, "attention", None)
            if attn is not None:
                hooks.append(
                    attn.register_forward_hook(
                        _internvl_vision_attn_hook, with_kwargs=True
                    )
                )
    return hooks


def _internvl_text_attn_hook(mod, args, kwargs, out):
    if isinstance(out, tuple) and len(out) == 2:
        _, w = out
        if w is not None:
            text_attn_weights.append(w.detach().cpu())
    # print("InternVL atten weights length. ",len(text_attn_weights))


def attach_internvl_text_hooks(model):
    hooks = []
    layers = None
    # print("Attaching InternVL text attention hooks...")
    if hasattr(model, "language_model"):
        lm = model.language_model
        layers = getattr(getattr(lm, "model", lm), "layers", None)

    if layers is None:
        return hooks

    for layer in layers:
        attn = (
            getattr(layer, "self_attn", None)
            or getattr(layer, "attn", None)
            or getattr(layer, "attention", None)
        )
        if attn is not None:
            hooks.append(
                attn.register_forward_hook(_internvl_text_attn_hook, with_kwargs=True)
            )
    return hooks


def _internvl35_vision_attn_hook(mod, args, kwargs, out):
    # print("InternVL-3.5 vision attention hook called.")
    hidden_states = args[0]  # (B, N, C)
    B, N, C = hidden_states.shape
    H = mod.num_heads
    D = C // H

    qkv = mod.qkv(hidden_states)
    qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if getattr(mod, "qk_normalization", False):

        B_, H_, N_, D_ = q.shape
        q = (
            mod.q_norm(q.transpose(1, 2).flatten(-2, -1))
            .view(B_, N_, H_, D_)
            .transpose(1, 2)
        )
        k = (
            mod.k_norm(k.transpose(1, 2).flatten(-2, -1))
            .view(B_, N_, H_, D_)
            .transpose(1, 2)
        )

    scores = torch.matmul(q * mod.scale, k.transpose(-2, -1))
    attn = torch.softmax(scores, dim=-1)

    vision_attn_weights.append(attn.detach().cpu())
    # print("InternVL-3.5 vision attention hook called. ",vision_attn_weights)


def _gemma3_vision_attn_hook(mod, args, kwargs, out):
    # print("Gemma-3 vision attention hook called.")

    if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
        attn_weights = out[1].detach().cpu()
        vision_attn_weights.append(attn_weights)
        # print("Gemma-3 vision attention hook called. ",vision_attn_weights)
        return

    hidden_states = args[0]
    B, N, C = hidden_states.shape
    H = getattr(mod, "num_heads", None) or getattr(mod, "n_head", None)
    D = C // H

    q = mod.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
    k = mod.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

    if hasattr(mod, "q_norm"):
        q = mod.q_norm(q)
    if hasattr(mod, "k_norm"):
        k = mod.k_norm(k)

    scale = (D**-0.5) if hasattr(mod, "scale") else (D**-0.5)
    scores = torch.matmul(q * scale, k.transpose(-2, -1))  # (B,H,N,N)
    attn = torch.softmax(scores, dim=-1)
    vision_attn_weights.append(attn.detach().cpu())
    # print("Gemma-3 vision attention hook called. ",vision_attn_weights)
