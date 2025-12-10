import torch


def _maybe_avg_heads(x: torch.Tensor, avg=True):
    # x is (H, Lq, Lk) or (B,H,L,L) or (1,H,Lq,Lk)
    if x.dim() == 4:
        # (B,H,L,L) or (1,H,Lq,Lk)
        if x.size(0) == 1:
            x = x.squeeze(0)  # (H, Lq, Lk)
        else:
            x = x.mean(0)  # (H, L, L)
    # average heads if asked
    return x.mean(0) if (avg and x.dim() == 3) else x


def _pack_vision_maps(vision_attn_weights, avg_heads=True):
    """
    vision_attn_weights: list[num_layers] of list[num_chunks] of (1,H,Lq,Lk)
    returns list of dicts: {"name": "vision.L{li}.C{ci}", "type":"vision", "map": (Lq,Lk) or (H,Lq,Lk)}
    """
    out = []
    for li, chunks in enumerate(vision_attn_weights):
        for ci, chunk in enumerate(chunks):
            # chunk: (1,H,Lq,Lk)
            m = chunk  # keep on CPU as you already do
            m = _maybe_avg_heads(m, avg=avg_heads)  # -> (Lq,Lk) or (H,Lq,Lk)
            out.append({"name": f"vision.L{li}.C{ci}", "type": "vision", "map": m})
    return out


def _slice_text_modal_maps(A: torch.Tensor, is_vision: torch.Tensor, avg_heads=True):
    """
    A: (B,H,L,L) LLM attention (one layer). Usually B=1.
    is_vision: (B,L) bool/int mask where 1 == vision token, 0 == text token.
    Returns dict of four maps (each averaged over heads if requested).
    """
    if A.size(0) != is_vision.size(0):
        raise ValueError("Batch mismatch for text attention and mask")
    B = A.size(0)
    out = []
    for b in range(B):
        H, L, _ = A[b].shape
        txt_idx = (~is_vision[b].bool()).nonzero(as_tuple=False).squeeze(1)
        vis_idx = (is_vision[b].bool()).nonzero(as_tuple=False).squeeze(1)

        # (H, Lt, Lv) etc
        TT = A[b][:, txt_idx][:, :, txt_idx]
        VV = A[b][:, vis_idx][:, :, vis_idx]
        TV = A[b][:, txt_idx][:, :, vis_idx]  # text -> vision
        VT = A[b][:, vis_idx][:, :, txt_idx]  # vision -> text

        TT = _maybe_avg_heads(TT, avg=avg_heads)
        VV = _maybe_avg_heads(VV, avg=avg_heads)
        TV = _maybe_avg_heads(TV, avg=avg_heads)
        VT = _maybe_avg_heads(VT, avg=avg_heads)

        out.append({"TT": TT, "VV": VV, "TV": TV, "VT": VT})

    return out[0] if len(out) == 1 else out


def _pack_text_maps(text_attn_weights, mm_token_type_ids=None, avg_heads=True):
    """
    text_attn_weights: list[num_layers] of (B,H,L,L) tensors
    mm_token_type_ids: (B,L) with 1 for vision tokens; if provided, we emit cross-modal slices as separate named entries.
    returns list of dicts with name/type/map
    """
    out = []
    for li, layer_map in enumerate(text_attn_weights):

        full = _maybe_avg_heads(
            layer_map, avg=avg_heads
        )  # -> (B,L,L) or (H,L,L) or (L,L)
        out.append({"name": f"text.L{li}.full", "type": "text", "map": full})

        if mm_token_type_ids is not None:
            slices = _slice_text_modal_maps(
                layer_map, mm_token_type_ids, avg_heads=avg_heads
            )

            if isinstance(slices, dict):
                for key, mat in slices.items():

                    kind = {
                        "TT": "text↔text",
                        "VV": "vision↔vision",
                        "TV": "text→vision",
                        "VT": "vision→text",
                    }[key]
                    out.append({"name": f"text.L{li}.{key}", "type": kind, "map": mat})
            else:

                for b, sl in enumerate(slices):
                    for key, mat in sl.items():
                        kind = {
                            "TT": "text↔text",
                            "VV": "vision↔vision",
                            "TV": "text→vision",
                            "VT": "vision→text",
                        }[key]
                        out.append(
                            {"name": f"text.L{li}.B{b}.{key}", "type": kind, "map": mat}
                        )

    return out


def package_attention_run(
    vision_attn_weights, text_attn_weights, mm_token_type_ids=None, avg_heads=True
):
    """
    Returns a flat list: [{name, type, map}, ...] suitable for compare_attention_runs.
    """
    items = []
    items += _pack_vision_maps(vision_attn_weights, avg_heads=avg_heads)
    items += _pack_text_maps(
        text_attn_weights, mm_token_type_ids=mm_token_type_ids, avg_heads=avg_heads
    )
    return items
