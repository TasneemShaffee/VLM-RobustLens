import torch

def _maybe_avg_heads(x: torch.Tensor, avg=True):
  
    if x.dim() == 4:
       
        if x.size(0) == 1:
            x = x.squeeze(0) 
        else:
            x = x.mean(0)   
   
    return x.mean(0) if (avg and x.dim() == 3) else x

def _pack_vision_maps(vision_attn_weights, avg_heads=True):
    
    out = []
    for li, chunks in enumerate(vision_attn_weights):
        for ci, chunk in enumerate(chunks):
         
            m = chunk  
            m = _maybe_avg_heads(m, avg=avg_heads) 
            out.append({"name": f"vision.L{li}.C{ci}", "type": "vision", "map": m})
    return out


def _slice_text_modal_maps(A: torch.Tensor, is_vision: torch.Tensor, avg_heads=True):
 
    if A.size(0) != is_vision.size(0):
        raise ValueError("Batch mismatch for text attention and mask")
    B = A.size(0)
    out = []
    for b in range(B):
        H, L, _ = A[b].shape
        txt_idx = (~is_vision[b].bool()).nonzero(as_tuple=False).squeeze(1)
        vis_idx = ( is_vision[b].bool()).nonzero(as_tuple=False).squeeze(1)

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
   
    out = []
    for li, layer_map in enumerate(text_attn_weights):
       
        full = _maybe_avg_heads(layer_map, avg=avg_heads)  # -> (B,L,L) or (H,L,L) or (L,L)
        out.append({"name": f"text.L{li}.full", "type": "text", "map": full})
       
        if mm_token_type_ids is not None:
            slices = _slice_text_modal_maps(layer_map, mm_token_type_ids, avg_heads=avg_heads)

            if isinstance(slices, dict):
                for key, mat in slices.items():
                 
                    kind = {"TT":"text2text","VV":"vision2vision","TV":"text2vision","VT":"vision2text"}[key]
                    out.append({"name": f"text.L{li}.{key}", "type": kind, "map": mat})
            else:
               
                for b, sl in enumerate(slices):
                    for key, mat in sl.items():
                        kind = {"TT":"text2text","VV":"vision2vision","TV":"text2vision","VT":"vision2text"}[key]
                        out.append({"name": f"text.L{li}.B{b}.{key}", "type": kind, "map": mat})
   
    return out

def _pack_text_block_maps(text_attn_blocks, avg_heads=True):
   
    out = []
    block_meta = {
        "t2t": ("text2text",   "TT"),
        "t2v": ("text2vision", "TV"),
        "v2t": ("vision2text", "VT"),
        "v2v": ("vision2vision","VV"),
    }

    for li, blocks in enumerate(text_attn_blocks):
        if not isinstance(blocks, dict):
         
            continue

        for key, (kind, suffix) in block_meta.items():
            mat = blocks.get(key, None)
            if mat is None:
                continue

  
            mat = _maybe_avg_heads(mat, avg=avg_heads)
            out.append({
                "name": f"text.L{li}.{suffix}", 
                "type": kind,                    
                "map":  mat,
            })

    return out
def package_attention_run(vision_attn_weights, text_attn_weights, mm_token_type_ids=None, avg_heads=True,attn_mode="full"):
    
    items = []
    items += _pack_vision_maps(vision_attn_weights, avg_heads=avg_heads)
    if attn_mode=="blocks":
        items += _pack_text_block_maps(text_attn_weights, avg_heads=avg_heads)
    else:
        items += _pack_text_maps(text_attn_weights, mm_token_type_ids=mm_token_type_ids, avg_heads=avg_heads)
    return items