import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

vision_attn_weights = []   
text_attn_weights   = []   
text_attn_blocks    = []
def clear_attn_buffers():
    vision_attn_weights.clear()
    text_attn_weights.clear()

def _qkv_hook(mod, args, out):
    mod._cached_qkv_linear_out = out.detach()


def _vision_attn_hook(mod, args, kwargs, out):
   
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
        scale = D ** -0.5
        for q_i, k_i in zip(q_splits, k_splits):
            scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale  
            probs = torch.softmax(scores, dim=-1)                     
            layer_chunks.append(probs.detach().cpu())
        vision_attn_weights.append(layer_chunks)

        # =========================
        #   uncomment below for  verification 
        # =========================
       
        """try:
            recon_chunks = []
            for probs_i, v_i in zip(layer_chunks, v_splits):
                p = probs_i.to(v_i.device, v_i.dtype)        
                o = torch.matmul(p, v_i)                     
                recon_chunks.append(o)
            recon = torch.cat(recon_chunks, dim=2)           
            recon = recon.transpose(1, 2).reshape(1, -1, H*D).squeeze(0)  

           
            recon_proj = mod.proj(recon)                     

         
            diff = (recon_proj - out).abs()
            max_err = float(diff.max().detach().cpu())
            mean_err = float(diff.mean().detach().cpu())
            #print(f"[VisionAttn check] max|Δ|={max_err:.3e}  mean|Δ|={mean_err:.3e}")
        except Exception as e:
          
            print(f"[VisionAttn check] verification skipped due to: {e}")"""


def make_text_attn_hook(runner):
    """
    Generic text-attention hook for Qwen3, Gemma3, etc.
    """
    def _text_attn_hook(mod, args, kwargs, out):

        if isinstance(out, tuple) and len(out) == 2:
            attn_output, attn_weights = out
        else:
            attn_weights = None

        if attn_weights is None:
            return

      
        text_attn_weights.append(attn_weights.detach().cpu())

        
        tm = getattr(runner, "last_text_mask", None)
        vm = getattr(runner, "last_vision_mask", None)
        #print("tm vm ",tm,vm)
        if tm is None or vm is None:
            return

    
        tm = tm.to(attn_weights.device)
        vm = vm.to(attn_weights.device)
        #print("tm device ",tm)
   
        if tm.dim() == 1:
            tm = tm.unsqueeze(0)
            vm = vm.unsqueeze(0)

  
        blocks = split_text_vision_attn(attn_weights, tm, vm, reduce=False)

        layer_idx = getattr(mod, "layer_idx", len(text_attn_blocks))
        text_attn_blocks.append({
            "name": f"text.L{layer_idx}",
            **blocks,
        })
    
    return _text_attn_hook

def _internvl_vision_attn_hook(mod, args, kwargs, out):
   
    if isinstance(out, tuple) and len(out) == 2:
        _, attn_weights = out
        if attn_weights is not None:
            vision_attn_weights.append([attn_weights.detach().cpu()])
    else:
        vision_attn_weights.append([None])  
    #print("InternVL vision attention hook called. ",vision_attn_weights)
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
                hooks.append(attn.register_forward_hook(internvl_vision_attn_hook, with_kwargs=True))
    return hooks


def make_internvl_text_attn_hook(runner):
    def _internvl_text_attn_hook(mod, args, kwargs, out):

        if isinstance(out, tuple) and len(out) == 2:
            attn_output, w = out
        else:
            w = None

        if w is None:
            return

     
        text_attn_weights.append(w.detach().cpu())

      
        tm = getattr(runner, "last_text_mask", None)
        vm = getattr(runner, "last_vision_mask", None)
        if tm is None or vm is None:
            return

      
        tm = tm.to(w.device)
        vm = vm.to(w.device)

        if tm.dim() == 1:
            tm = tm.unsqueeze(0)
            vm = vm.unsqueeze(0)

        blocks = split_text_vision_attn(w, tm, vm, reduce=False)

        layer_idx = getattr(mod, "layer_idx", len(text_attn_blocks))
        text_attn_blocks.append({
            "name": f"text.L{layer_idx}",
            **blocks,
        })
        #print("Text attention hook called. ",len(text_attn_blocks))
    return _internvl_text_attn_hook

def attach_internvl_text_hooks(model):
    hooks = []
    layers = None
    #print("Attaching InternVL text attention hooks...")
    if hasattr(model, "language_model"):
        lm = model.language_model
        layers = getattr(getattr(lm, "model", lm), "layers", None)

    if layers is None:
        return hooks

    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
        if attn is not None:
            hooks.append(attn.register_forward_hook(internvl_text_attn_hook, with_kwargs=True))
    return hooks

def _internvl35_vision_attn_hook(mod, args, kwargs, out):
    #print("InternVL-3.5 vision attention hook called.")
    hidden_states = args[0]                         
    B, N, C = hidden_states.shape
    H = mod.num_heads
    D = C // H


    qkv = mod.qkv(hidden_states)                    
    qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)  
    q, k, v = qkv.unbind(0)                         

  
    if getattr(mod, "qk_normalization", False):

        B_, H_, N_, D_ = q.shape
        q = mod.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        k = mod.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)


    scores = torch.matmul(q * mod.scale, k.transpose(-2, -1)) 
    attn   = torch.softmax(scores, dim=-1)                     

    vision_attn_weights.append(attn.detach().cpu()) 
    #print("InternVL-3.5 vision attention hook called. ",vision_attn_weights)   

"""
   # =========================
        # un comment below for verification 
    # =========================
def _internvl35_vision_attn_hook(mod, args, kwargs, out):
    try:
        # If flash attention is active, we can’t reconstruct weights the same way.
        if getattr(mod, "use_flash_attn", False):
            # Uncomment if you want a reminder:
            # print("[InternVL-3.5] FlashAttention is ON; skip verification.")
            return

        hidden_states = args[0]   # (B, N, C) *already* normed by the layer before calling mod.attn
        B, N, C = hidden_states.shape
        H = mod.num_heads
        D = C // H

        # Use cached qkv from the same forward if available; otherwise recompute
        qkv = getattr(mod.qkv, "_cached_qkv_out", None)
        if qkv is None:
            qkv = mod.qkv(hidden_states)                  # (B, N, 3*C)
        qkv = qkv.to(hidden_states.dtype)

        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)                                  # each (B, H, N, D)

        # q/k RMSNorm if enabled (same as module)
        if getattr(mod, "qk_normalization", False):
            B_, H_, N_, D_ = q.shape
            q = mod.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = mod.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        # Reconstruct attention weights and context
        scores = torch.matmul(q * mod.scale, k.transpose(-2, -1))    # (B, H, N, N)
        attn   = torch.softmax(scores, dim=-1)                        # (B, H, N, N)
        ctx    = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Final projection stack (dropout is no-op in eval)
        proj_out = mod.proj(ctx)
        proj_out = mod.proj_drop(proj_out)

        # Module output (InternAttention.forward returns (B, N, C))
        module_out = out

        # Tolerances by dtype
        if proj_out.dtype in (torch.bfloat16, torch.float16):
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-4, 1e-5

        # Print verification diagnostics
        if not torch.allclose(proj_out, module_out, rtol=rtol, atol=atol):
            diff = (proj_out - module_out).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            denom = module_out.abs().clamp_min(1e-8)
            max_rel = (diff / denom).max().item()
            print(f"[InternVL-3.5 verify] MISMATCH: max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, "
                  f"max_rel={max_rel:.4e} (rtol={rtol}, atol={atol})")
        else:
            print("[InternVL-3.5 verify] OK: proj(attn@V) matches module output.")

        # Save weights for your analysis (keep on CPU to save VRAM)
        vision_attn_weights.append(attn.detach().cpu())

    except Exception as e:
        # Don’t break the forward pass; just report
        print(f"[InternVL-3.5 verify] hook error: {e}")    
"""        

def _gemma3_vision_attn_hook(mod, args, kwargs, out):
   
    #print("Gemma-3 vision attention hook called.")

    if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
        attn_weights = out[1].detach().cpu()
        vision_attn_weights.append(attn_weights)
        #print("Gemma-3 vision attention hook called. ",vision_attn_weights)
        return

 
    hidden_states = args[0]  
    B, N, C = hidden_states.shape
    H = getattr(mod, "num_heads", None) or getattr(mod, "n_head", None)
    D = C // H

   
    q = mod.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)   
    k = mod.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)   

    if hasattr(mod, "q_norm"): q = mod.q_norm(q)
    if hasattr(mod, "k_norm"): k = mod.k_norm(k)

    scale = (D ** -0.5) if hasattr(mod, "scale") else (D ** -0.5)
    scores = torch.matmul(q * scale, k.transpose(-2, -1))            # (B,H,N,N)
    attn = torch.softmax(scores, dim=-1)
    vision_attn_weights.append(attn.detach().cpu())
    #print("Gemma-3 vision attention hook called. ",vision_attn_weights)
def split_text_vision_attn(layer_attn, text_mask, vision_mask, reduce=True):
    device = layer_attn.device
    text_mask   = text_mask.to(device).bool()
    vision_mask = vision_mask.to(device).bool()

    B, H, Q, K = layer_attn.shape
    S = text_mask.shape[1]


    if Q != S or K != S:
        Q_eff = min(Q, S)
        K_eff = min(K, S)
        layer_attn = layer_attn[:, :, :Q_eff, :K_eff]
        Q, K = Q_eff, K_eff
        text_mask   = text_mask[:, :Q_eff]
        vision_mask = vision_mask[:, :Q_eff]  

  
    text_q   = text_mask[:, :Q]     
    vision_q = vision_mask[:, :Q]    
    text_k   = text_mask[:, :K]     
    vision_k = vision_mask[:, :K]    

    blocks_t2t, blocks_t2v, blocks_v2t, blocks_v2v = [], [], [], []

    for b in range(B):
        tq_idx = text_q[b].nonzero(as_tuple=True)[0]
        vq_idx = vision_q[b].nonzero(as_tuple=True)[0]
        tk_idx = text_k[b].nonzero(as_tuple=True)[0]
        vk_idx = vision_k[b].nonzero(as_tuple=True)[0]

        def slice_block(q_idx, k_idx):
            if q_idx.numel() == 0 or k_idx.numel() == 0:
                return None
            return layer_attn[b, :, q_idx][:, :, k_idx] 

        TT = slice_block(tq_idx, tk_idx)
        TV = slice_block(tq_idx, vk_idx)
        VT = slice_block(vq_idx, tk_idx)
        VV = slice_block(vq_idx, vk_idx)

        blocks_t2t.append(TT)
        blocks_t2v.append(TV)
        blocks_v2t.append(VT)
        blocks_v2v.append(VV)

    if not reduce:
        def stack_or_none(lst):
            if all(x is None for x in lst):
                return None
            valid = [x for x in lst if x is not None]
            return torch.stack(valid, dim=0)
        return {
            "t2t": stack_or_none(blocks_t2t),
            "t2v": stack_or_none(blocks_t2v),
            "v2t": stack_or_none(blocks_v2t),
            "v2v": stack_or_none(blocks_v2v),
        }

    def agg_list(lst):
        valid = [x for x in lst if x is not None]
        if len(valid) == 0:
            return 0.0
        blk = torch.cat([v.unsqueeze(0) for v in valid], dim=0)
        return blk.mean().item()

    return {
        "t2t": agg_list(blocks_t2t),
        "t2v": agg_list(blocks_t2v),
        "v2t": agg_list(blocks_v2t),
        "v2v": agg_list(blocks_v2v),
    }
def split_text_vision_attn2(layer_attn, text_mask, vision_mask, reduce=True):
   
    device = layer_attn.device
    text_mask   = text_mask.to(device).bool()
    vision_mask = vision_mask.to(device).bool()

    B, H, Q, K = layer_attn.shape
    S = text_mask.shape[1]

    if Q > S or K > S:
        raise ValueError(f"Mask length {S} shorter than attention dims Q={Q}, K={K}")

   
    text_q   = text_mask[:, :Q]     
    vision_q = vision_mask[:, :Q]    
    text_k   = text_mask[:, :K]      
    vision_k = vision_mask[:, :K]   

   
    blocks_t2t = []
    blocks_t2v = []
    blocks_v2t = []
    blocks_v2v = []

    for b in range(B):
       
        tq_idx = text_q[b].nonzero(as_tuple=True)[0]    
        vq_idx = vision_q[b].nonzero(as_tuple=True)[0]  
        tk_idx = text_k[b].nonzero(as_tuple=True)[0]    
        vk_idx = vision_k[b].nonzero(as_tuple=True)[0]  

        def slice_block(q_idx, k_idx):
            if q_idx.numel() == 0 or k_idx.numel() == 0:
                return None
         
            return layer_attn[b, :, q_idx][:, :, k_idx]  

        TT = slice_block(tq_idx, tk_idx)   # text -> text
        TV = slice_block(tq_idx, vk_idx)   # text -> vision
        VT = slice_block(vq_idx, tk_idx)   # vision -> text
        VV = slice_block(vq_idx, vk_idx)   # vision -> vision

        blocks_t2t.append(TT)
        blocks_t2v.append(TV)
        blocks_v2t.append(VT)
        blocks_v2v.append(VV)
    #print("****** blocks_t2v ",blocks_t2v)

    if not reduce:
      
        def stack_or_none(lst):
           
            if all(x is None for x in lst):
                return None
          
            valid = [x for x in lst if x is not None]
          
            return torch.stack(valid, dim=0)

        return {
            "t2t": stack_or_none(blocks_t2t),
            "t2v": stack_or_none(blocks_t2v),
            "v2t": stack_or_none(blocks_v2t),
            "v2v": stack_or_none(blocks_v2v),
        }


    def agg_list(lst):
   
        valid = [x for x in lst if x is not None]
        if len(valid) == 0:
            return 0.0
    
        blk = torch.cat([v.unsqueeze(0) for v in valid], dim=0)
      
        return blk.mean().item()

    out = {
        "t2t": agg_list(blocks_t2t),
        "t2v": agg_list(blocks_t2v),
        "v2t": agg_list(blocks_v2t),
        "v2v": agg_list(blocks_v2v),
    }
    return out
