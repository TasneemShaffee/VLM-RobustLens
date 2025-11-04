import re
import torch
from typing import Dict, List, Tuple


_IS_ATTN = re.compile(r"(?:^|\.)(self_attn|cross_attn|crossattention)(?:$|\.)", re.I)

def discover_attention_blocks(model) -> List[Tuple[str, torch.nn.Module]]:
    hits = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "Attention" in cls or _IS_ATTN.search(name):
          
            tail = name.rsplit(".", 1)[-1].lower()
            if tail not in {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}:
                hits.append((name, mod))
    return hits

def register_attn_hooks(attn_blocks):
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def mk(name):
        def hook(module, inputs, output):
          
            attn = None
            if isinstance(output, (tuple, list)):
                
                for idx in (1, -1):
                    if -len(output) <= idx < len(output) and torch.is_tensor(output[idx]):
                        attn = output[idx]; break
            elif isinstance(output, dict):
                for k in ("attentions", "attn_probs", "attn_weights"):
                    if k in output and torch.is_tensor(output[k]):
                        attn = output[k]; break
            else:
                for k in ("attentions", "attn_probs", "attn_weights"):
                    if hasattr(output, k) and torch.is_tensor(getattr(output, k)):
                        attn = getattr(output, k); break

            
            if attn is None:
                for k in ("attentions", "attn_probs", "attn_weights"):
                    if hasattr(module, k) and torch.is_tensor(getattr(module, k)):
                        attn = getattr(module, k); break

            if attn is not None:
                captured[name] = attn.detach().cpu()
        return hook

    for name, mod in attn_blocks:
        handles.append(mod.register_forward_hook(mk(name)))
    return handles, captured

@torch.no_grad()
def probe_attentions_with_runner(
    runner,                 
    images,                  
    text: str,
    *,
    print_first: int = 3     
):
   
    if not getattr(runner, "enable_attn", False):
        raise ValueError("runner was created without enable_attn=True; attentions may not be produced.")

   
    attn_blocks = discover_attention_blocks(runner.model)
    handles, captured = register_attn_hooks(attn_blocks)

   
    inputs = runner._pack_inputs(images if isinstance(images, list) else [images], text)
  
    device = next(runner.model.parameters()).device
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)


    """_ = runner.model(
        **inputs,
        output_attentions=True,  # double-ensure
        return_dict=True,
        use_cache=False,
    )"""

    

    if runner.__class__.__name__ == "InternVLRunner":
    
        question = text if isinstance(text, str) else (text[0] if text else "Describe this image.")
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        import inspect

        sig = inspect.signature(runner.model.vision_model.forward)
        #print("vision_model.forward params:", list(sig.parameters.keys()))
        runner.model.vision_model(
        pixel_values=inputs["pixel_values"],
        output_attentions=True,
        return_dict=True,
    )
    else:
        _ = runner.model(
        **inputs,
        output_attentions=True,  
        return_dict=True,
        use_cache=False,)

    for h in handles:
        h.remove()

    found = [n for n, _ in attn_blocks]
    got = list(captured.keys())
    missing = [n for n in found if n not in got]

  
    print("\n=== Attention Probe Report ===")
    print(f"Total attention blocks found: {len(found)}")
    print(f"Blocks that produced attention maps: {len(got)}")
    print(f"Blocks with no captured maps: {len(missing)}")
    for n in missing[:10]:
        print("  -", n)
    for n in got[:print_first]:
        t = captured[n]
        print(f"[{n}] shape: {tuple(t.shape)}")

    return captured, found, missing