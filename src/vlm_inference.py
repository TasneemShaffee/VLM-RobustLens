
import os
from abc import ABC
from typing import Optional, Dict, Any, Union, List
from PIL import Image
import torch
import torchvision.transforms as T
from io import BytesIO
import os
import requests
from transformers import (
    AutoConfig, AutoProcessor, AutoTokenizer,
    AutoModelForVision2Seq, AutoModelForCausalLM,AutoModel
)
from attention_checker import probe_attentions_with_runner
from torchvision.transforms.functional import InterpolationMode
import argparse
def _dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def _to_list(x): return x if isinstance(x, list) else [x]

def _pilify(im):
    if isinstance(im, Image.Image): return im
    if isinstance(im, str) and not im.startswith(("http://", "https://")):
        return Image.open(im).convert("RGB")
    return im 

class VLMRunner(ABC):
    supports_template_packs_images: bool = False  

    def __init__(
        self,
        model_id: str,
        *,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        enable_attn: bool = False,
    ):
        self.model_id = model_id
      
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.torch_dtype = torch_dtype or _dtype()
        self.enable_attn = enable_attn

        if cache_dir:
            os.environ.setdefault("HF_HOME", cache_dir)

        self.cfg = None
        if enable_attn:
            self.cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
            if getattr(self.cfg, "attn_implementation", None) in (None, "sdpa", "flash_attention_2"):
                self.cfg.attn_implementation = "eager"
            if hasattr(self.cfg, "output_attentions"): self.cfg.output_attentions = True
            if hasattr(self.cfg, "use_cache"): self.cfg.use_cache = False

      
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.device = None
        self.common = dict(
            cache_dir=cache_dir, trust_remote_code=trust_remote_code, token=hf_token,
            torch_dtype=self.torch_dtype, device_map=device_map, config=self.cfg,
        )
        self.model_initialize()
    def model_initialize(self):
        self.tokenizer = self.build_tokenizer(self.model_id)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            token=self.hf_token,
            tokenizer=self.tokenizer,
        )
      
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **self.common)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **self.common)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _pack_inputs(self, images, text) -> Dict[str, Any]:
        msgs = self._messages(images, text)
        if self.supports_template_packs_images:
           
            return self.processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True
            )
      
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
       
        pil_imgs = []
        for im in images:
            if isinstance(im, Image.Image):
                pil_imgs.append(im)
            elif isinstance(im, str) and not im.startswith("http"):
                pil_imgs.append(Image.open(im).convert("RGB"))
        return self.processor(text=[prompt], images=pil_imgs or None, return_tensors="pt")

    def build_tokenizer(self, model_id: str):
       
        return AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.cache_dir, trust_remote_code=self.trust_remote_code
        )


    def _messages(self, images, text):
        content = [{"type": "image", "image": _pilify(im)} for im in images]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _pack(self, images, text) -> Dict[str, Any]:
        msgs = self._messages(images, text)
        if self.supports_template_packs_images:
            return self.processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True
            )
       
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        pil_images = [x for x in map(_pilify, images) if isinstance(x, Image.Image)]
        return self.processor(text=[prompt], images=pil_images or None, return_tensors="pt")

    def run(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]], text: str,
            *, do_generate=False, gen_kwargs: Optional[Dict[str, Any]] = None):
        images = _to_list(images)
        inputs = self._pack(images, text)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v): inputs[k] = v.to(self.device)

        with torch.no_grad():
            if do_generate:
                gen_kwargs = gen_kwargs or {}
                out_ids = self.model.generate(**inputs, **gen_kwargs)
                if "input_ids" in inputs:
                    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
                else:
                    trimmed = out_ids
                return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
          
            return self.model(**inputs, output_attentions=self.enable_attn, return_dict=True, use_cache=False)




class Qwen3VLRunner(VLMRunner):
    supports_template_packs_images = True
   

class InternVLRunner(VLMRunner):
    
    supports_template_packs_images = False 
    

    def model_initialize(self):
        self.tokenizer = self.build_tokenizer(self.model_id)
        self.model =  AutoModel.from_pretrained(self.model_id, **self.common)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.image_size=448 
        self.max_num=12
    def _pilify(self,x):
    
        if isinstance(x, Image.Image):
            return x.convert("RGB")
    

        if isinstance(x, str) and (x.startswith("http://") or x.startswith("https://")):
            try:
                resp = requests.get(x, timeout=10)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Failed to load image from URL {x}: {e}")


        if isinstance(x, str):
            if not os.path.exists(x):
                raise FileNotFoundError(f"Image path not found: {x}")
            return Image.open(x).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(x)}")    
    def _pack_inputs(self, images, text) -> Dict[str, Any]:
    
   
        if not isinstance(images, list):
            images = [images]

   
        transform = self.build_transform(self.image_size)

        all_tile_tensors = []
        num_patches_list = []

        for im in images:
           
            pil = self._pilify(im)  
            tiles = self.dynamic_preprocess(
            pil,
            image_size=self.image_size,
            use_thumbnail=True,
            max_num=self.max_num,
        )
            num_patches_list.append(len(tiles))
        
            tile_tensor = torch.stack([transform(t) for t in tiles])
            all_tile_tensors.append(tile_tensor)

   
        pixel_values = torch.cat(all_tile_tensors, dim=0).to(self.device, dtype=torch.bfloat16)

        return {
        "pixel_values": pixel_values,
        "num_patches_list": num_patches_list,  
        }
              
    def build_tokenizer(self, model_id: str):
        tok = AutoTokenizer.from_pretrained(
            model_id, use_fast=False, 
            cache_dir=self.cache_dir, trust_remote_code=self.trust_remote_code
        )
       
        needs = []
        if not hasattr(tok, "start_image_token"): needs.append("<image>")
        if not hasattr(tok, "end_image_token"):   needs.append("</image>")
        if needs:
            
            extra = list(set(getattr(tok, "additional_special_tokens", []) + needs))
            tok.add_special_tokens({"additional_special_tokens": extra})
        return tok

    def run(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        text: Union[str, List[str]],
        *,
        do_generate: bool = True,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        history: Optional[Any] = None,
        ):
   
     
        gen_cfg = dict(max_new_tokens=1024, do_sample=True)

        if isinstance(images, (str, Image.Image)):
            images = [images]

        is_batch_text = isinstance(text, list)
        if is_batch_text and len(text) != len(images):
            raise ValueError("When passing a list of prompts, its length must match the number of images.")


        all_tiles = []
        num_patches_list = []
        transform = self.build_transform(self.image_size) 

        for im in images:
         
            pil = self._pilify(im)  
        
            tiles = self.dynamic_preprocess(
            pil,
            image_size=self.image_size,
            use_thumbnail=True,
            max_num=self.max_num,
            )
            num_patches_list.append(len(tiles))
         
            pv = torch.stack([transform(t) for t in tiles])
            all_tiles.append(pv)

   
        pixel_values = torch.cat(all_tiles, dim=0).to(self.device, dtype=torch.bfloat16)

        if len(images) == 1 and not is_batch_text:
       
            response, hist = self.model.chat(
                self.tokenizer,
                pixel_values,
                text,
                gen_cfg,
                num_patches_list=num_patches_list,   
                history=history,
                return_history=True,
                )
            return {"text": response, "history": hist}

    
        questions = text if is_batch_text else [text] * len(images)
        responses = self.model.batch_chat(
        self.tokenizer,
        pixel_values,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=gen_cfg,
        )
   
        return responses

    def build_transform(self,input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self,aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self,image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
     
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

    
        target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])


        target_aspect_ratio = self.find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

   
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

  
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
            )
       
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
        return processed_images

class Gemma3Runner(VLMRunner):
    supports_template_packs_images = False
    def build_tokenizer(self, model_id: str):
        return AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.cache_dir, trust_remote_code=self.trust_remote_code
        )


class CogVLMRunner(VLMRunner):
    supports_template_packs_images = False
    def build_tokenizer(self, model_id: str):
        tok = AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.cache_dir, trust_remote_code=self.trust_remote_code, use_fast=True
        )
     
        if "<image>" not in getattr(tok, "additional_special_tokens", []):
            tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
        return tok

DEFAULT_IDS = {
    "qwen3vl": "Qwen/Qwen3-VL-4B-Instruct",      
    "gemma3":  "google/gemma-3-4b-pt",
    "llama3v": "meta-llama/Llama-3.2-11B-Vision",
    "internvl":"OpenGVLab/InternVL3_5-4B",
    "cogvlm2": "THUDM/cogvlm2-llama3-chat-19B",
}


def load_runner(name: str, *, model_id: Optional[str] = None, **kwargs) -> VLMRunner:
    key = name.lower()
    model_id = model_id or DEFAULT_IDS.get(key)
    if key.startswith("qwen"):   return Qwen3VLRunner(model_id, **kwargs)
    if key.startswith("intern"): return InternVLRunner(model_id, **kwargs)
    if key.startswith("gemma"):  return Gemma3Runner(model_id, **kwargs)
    if key.startswith("cog"):    return CogVLMRunner(model_id, **kwargs)
 
    return VLMRunner(model_id, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="Run attention probe with arguments.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cache directory.")
    parser.add_argument("--model_name", type=str, required=True, choices=["gemma3", "qwen3vl", "internvl"], ,help="Model name to load.")
    parser.add_argument("--img_url", type=str, required=True, help="URL of the input image.")
    parser.add_argument("--text", type=str, required=True, help="Prompt text.")
    parser.add_argument("--enable_attn", action="store_true", help="Enable attention capture.")
    parser.add_argument("--do_generate", action="store_true", help="Generate output text.")
    parser.add_argument("--enable_attn_checker", action="store_true", help="Check if attention maps are accessible.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation.")
    return parser.parse_args()
if __name__ == "__main__":
    args=parse_args()
    CACHE = args.cache_dir 
    runner = load_runner(args.model_name, cache_dir=CACHE, enable_attn=True)

   
    img_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    text = "Describe this image."
    if args.enable_attn_checker:
        captured, found, missing = probe_attentions_with_runner(runner, img_url, text)
        print(f"Captured {len(captured)} attention tensors.")
 
    out_texts = runner.run(img_url, text, do_generate=True, gen_kwargs={"max_new_tokens": 128})
    print(out_texts)

   
    #outputs = runner.run(img_url, text, do_generate=False)
    #if hasattr(outputs, "attentions") and outputs.attentions is not None:
    #    print("Got attention maps.")