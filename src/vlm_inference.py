
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
    AutoModelForVision2Seq, AutoModelForCausalLM,AutoModel,AutoModelForImageTextToText
)

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
            self.cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True,trust_remote_code=trust_remote_code)
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
            dtype=self.torch_dtype, device_map=device_map, config=self.cfg,
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
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **self.common)
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
        #return self.processor(text=[prompt], images=pil_imgs or None, return_tensors="pt",return_mm_token_type_ids=True)
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
  

def run_onepass(
    self,
    images: Union[str, Image.Image, List[Union[str, Image.Image]]],
    text: str,
):
    # 1) preprocess images -> pixel_values, num_patches_list (you already do this)
    if isinstance(images, (str, Image.Image)):
        images = [images]

    all_tiles, num_patches_list = [], []
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

    # 2) build query using the same template logic as `chat`
    question = text
    if pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    template = self.model.conv_template  # or get_conv_template(self.model.template)
    template = template.copy() if hasattr(template, "copy") else template
    template.system_message = self.model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

    # set img_context_token_id, like chat/batch_chat do
    self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # insert image context tokens
    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    # 3) tokenize
    model_inputs = self.tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(self.device)
    attention_mask = model_inputs['attention_mask'].to(self.device)

    # 4) simple image_flags: mark all images as used
    # shape [batch, 1] if you have one sequence here
    image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.bool, device=self.device)

    # 5) ONE FORWARD PASS with attentions
    with torch.no_grad():
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    return outputs
class InternVLRunner(VLMRunner):
    
    supports_template_packs_images = False 
    
    
    def model_initialize(self):
        self.tokenizer = self.build_tokenizer(self.model_id)
        self.model =  AutoModel.from_pretrained(self.model_id, **self.common)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.image_size= 336 #448 
        self.max_num= 4 #12
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

    def run(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]], text: str,
            *, do_generate=False, gen_kwargs: Optional[Dict[str, Any]] = None):
    
   
        if isinstance(images, (str, Image.Image)):
            images = [images]

        all_tiles, num_patches_list = [], []
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

   
        question = text
        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        template = self.model.conv_template  # or get_conv_template(self.model.template)
        template = template.copy() if hasattr(template, "copy") else template
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)


        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)


        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)


        image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.bool, device=self.device)

  
        with torch.no_grad():
            outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        return outputs   
              
    """def run(
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
            #print("Single image and single text input mode.")
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                text,
                gen_cfg,
                num_patches_list=num_patches_list,   
                history=history,
                return_history=False,
                )
            #return {"text": response, "history": hist}
            return {"text": response}

        else:
            questions = text if is_batch_text else [text] * len(images)
            responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=gen_cfg,
            )
   
            return responses """

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
    "gemma3":  "google/gemma-3-4b-it",
    "llama3v": "meta-llama/Llama-3.2-11B-Vision",
    "internvl":"OpenGVLab/InternVL3_5-4B",
    "cogvlm2": "THUDM/cogvlm2-llama3-chat-19B",
}


def load_runner(name: str, *, model_id: Optional[str] = None, **kwargs) -> VLMRunner:
    key = name.lower()
    print(f"Loading runner for model: {key}")
    model_id = model_id or DEFAULT_IDS.get(key)
    if key.startswith("qwen"):   return Qwen3VLRunner(model_id, **kwargs)
    if key.startswith("intern"): return InternVLRunner(model_id, **kwargs)
    if key.startswith("gemma"):  return Gemma3Runner(model_id, **kwargs)
    if key.startswith("cog"):    return CogVLMRunner(model_id, **kwargs)
 
    return VLMRunner(model_id, **kwargs)


    #outputs = runner.run(img_url, text, do_generate=False)
    #if hasattr(outputs, "attentions") and outputs.attentions is not None:
    #    print("Got attention maps.")