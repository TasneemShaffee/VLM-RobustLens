from transformers import pipeline
import torch
import gc

model = None


def get_pipeline():

    global model
    if model is None:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        model = pipeline(
            "text-generation",
            model=model_id,
            dtype="auto",
            device_map="auto",
        )
    return model


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_rephrasings(messages):
    # this def brick is form the hugging face docs found here:
    # https://huggingface.co/openai/gpt-oss-120b

    clear_gpu_memory()
    pipe = get_pipeline()

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]


def parse_rephrasings(rephrasings):
    split_rephrasings = rephrasings.split("\n")
    return [s.split(". ", 1)[1] for s in split_rephrasings]


def gram_var_and_syn_rep(question, count):

    messages = [
        {
            "role": "user",
            "content": f"You are am English grammar expert. \
         Concisely generate rephrasings of this question: {question}. \
          Make sure all rephrasings keep the same semantic meaning while varying aspects such as grammar,\
          word order, and diction. Generate {count} rephrasings numbering them 1 through {count}. \
          Return only the list of questions, no other text.",
        },
    ]
    rephrasings = get_rephrasings(messages)

    return parse_rephrasings(rephrasings.get("content"))


def back_translate(question, count):
    forward_messages = [
        {
            "role": "user",
            "content": f"You are an expert in English to German translations. \
         Concisely generate a translation of this question: {question}. \
          Make the translation keeps the same semantic meaning."
            "Generate a single translation. \
          Return only the translation, no other text.",
        },
    ]
    forward_translation = get_rephrasings(forward_messages)
    translated_question = forward_translation.get("content")

    backward_messages = [
        {
            "role": "user",
            "content": f"You are an expert in German to English translations. \
         Concisely generate translations of this question: {translated_question}. \
        Make sure all translations keep the same semantic meaning.\
        Generate {count} translations, numbering them 1 through {count}. \
        Return only the list of questions, no other text.",
        }
    ]

    backward_translations = get_rephrasings(backward_messages)
    return parse_rephrasings(backward_translations.get("content"))
