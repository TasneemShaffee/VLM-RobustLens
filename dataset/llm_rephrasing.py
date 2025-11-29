from transformers import pipeline
import torch


def get_rephrasings(messages):
    #this def brick is form the hugging face docs found here:
    # https://huggingface.co/openai/gpt-oss-120b
    model_id = "openai/gpt-oss-120b"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]

def parse_rephrasings(rephrasings):
    print(rephrasings)


def gram_var_and_syn_rep(question, count):
    messages = [
        {"role": "user", f"content": "You are am English grammar expert. \
         Concisely generate rephrasings of this question: {question}. \
          Make sure all rephrasings keep the same semantic meaning while varying aspects such as grammar,\
          word order, and diction. Generate {count} rephrasings numbering them 1 through {count}"},
    ]
    rephrasings = get_rephrasings(messages)

    return parse_rephrasings(rephrasings)



def back_translate(question, count):
    forward_messages = [
        {"role": "forward_translator", f"content": "You are an expert in German to English translations. \
         Concisely generate a translation of this question: {question}. \
          Make the translation keeps the same semantic meaning."
          "Generate a single translation"},
    ]
    forward_translation = get_rephrasings(forward_messages)

    translated_question = parse_rephrasings(forward_translation)

    backward_messages = [
         {"role": "backward_translator", f"content": "You are an expert in English to German translations. \
         Concisely generate translations of this question: {translated_question}. \
        Make sure all translations keep the same semantic meaning.\
        Generate {count} translations, numbering them 1 through {count}"}
    ]

    backward_translations = get_rephrasings(backward_messages)

    return parse_rephrasings(backward_translations)

