CATEGORIES_IN_ORDER = [
  "paraphrases_grammar",
  "paraphrases_synonyms",
  "backtranslation_ar",
  "vague_questions", 
  "distractors",
  "hallucination_forcing",
  "semantic_flips",
  "word_order_noise",
  "attribute_mismatch",
]

def build_variants_for_gid(stress_bank, gid):
    
    entry = stress_bank.get(gid, None)
    if entry is None:
        return None, []

    q0 = entry["q0"]
    variants = []
    for cat in CATEGORIES_IN_ORDER:
        for q in entry.get(cat, []):
            variants.append({"type": cat, "text": q})
    return q0, variants