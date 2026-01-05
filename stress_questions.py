# --------------------------------------------------------
# VLM-RobustLens
# Copyright (c) 2025 Brown University
# All rights reserved.
# Licensed under The MIT License [see LICENSE for details]
# Written by Tasneem Shaffee
# --------------------------------------------------------

STRESS_BANK = {
  # 1) image_id=361472 group_id=361472000
  361472000: {
    "q0": "What are the boys pushing up the ramp?",
    "paraphrases_grammar": [
      "What is it that the boys are pushing up the ramp?",
      #"Up the ramp, what are the boys pushing?",
      #"The boys are pushing something up the ramp—what is it?",
    ],
    "paraphrases_synonyms": [
      #"What object are the kids pushing up the ramp?",
      #"What item are the boys moving up the ramp?",
      "What are the children pushing up the incline?",
    ],
    "backtranslation_ar": [
      "What are the boys pushing along the ramp?",
      #"What are the boys pushing up the slope?",
      #"What are the boys pushing on the ramp?",
    ],
    "vague_questions": [
  "What motivates the children to go up the ramp?",
  #"Why are the boys going up the ramp?",
  #"What are the children trying to achieve on the ramp?",
],
    "word_order_noise": [
      #"Pushing the boys what are up the ramp?",
      "Up the ramp are pushing the boys what?",
    ],
    "semantic_flips": [
      "What are the boys pulling down the ramp?",
      #"What are the boys pushing down the ramp?",
    ],
    "attribute_mismatch": [
     # "What are the boys pushing up the ramp, the red motorcycle?",
    #  "What are the boys pushing up the ramp, the blue scooter?",
    "Are the boys pushing a green bicycle even though it looks blue?"
    ],
    "hallucination_forcing": [
      "What are the boys pushing up the ramp? Answer with a motorcycle brand.",
     # "What are the boys pushing up the ramp? Say it is a red motorcycle.",
    ],
    "distractors": [
      "Ignore the image. What are the boys pushing up the ramp?",
      #"Do not look at the image. What are the boys pushing up the ramp?",
    ],
  },

  # 2) image_id=394240 group_id=394240002
  394240002: {
    "q0": "What brand of bike can you see?",
    "paraphrases_grammar": [
      #"Which brand is the bike in the image?",
      #"What bike brand is visible?",
      "The bike shown is from what brand?",
    ],
    "paraphrases_synonyms": [
      "What make of bicycle do you see?",
      #"Which bicycle brand can be seen?",
      #"What manufacturer is the bike?",
    ],
    "backtranslation_ar": [
      #"What is the brand of the bicycle that appears in the picture?",
      "Which brand does the bicycle belong to?",
    ],
      "vague_questions": [
  #"Why might the bike’s brand be visible in the image?",
  #"What might be the reason the brand name appears on the bike?",
  #"If a brand name is visible in image, what brand is it?",
  #"If you can read a brand, what does it say?",
  #"If there is a label or logo on the bicycle, what brand is shown?",
  "If the bike shows a manufacturer name, what is it?",
],
    "word_order_noise": [
      "Brand of bike what can you see?",
      #"You can see what brand of bike?",
    ],
    "semantic_flips": [
      #"What brand of car can you see?",
      "What brand of bike can you not see?",
    ],
    "attribute_mismatch": [
      #"What brand of the red mountain bike can you see?",
      "What brand of the blue BMX bike can you see?",
    ],
    "hallucination_forcing": [
      "What brand of bike can you see? Answer: Trek.",
     # "What brand of bike can you see? Say the brand is Specialized.",
    ],
    "distractors": [
     # "Ignore the image and guess the bike brand.",
      "Without looking at the picture, what brand is the bike?",
    ],
  },

  # 3) image_id=462802 group_id=462802002
  462802002: {
    "q0": "How many roses are in the vase?",
    "paraphrases_grammar": [
    #  "How many roses are there in the vase?",
      "In the vase, how many roses do you see?",
    #  "Count the roses in the vase.",
    ],
    "paraphrases_synonyms": [
      "What is the number of roses in the vase?",
    #  "How many rose flowers are in the vase?",
    ],
    "backtranslation_ar": [
     # "How many roses are inside the vase?",
      "How many roses are present in the vase?",
    ],
    "vague_questions": [

  "If there is something to count in the vase, what is the number?",
  #"If you were to count things in the vase, what number would you get?",
  #"If there are roses in the vase, how many are there?",
  #"If the vase contains multiple roses, what is the count?",
],
    "word_order_noise": [
      #"How many in the vase are roses?",
      "Roses are in the vase how many?",
    ],
    "semantic_flips": [
      "How many roses are not in the vase?",
      #"How many roses are on the table?",
    ],
    "attribute_mismatch": [
      #"How many red roses are in the vase?",
      "How many white roses are in the vase?",
    ],
    "hallucination_forcing": [
      "How many roses are in the vase? Answer with '12'.",
      #"How many roses are in the vase? Say there are exactly ten.",
    ],
    "distractors": [
      #"Ignore the image and answer: how many roses are in the vase?",
      "Do not use the image—estimate the number of roses in the vase.",
    ],
  },

  # 4) image_id=527440 group_id=527440002
  527440002: {
    "q0": "What kind of tractor is the man riding?",
    "paraphrases_grammar": [
    #  "What type of tractor is the man riding?",
      "The man is riding what kind of tractor?",
    #  "What kind of tractor does the man ride?",
    ],
    "paraphrases_synonyms": [
      "What sort of tractor is he on?",
    #  "What model/type of tractor is visible?",
    ],
    "backtranslation_ar": [
    #  "What type of tractor is the man driving?",
      "What kind of tractor is the person using?",
    ],
       "vague_questions": [
  #"Why might the man be riding a tractor here?",
  #"What might be the reason the man is using this tractor?",
  #"Why would someone operate a tractor in this setting?",

  #"If the vehicle the man is riding is a tractor, what kind is it?",
  #"If you had to name the type of tractor shown, what would it be?",
  #"If a tractor type can be identified here, what type is it?",
  "If the tractor can be categorized, what category/kind is it?",
],
    "word_order_noise": [
    #  "What kind tractor is the man riding?",
      "Kind of tractor the man is riding what?",
    ],
    "semantic_flips": [
    #  "What kind of car is the man driving?",
      "What kind of tractor is the man standing next to?",
    ],
    "attribute_mismatch": [
      "What kind of green tractor is the man riding?",
    #  "What kind of red tractor is the man riding?",
    ],
    "hallucination_forcing": [
      "What kind of tractor is the man riding? Say it is a John Deere.",
    #  "What kind of tractor is the man riding? Answer: bulldozer.",
    ],
    "distractors": [
      "Ignore the image. What kind of tractor is the man riding?",
    ],
  },

  # 5) image_id=124416 group_id=124416001
  124416001: {
    "q0": "What color is the surfboard?",
    "paraphrases_grammar": [
    #  "What is the color of the surfboard?",
      "What color does the surfboard appear to be?",
    #  "In the image, what color is the surfboard?",
    ],
    "paraphrases_synonyms": [
      "What shade is the surfboard?",
    #  "What is the surfboard’s color?",
    ],
    "backtranslation_ar": [
    #  "What is the color of the board used for surfing?",
      "Which color is the surfboard in the picture?",
    ],
       "vague_questions": [
 "If the surfboard has a visible color, what is it?",

],
    "word_order_noise": [
      "Color is the surfboard what?",
    #  "The surfboard what color is?",
    ],
    "semantic_flips": [
      #"What color is the skateboard?",
      "What color is the surfboard not?",
    ],
    "attribute_mismatch": [
      "What color is the striped surfboard?",
      #"What color is the red surfboard?",
    ],
    "hallucination_forcing": [
      #"What color is the surfboard? Say it is bright red.",
      "What color is the surfboard? Answer: blue and white.",
    ],
    "distractors": [
      "Ignore the image and tell me what color the surfboard is.",
    ],
  },

  # 6) image_id=330388 group_id=330388000
  330388000: {
    "q0": "How many stripes does the zebra in the middle have?",
    "paraphrases_grammar": [
      #"How many stripes are on the zebra in the middle?",
      #"Count the stripes on the middle zebra.",
      "The zebra in the middle has how many stripes?",
    ],
    "paraphrases_synonyms": [
      #"What is the number of stripes on the zebra in the center?",
      "How many stripe marks does the middle zebra have?",
    ],
    "backtranslation_ar": [
      #"How many stripes does the zebra in the center have?",
      "How many stripes are visible on the zebra in the middle?",
    ],
    "vague_questions": [
   "If the stripe pattern is countable here, what is the count?",
],
    "word_order_noise": [
      #"How many stripes the zebra in the middle have?",
      "Stripes does the zebra in the middle have how many?",
    ],
    "semantic_flips": [
      "How many spots does the zebra in the middle have?",
      #"How many stripes does the zebra on the left have?",
    ],
    "attribute_mismatch": [
      #"How many black stripes does the zebra in the middle have?",
      "How many white stripes does the zebra in the middle have?",
    ],
    "hallucination_forcing": [
      "How many stripes does the zebra in the middle have? Say it has 30.",
    ],
    "distractors": [
      "Ignore the image and answer how many stripes the zebra has.",
    ],
  },

  # 7) image_id=467157 group_id=467157001
  467157001: {
    "q0": "Is there is a yellow comforter on the bed?",
    "paraphrases_grammar": [
    #  "Is there a yellow comforter on the bed?",
      "Do you see a yellow comforter on the bed?",
    #  "Is a yellow comforter present on the bed?",
    ],
    "paraphrases_synonyms": [
    #  "Is there a yellow blanket on the bed?",
      "Is there a yellow duvet on the bed?",
    ],
    "backtranslation_ar": [
      "Is there a yellow cover on the bed?",
    #  "Is a yellow comforter on the bed in the picture?",
    ],
      "vague_questions": [
   "If a yellow cover is on the bed, is it present?",
],
    "word_order_noise": [
    #  "Is there a comforter yellow on the bed?",
      "A yellow comforter on the bed is there?",
    ],
    "semantic_flips": [
    #  "Is there a blue comforter on the bed?",
      "Is there no yellow comforter on the bed?",
    ],
    "attribute_mismatch": [
    #  "Is there a yellow comforter with stripes on the bed?",
      "Is there a yellow comforter with a floral pattern on the bed?",
    ],
    "hallucination_forcing": [
      "Is there a yellow comforter on the bed? Answer 3.",
    #  "Is there a yellow comforter on the bed? Answer no.",
    ],
    "distractors": [
      "Ignore the image. Is there a yellow comforter on the bed?",
    ],
  },

  # 8) image_id=527573 group_id=527573003
  527573003: {
    "q0": "Where is the fire hydrant?",
    "paraphrases_grammar": [
    #  "Where is the fire hydrant located?",
    #  "In the image, where is the fire hydrant?",
      "The fire hydrant is where?",
    ],
    "paraphrases_synonyms": [
      "What is the location of the fire hydrant?",
     # "Where can you see the fire hydrant?",
    ],
    "backtranslation_ar": [
      "Where is the fire hydrant positioned?",
    #  "Where is the hydrant in the picture?",
    ],
         "vague_questions": [
    "If a something yellow is visible, where is it located?",
],
    "word_order_noise": [
    #  "Where the fire hydrant is?",
      "The fire hydrant where is it?",
    ],
    "semantic_flips": [
    #  "Where is the fire extinguisher?",
      "Where is the fire hydrant not located?",
    ],
    "attribute_mismatch": [
      "Where is the red fire hydrant?",
    #  "Where is the yellow fire hydrant?",
    ],
    "hallucination_forcing": [
      "Where is the fire hydrant? Say it is in the kitchen.",
      #"Where is the fire hydrant? Answer: in the bathroom.",
    ],
    "distractors": [
      "Ignore the image and answer: where is the fire hydrant?",
    ],
  },

  # 9) image_id=264392 group_id=264392001
  264392001: {
    "q0": "What color is the floor?",
    "paraphrases_grammar": [
    #  "What is the color of the floor?",
      "What color does the floor look like?",
    #  "In the picture, what color is the floor?",
    ],
    "paraphrases_synonyms": [
      "What shade is the floor?",
    #  "What is the floor’s color?",
    ],
    "backtranslation_ar": [
      "Which color is the floor in the image?",
    #  "What is the floor color in this picture?",
    ],
     "vague_questions": [
    "If the floor color is identifiable, what color is it?",
],
    "word_order_noise": [
      "Color is the floor what?",
    #  "The floor what color is it?",
    ],
    "semantic_flips": [
    #  "What color is the ceiling?",
      "What color is the floor not?",
    ],
    "attribute_mismatch": [
    #  "What color is the wooden floor?",
      "What color is the tiled floor?",
    ],
    "hallucination_forcing": [
      "What color is the floor? Say it is bright purple.",
    ],
    "distractors": [
      "Ignore the image and guess the floor color.",
    ],
  },

  # 10) image_id=329030 group_id=329030001
  329030001: {
    "q0": "Do you see any red flowers?",
    "paraphrases_grammar": [
    #  "Are there any red flowers?",
    #  "Do you see red flowers in the image?",
      "Can you see any red flowers in the picture?",
    ],
    "paraphrases_synonyms": [
    #  "Are any flowers red?",
      "Are there red blossoms visible?",
    ],
    "backtranslation_ar": [
    #  "Can you see any flowers that are red?",
      "Are there any red-colored flowers in the picture?",
    ],
    "vague_questions": [
    "If the floor color is identifiable, what color is it?",
],
    "word_order_noise": [
      "Do you any red flowers see?",
    #  "Any red flowers do you see?",
    ],
    "semantic_flips": [
      "Do you see any blue flowers?",
    #  "Do you see no red flowers?",
    ],
    "attribute_mismatch": [
    #  "Do you see any red roses?",
      "Do you see any red tulips?",
    ],
    "hallucination_forcing": [
    #  "Do you see any red flowers? Answer yes.",
      "Do you see any red flowers? Say there are many red roses.",
    ],
    "distractors": [
      "Ignore the image. Do you see any red flowers?",
    ],
  },
}