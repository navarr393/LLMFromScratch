- Code for book "Build a Large Language Model (From Scratch)"

Important Vocab:
- Byte pair encoding: A more sphisticated tokenization shceme based on the concept called byte pair encoding, used in GPT-2, GPT-3 and the original ChatGPT.
- BPE tokenizers break down unknown words into subwords and idividual characters. This way a BPE tokenizer can parse any word and doesnt need to replace unknown words with special tokens, such as '<|unk|>'

- Input Layer: the text turned into tokens and ID mappings for the LLM
- Hidden Layer: where the calcualtions, realations and context happens before making predicitons.
- Outout layer: what the LLM outputs the probability of each word.
- Example: "The cat sat on the"
- Input Layer: ["The", "cat", "sat", "on", "the"] -> embeddings
-Hidden Layer: understand the grammar, context, weight, subject etc
-Output Layer: 
    - "mat": 0.71
    - "floor": 0.12
    - "bed": 0.05

- Logits: raw unnormalized scores a model outputs before applying softmax or sigmoid
-ReLU (rectified linear unit): keeps positive signals and kills negative ones, enabling deep networks to learn nonlinear patterns.