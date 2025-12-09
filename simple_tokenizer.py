import re

class SimpleTokenizerV1:
    def __init__(self, vocab): # pass a dictionary of vocabs for the initialization, literally a buch of words the more the better. A big Dataset
        self.str_to_int = vocab # 1, store the word-to-integer mapping
        self.int_to_str = {i:s for s, i in vocab.items()} #2, integer-to-word mapping

    def encode(self, text): # 3
        preprocessed = re.split(r'([,.?"()\']|--|\s)', text) # split on punctuation and whitespace, keeping delimiters
        preprocessed = [item.strip() for item in preprocessed if item.strip()] # remove whitespace items and strip whitespace
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids): # 4
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # 5, # remove spaces before punctuation marks
        return text