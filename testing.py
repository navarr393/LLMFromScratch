# open the file and store as f 
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read() # read the file until EOF 
print("Total nuumber of chars:", len(raw_text))
print(raw_text[:99])

import re
text = "Hello, World! This-- a test?"
result = re.split(r'([,.!:;?_"()\']|--|\s)', text) # remove all the s's from text?
# splits the words every whitespace
print(result)

# remove the whitespaces from the list 
result = [item for item in result if item.strip()]
print(result)

# apply it to raw text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# tokens are sorted alphabetically and given a  token ID
all_words = sorted(set(preprocessed)) # no repeaded words is what a set returns
vocab_size = len(all_words)
print(vocab_size)

# return a dictionary of "word: integer" format from 'all_words'
vocab = {token:integer for integer, token in enumerate(all_words)}
print(vocab.items())

# print the first 50, we use enumerate to have a counter and know when to stop
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

from simple_tokenizer import SimpleTokenizerV1

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text) # encode to integer token ids
print(ids)

# decode the text back to strings
print(tokenizer.decode(ids))

# example 2 with error
text = "do you like tea?"
print(tokenizer.encode(text))

all_tokens = sorted(list(set(preprocessed))) # make a sorted list of unique tokens from preprocessed
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # add  the 2 new special tokens
vocab = {token:integer for integer, token in enumerate(all_tokens)} # create vocab dictionary mapping all tokens to integer IDs

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]): # print the last 5 items 
    print(item)

# the following will fail, this is why we need enumerate()
# for i, item in vocab.items()[-5:]:
#     print(item)

# test tokenizer_v2
from simple_tokenizer_v2 import SimpleTokenizerV2
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2)) # put the string " <|endoftext|> " between both text1 and text2
print(text)

# tokenize the sample text
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

# detokenize the sample text
print(tokenizer.decode(tokenizer.encode(text)))

