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
processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
processed = [item for item in processed if item.strip()]
print(len(processed))
print(processed[:30])

# tokens are sorted alphabetically and given a  token ID
all_words = sorted(set(processed)) # no repeaded words is what a set returns
vocab_size = len(all_words)
print(vocab_size)

# return a dictionary of "word: integer" format from 'all_words'
vocab = {token:integer for integer, token in enumerate(all_words)}
print(vocab.items())

# print the first 50
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break