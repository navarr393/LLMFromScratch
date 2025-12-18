from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

# instantiate the tokenizer 
tokenizer = tiktoken.get_encoding("gpt2") # load the GPT-2 encoding scheme (vocabulary and BPE merge rules)

# pass the text we want to tokenize, similar to SimpleTokenizerV2
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special= {"<|endoftext|>"})
print(integers)

# decode the token ids back into text
strings = tokenizer.decode(integers)
print(strings)

# Excersise 2.1 Byte pair encoding of unkown words
text = ("Akwirw ier")
integers = tokenizer.encode(text)
print(integers)

# turn back into strings
strings = tokenizer.decode(integers)
print(strings) 

# encode the .txt with the new set from gpt-2
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# remove the first 50 tokens from the dataset
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size] # input 
y = enc_sample[1:context_size+1] # target, shift by one but also predict the next one by adding 1
print(f"x: {x}")
print(f"y:     {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i] # everything up to i but not including index i
    desired = enc_sample[i]  # single token at index i, i is included by itself
    print(context, "---->", desired)

# same example but convert the token ids into text
for i in range(1, context_size+1):
    context = enc_sample[:i] # always a list 
    desired = enc_sample[i]  # always a single integer
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired])) # must pass a list this is why we do [desired]