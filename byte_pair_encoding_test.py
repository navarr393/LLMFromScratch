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