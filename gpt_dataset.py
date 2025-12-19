import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    # pass a:
    # raw text
    # tokenizer ex: gtp-2 used in the BPE example, tiktoken
    # the context window size, how many tokens in each example
    # stride: how many tokens to move the window forwards each time
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunck = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunck))
            self.target_ids.append(torch.tensor(target_chunk))
    
    # Returns how many training examples were created
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]