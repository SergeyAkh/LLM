from torch.utils.data import Dataset, DataLoader
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, pad_token_id=50256):
        self.input_ids = []
        self.target_ids = []
        eos_token = "<|endoftext|>"
        full_text = eos_token.join(txt.astype(str).tolist()) + eos_token
        # Tokenize the entire text
        token_ids = tokenizer.encode(full_text, allowed_special={eos_token})
        # Use a sliding window to chunk the book into overlapping sequences of max_length

        if len(token_ids) < max_length:
            input_chunk = token_ids + [pad_token_id] * (max_length - len(token_ids))
            target_chunk = input_chunk[1:] + [pad_token_id]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        else:
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1:i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(tokenizer, txt, batch_size=4, max_length=None,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader