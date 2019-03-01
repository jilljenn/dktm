import torch


# From https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
lengths = torch.Tensor([1, 3, 2, 4])
max_len = max(lengths)
idxes = torch.arange(0, max_len)  # if out=max_len, nothing will work
print('max', max_len)
print(lengths.unsqueeze(1))
print(idxes)
mask = torch.Tensor((idxes < lengths.unsqueeze(1)).float())
print(mask)
