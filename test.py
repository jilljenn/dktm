import torch


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# From https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
lengths = torch.LongTensor([1, 3, 2, 4]).to(device)
print(lengths)
max_len = max(lengths)
idxes = torch.arange(0, max_len).to(device)  # if out=max_len, will not work
print('max', max_len)
print(lengths.unsqueeze(1))
print('idxes', idxes)
mask = (idxes < lengths.unsqueeze(1)).float()
print(mask)
mask2 = ((idxes < lengths.unsqueeze(1))).byte()
print(mask2)


def sequence_mask(lengths, max_len):
    indexes = torch.arange(0, max_len).to(device)
    print('indexes', indexes.device)
    print('lengths', lengths.device)
    return torch.ByteTensor((indexes < lengths.unsqueeze(1)))


print(sequence_mask(lengths, 5))
