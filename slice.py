import torch
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import load_npz, eye, find


# X = load_npz('/Users/jilljenn/code/TF-recomm/data/fraction/ui0/X_train.npz')
X = eye(5)
print(X)

rows, cols, data = find(X)
indices = torch.LongTensor([rows, cols])
values = torch.FloatTensor(data)
sizes = X.shape
X_sp = torch.sparse_coo_tensor(indices, values, sizes)

print(X_sp)
# print(X_sp[3:5, :])
subset = torch.LongTensor([1, 3, 4])
# print(torch.index_select(X_sp, 0, subset))

X_de = X_sp.to_dense()
print(X_de[[1, 3, 4]])
print(torch.index_select(X_de, 0, subset))

subset_2d = torch.LongTensor([[1, 2], [3, 1]])
X_batch = (torch.index_select(X_de, 0, subset_2d.view(-1))
                .view(*subset_2d.shape, -1))

enc = OneHotEncoder()
features = [[1, 2], [2, 3], [2, 4]]
print(enc.fit(features))
print(enc.get_params())
X = torch.FloatTensor(enc.transform(features).todense())


bias = torch.nn.Embedding(5, 1)
embedding = torch.nn.Embedding(5, 5)

print(bias.weight)
print(embedding.weight)
print(torch.matmul(X, bias.weight))
print(X @ bias.weight)
print(X @ embedding.weight)

print(X_batch @ embedding.weight)
