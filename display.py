from scipy.sparse import load_npz
from collections import Counter
import numpy as np
import sys


# X = load_npz(sys.argv[1])
# print(type(X))
# print(X.shape)
# print(X.dtype)

# snap = np.load('data/assistments2/data.npz')
# print(snap['exercises'].min())
# print(snap['exercises'].max())

X_swf = load_npz('data/assistments2/metadata-swf.npz')
print(X_swf.shape)
X_iswf = load_npz('data/assistments2/metadata-iswf.npz')
print(X_iswf.shape)
print(Counter(X_iswf[:, :-369].sum(axis=1).A1))

print('meow', (X_iswf[:, -369:] - X_swf).sum())

indices = np.load("data/assistments2/indices.npy")
exercises = np.load("data/assistments2/exercises.npy")
pos = indices[5][0]
pos2 = indices[5][1]
print(pos, pos2)
print(exercises[5][0])
print(X_iswf[pos, :-369])
print(X_iswf[pos2, :-369])

for data in ['assistments2', 'berkeley']:
    snap = np.load('data/{}/data.npz'.format(data))
    print(data, sorted(set(snap['lengths']))[:5], snap['lengths'].max())
