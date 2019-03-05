from scipy.sparse import load_npz, save_npz, hstack

X_swf = load_npz('data/berkeley/metadata-swf.npz')
print(X_swf.shape)
# X_e = load_npz('data/assistments2/bonus.npz')
# X = hstack((X_swf, X_e))
# save_npz('data/assistments2/metadata-swfe.npz', X)

X_s = X_swf[:, :29]
print(X_s.shape)
save_npz('data/berkeley/metadata-s.npz', X_s)
