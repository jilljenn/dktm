import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from prepare import fraction, assistments, assistments2, berkeley
from datetime import datetime
import time
import os
import numpy as np
import sys
import json
import argparse

import random
random.seed(42)
torch.manual_seed(42)
optim, nn = torch.optim, torch.nn
rs = np.random.RandomState(42)
torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser(description='Run DKTM')
parser.add_argument('--batch_size', type=int, nargs='?', default=100)
parser.add_argument('--bptt', type=int, nargs='?', default=100)
parser.add_argument('--iter', type=int, nargs='?', default=200)
parser.add_argument('--data', type=str, nargs='?', default='assistments')
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--lr', type=float, nargs='?', default=0.005)
parser.add_argument('--reg', type=float, nargs='?', default=5e-4)
parser.add_argument('--layers', type=int, nargs='?', default=1)
parser.add_argument('--dropout', type=float, nargs='?', default=0.)
parser.add_argument('--train', type=bool, nargs='?', const=True,
                    default=False)
parser.add_argument('--no_bias', type=bool, nargs='?', const=True,
                    default=False)
parser.add_argument('--i', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--no_enc', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--decoder', type=str, nargs='?', default='s')
parser.add_argument('--dkt', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--learned', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--baseline', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--fold', type=int, nargs='?', default=0)
parser.add_argument('--last', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--rnn', type=str, nargs='?', default='gru')
parser.add_argument('--randomize', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()
print(options)


hidden_size = options.d
learning_rate = options.lr
metrics = defaultdict(list)


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print('device', device)


def gen_int(n, m, max_val):
    return rs.randint(max_val, size=(n, m))


if options.data == 'sim':
    nb_students = 5
    nb_questions = 3
    actions = gen_int(nb_students, nb_questions, 42)
    lengths = [nb_questions] * nb_students
    exercises = gen_int(nb_students, nb_questions, 20)
    targets = gen_int(nb_students, nb_questions, 2)
elif options.data == 'dummy':
    nb_students = 4
    nb_questions = 2
    inverse_dict = {
        (1, 1): 0,
        (2, 0): 1,
        (1, 0): 2,
        (0, 0): 3
    }
    actions = [
        list(map(inverse_dict.get, [(1, 1), (2, 0)])),
        list(map(inverse_dict.get, [(1, 0), (2, 0)])),
        list(map(inverse_dict.get, [(1, 0), (0, 0)])),
        list(map(inverse_dict.get, [(1, 1), (0, 0)]))
    ]
    lengths = [2, 2, 1, 1]
    exercises = [
        [2, 2],
        [2, 2],
        [2, 0],
        [2, 0]
    ]
    targets = [
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
elif options.data == 'fraction':
    # Load Fraction dataset (or Assistments)
    actions, lengths, exercises, targets, metadata, indices = fraction(options.decoder)
elif options.data == 'assistments':
    # Load Assistments 200k dataset
    actions, lengths, exercises, targets, metadata, indices = assistments()
elif options.data == 'assistments2':
    # Load Assistments 300k dataset
    actions, lengths, exercises, targets, metadata, indices = assistments2(options.decoder)
elif options.data == 'berkeley':
    # Load Berkeley 560k dataset
    actions, lengths, exercises, targets, metadata, indices = berkeley(options.decoder)


nb_students = len(actions)
nb_distinct_actions = 1 + max(v for line in actions for v in line)
nb_distinct_questions = 1 + max(v for line in exercises for v in line)
nb_distinct_features = metadata.shape[1]
print(nb_students, 'students', nb_distinct_actions, 'actions',
      nb_distinct_questions, 'questions', nb_distinct_features, 'features')

# sys.exit(0)
NB_FOLDS = 5
FOLD_FILE = 'data/{}/folds.npy'.format(options.data)
if not os.path.isfile(FOLD_FILE):
    students = np.arange(nb_students)
    rs.shuffle(students)
    size = nb_students // NB_FOLDS
    folds = [students[i * size:(i + 1) * size] for i in range(NB_FOLDS - 1)] + [students[(NB_FOLDS - 1) * size:]]
    np.save(FOLD_FILE, folds)
    print('Computed folds')
else:
    folds = np.load(FOLD_FILE, allow_pickle=True)
    print('Loaded folds')
    print([folds[i].shape for i in range(NB_FOLDS)])
# sys.exit(0)


# UNTIL_TRAIN = (round(0.8 * nb_students)
#                if nb_students > 5 and not options.train else 1)
i_test = folds[options.fold]
i_train = list(set(range(nb_students)) - set(i_test))
print(len(i_train), 'train', len(i_test), 'test')
train_actions = actions[i_train]
train_lengths = lengths[i_train]
train_exercises = exercises[i_train]
train_targets = targets[i_train]
train_indices = indices[i_train]
# print(train_lengths)
if options.train:
    test_actions = train_actions = actions[:1]
    test_lengths = train_lengths = lengths[:1]
    test_exercises = train_exercises = exercises[:1]
    test_targets = train_targets = targets[:1]
    test_indices = train_indices = indices[:1]
else:
    test_actions = actions[i_test]
    test_lengths = lengths[i_test]
    test_exercises = exercises[i_test]
    test_targets = targets[i_test]
    test_indices = indices[i_test]


class EncoderRNN(nn.Module):
    # 3 0.4 / 2 0.0 good for Fraction
    def __init__(self, hidden_size, embedding, n_layers=options.layers,
                 dropout=options.dropout, rnn=options.rnn):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        if rnn == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
                               dropout=(0 if n_layers == 1 else dropout),
                               batch_first=True)
        else:
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout),
                              batch_first=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # print('inp seq', input_seq.device)
        embedded = self.embedding(input_seq)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(
        #     embedded, input_lengths, batch_first=True)
        # outputs, hidden = self.gru(packed, hidden)
        # print('emb', embedded.device)
        # print('hid', hidden.device)
        outputs, hidden = self.rnn(embedded, hidden)
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #     outputs, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, nb_distinct_questions,
                 nb_distinct_features):
        super().__init__()
        self.hidden_size = hidden_size
        # self.item_embedding = nn.Embedding(
        #     nb_distinct_questions, self.hidden_size)
        if options.i:
            self.item_bias = nn.Embedding(nb_distinct_questions, 1)
        if options.dkt:
            self.feat_embedding = nn.Embedding(
                nb_distinct_features, self.hidden_size)
        self.feat_bias = nn.Embedding(nb_distinct_features, 1)
        self.linear = nn.Linear(nb_distinct_features, 1)

    def forward(self, exercises, indices):
        decoded = None
        decoded_bias = None
        # decoded = self.item_embedding(exercises)
        # print('metadata', metadata.shape)
        # print('indices', indices.shape)
        # print('indices view', indices.flatten().shape)
        # print('max indice', max(v for line in indices for v in line))
        # print(*indices.shape, -1)
        batch_metadata = (torch.index_select(metadata, 0, indices.flatten())
                               .view(*indices.shape, -1))
        if options.dkt:
            decoded = batch_metadata @ self.feat_embedding.weight
        else:
            # decoded_bias0 = (batch_metadata @ self.feat_bias.weight).squeeze(2)
            # print(decoded_bias0.shape)
            decoded_bias = self.linear(batch_metadata).squeeze(2)
            # print(decoded_bias.shape)
            # sys.exit(0)
            if options.i:
                decoded_bias += self.item_bias(exercises).squeeze(2)
        # sys.exit(0)
        return decoded, decoded_bias


class DKTM(nn.Module):
    def __init__(self, nb_distinct_actions, nb_distinct_questions,
                 hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        action_embedding = nn.Embedding(nb_distinct_actions, self.hidden_size)
        if not options.learned:
            action_embedding.weight.requires_grad = False

        # print('user emb', user_embedding.device)
        # print('item emb', item_embedding.device)
        self.encoder = EncoderRNN(self.hidden_size, action_embedding)
        # Try factorization machines
        self.decoder = Decoder(self.hidden_size, nb_distinct_questions,
                               nb_distinct_features)
        # self.decoder = Decoder(item_embedding)
        self.bias_encoder = nn.Linear(hidden_size, 1)

    def forward(self, actions, lengths, exercises, indices, hidden=None):
        # print('act', actions.device)
        # print('len', lengths.device)
        encoder_outputs, enc_hidden = self.encoder(actions, lengths, hidden)
        # print('enc out', encoder_outputs.shape)
        encoded_bias = self.bias_encoder(encoder_outputs).squeeze(2)
        # print('enc bias', encoded_bias.shape)
        # sys.exit(0)
        decoded, decoded_bias = self.decoder(exercises, indices)
        # print('bias', decoded_bias.shape)
        if options.dkt:
            logits = torch.einsum('bsd,bsd->bs', encoder_outputs, decoded)
        else:
            if options.no_enc:
                logits = decoded_bias
            else:
                logits = encoded_bias
                # print('logits', logits.shape)
                if not options.no_bias:
                    logits += decoded_bias
        return logits, enc_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if options.rnn == 'lstm':
            return (weight.new_zeros(self.encoder.n_layers, batch_size,
                                     self.encoder.hidden_size, device=device),
                    weight.new_zeros(self.encoder.n_layers, batch_size,
                                     self.encoder.hidden_size, device=device))
        return weight.new_zeros(self.encoder.n_layers, batch_size,
                                self.encoder.hidden_size, device=device)


def sequence_mask(lengths, max_len):
    indexes = torch.arange(0, max_len).to(device)
    # print('dev', indexes.device)
    # print('dev', lengths.device)
    return (indexes < lengths.unsqueeze(1)) # .byte()


def criterion(inp, target, mask=None):
    cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
    # print('inp', inp.shape)
    # print('target', target.shape)
    # print('mask', mask.shape)
    loss = cross_entropy(inp, target).masked_select(mask).mean()
    # print('loss', loss)
    loss = loss.to(device)
    return loss  # mask.sum()


def eval(mode, logits, target, mask=None):
    with torch.no_grad():
        proba = 1/(1 + torch.exp(-logits))
        pred = proba.round()

        proba0 = proba.masked_select(mask).cpu().numpy()
        pred0 = pred.masked_select(mask).cpu().numpy()
        target0 = target.masked_select(mask).cpu().numpy()
        acc = (pred0 == target0).mean()
        try:
            auc = roc_auc_score(target0, proba0)
        except ValueError:
            auc = -1
        metrics[mode + ' acc'].append(acc)
        metrics[mode + ' auc'].append(auc)
        if mode == 'test':
            print(mode, 'acc={:f} auc={:f}'.format(acc, auc))


def get_batch(actions, lengths, exercises, targets, indices, pos, t):
    return (actions[pos:pos + options.batch_size, t:t + options.bptt],
            torch.clamp(lengths[pos:pos + options.batch_size] - t,
                        0, options.bptt),
            exercises[pos:pos + options.batch_size, t:t + options.bptt],
            targets[pos:pos + options.batch_size, t:t + options.bptt],
            indices[pos:pos + options.batch_size, t:t + options.bptt])


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(train_actions, train_lengths, train_exercises, train_targets,
          train_indices, optimizer):
    model.train()

    for pos in range(0, len(train_actions), options.batch_size):  # .size()
        actual_batch_size = min(options.batch_size, len(train_actions) - pos)
        hidden = model.init_hidden(actual_batch_size)
        for t in range(0, train_lengths[pos], options.bptt):
            # Get batch limited to current time window (bptt)
            actions, lengths, exercises, targets, indices = get_batch(
                train_actions, train_lengths, train_exercises, train_targets,
                train_indices, pos, t)
            mask = sequence_mask(lengths, targets.shape[1])
            # print('train', lengths.shape, mask.shape)

            hidden = repackage_hidden(hidden)  # Detach previous hidden state
            optimizer.zero_grad()
            # Predict
            logits, hidden = model(actions, lengths, exercises, indices,
                                   hidden)
            # print('triste', logits.shape, targets.shape, mask.shape)
            # If we want to log train metrics
            # eval('train', logits, targets, mask)

            loss = criterion(logits, targets, mask)
            loss.backward()
            optimizer.step()


def test(actions, lengths, exercises, targets, indices):
    model.eval()

    hidden = None
    with torch.no_grad():
        logits, hidden = model(actions, lengths, exercises, indices, hidden)
        # print('test', lengths, sequence_mask(lengths, targets.shape[1]))
        mask = sequence_mask(lengths, targets.shape[1])
        eval('test', logits, targets, mask)


if __name__ == '__main__':
    # Model
    model = DKTM(nb_distinct_actions,
                 nb_distinct_questions, hidden_size).to(device)
    nb_params = 0
    nb_opt_params = 0
    for name, parameter in model.named_parameters():
        nb_params += 1
        nb_opt_params += parameter.requires_grad
        if parameter.requires_grad:
            print('hop', name)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(nb_params, 'variables')
    print(nb_opt_params, 'variables requiring gradient')
    print(pytorch_total_params, 'total parameters requiring gradient')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=options.reg)

    # Prepare data
    metadata = metadata.tocsr()

    train_actions = torch.LongTensor(train_actions).to(device)
    train_lengths = torch.LongTensor(train_lengths).to(device)
    train_exercises = torch.LongTensor(train_exercises).to(device)
    train_targets = torch.FloatTensor(train_targets).to(device)
    train_indices = torch.LongTensor(train_indices).to(device)
    train_mask = sequence_mask(train_lengths, train_targets.shape[1])
    X_train = metadata[train_indices.masked_select(train_mask).cpu().numpy()]
    y_train = train_targets.masked_select(train_mask).detach().cpu().numpy()
    print(X_train.shape)

    test_actions = torch.LongTensor(test_actions).to(device)
    test_lengths = torch.LongTensor(test_lengths).to(device)
    test_exercises = torch.LongTensor(test_exercises).to(device)
    test_targets = torch.FloatTensor(test_targets).to(device)
    test_indices = torch.LongTensor(test_indices).to(device)
    test_mask = sequence_mask(test_lengths, test_targets.shape[1])
    X_test = metadata[test_indices.masked_select(test_mask).cpu().numpy()]
    y_test = test_targets.masked_select(test_mask).detach().cpu().numpy()
    print(X_test.shape)

    # print('petit test', train_exercises.masked_select(train_mask).cpu().numpy()[:5])
    # print('petit test', X_train[:5])
    # print('petit test', test_exercises.masked_select(test_mask).cpu().numpy()[:5])
    # print('petit test', X_test[:5])
    # sys.exit(0)

    # Baseline
    if options.baseline:
        dt = time.time()
        baseline = LogisticRegression()  # Has L2 regularization by default
        baseline.fit(X_train, y_train)
        print('learned bias', baseline.intercept_)
        print('norm of coef weights', np.linalg.norm(baseline.coef_))
        print('fit', time.time() - dt)

        dt = time.time()
        y_pred_train = baseline.predict_proba(X_train)[:, 1]
        print('Train predict:', y_pred_train)
        print('Train was:', y_train)
        print('Train ACC:', np.mean(y_train == np.round(y_pred_train)))
        print('Train AUC:', roc_auc_score(y_train, y_pred_train))
        print('predict', time.time() - dt)

        dt = time.time()
        y_pred_test = baseline.predict_proba(X_test)[:, 1]
        print('Test predict:', y_pred_test)
        print('Test was:', y_test)
        print('Test ACC:', np.mean(y_test == np.round(y_pred_test)))
        try:
            print('Test AUC:', roc_auc_score(y_test, y_pred_test))
        except ValueError:
            pass
        print('predict', time.time() - dt)
        sys.exit(0)

    metadata = torch.FloatTensor(metadata.todense()).to(device)

    for epoch in range(options.iter):  # Number of epochs
        if epoch % 10 == 0:
            print('Epoch', epoch)
        # Randomize batches
        if options.randomize:
            shuffle = rs.permutation(len(train_actions))
            train_actions = train_actions[shuffle]
            train_lengths = train_lengths[shuffle]
            train_exercises = train_exercises[shuffle]
            train_targets = train_targets[shuffle]
            train_indices = train_indices[shuffle]
        train(train_actions, train_lengths, train_exercises, train_targets,
              train_indices, optimizer)
        if not options.last and epoch % 10 == 0:
            test(test_actions, test_lengths, test_exercises, test_targets,
                 test_indices)
    print('Results')
    test(test_actions, test_lengths, test_exercises, test_targets,
         test_indices)


timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
with open('logs/output-{}.json'.format(timestamp), 'w') as f:
    f.write(json.dumps({
        'options': vars(options),
        'metrics': metrics
    }, indent=4))
