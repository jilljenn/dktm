import torch
from torch import optim
import torch.nn as nn
from random import randint
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from prepare import fraction, assistments
from datetime import datetime
import numpy as np
import sys
import json
import argparse


parser = argparse.ArgumentParser(description='Run DKTM')
parser.add_argument('--batch_size', type=int, nargs='?', default=100)
parser.add_argument('--bptt', type=int, nargs='?', default=500)
parser.add_argument('--iter', type=int, nargs='?', default=200)
parser.add_argument('--data', type=str, nargs='?', default='assistments')
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--lr', type=float, nargs='?', default=0.005)
parser.add_argument('--reg', type=float, nargs='?', default=5e-4)
parser.add_argument('--layers', type=int, nargs='?', default=2)
parser.add_argument('--dropout', type=float, nargs='?', default=0.)
parser.add_argument('--train', type=bool, nargs='?', const=True,
                    default=False)
parser.add_argument('--no_bias', type=bool, nargs='?', const=True,
                    default=False)
options = parser.parse_args()
print(options)


hidden_size = options.d
learning_rate = options.lr
metrics = defaultdict(list)


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print('device', device)


def gen_int(n, m, max_val):
    return np.random.randint(max_val, size=(n, m))


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
    actions, lengths, exercises, targets, metadata, indices = fraction()
else:
    # Load Assistments dataset
    actions, lengths, exercises, targets, metadata, indices = assistments()


nb_students = len(actions)
nb_distinct_actions = 1 + max(v for line in actions for v in line)
nb_distinct_questions = 1 + max(v for line in exercises for v in line)
nb_distinct_features = metadata.shape[1]
print(nb_students, 'students', nb_distinct_actions, 'actions',
      nb_distinct_questions, 'questions', nb_distinct_features, 'features')


UNTIL_TRAIN = (round(0.8 * nb_students)
               if nb_students > 5 and not options.train else 1)
train_actions = actions[:UNTIL_TRAIN]
train_lengths = lengths[:UNTIL_TRAIN]
train_exercises = exercises[:UNTIL_TRAIN]
train_targets = targets[:UNTIL_TRAIN]
train_indices = indices[:UNTIL_TRAIN]
# print(train_lengths)
if options.train:
    test_actions = train_actions
    test_lengths = train_lengths
    test_exercises = train_exercises
    test_targets = train_targets
    test_indices = train_indices
else:
    test_actions = actions[UNTIL_TRAIN:]
    test_lengths = lengths[UNTIL_TRAIN:]
    test_exercises = exercises[UNTIL_TRAIN:]
    test_targets = targets[UNTIL_TRAIN:]
    test_indices = indices[UNTIL_TRAIN:]


class EncoderRNN(nn.Module):
    # 3 0.4 / 2 0.0 good for Fraction
    def __init__(self, hidden_size, embedding, n_layers=options.layers,
                 dropout=options.dropout):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
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
        outputs, hidden = self.gru(embedded, hidden)
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #     outputs, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, nb_distinct_questions,
                 nb_distinct_features):
        super().__init__()
        self.hidden_size = hidden_size
        self.item_embedding = nn.Embedding(
            nb_distinct_questions, self.hidden_size)
        self.item_bias = nn.Embedding(nb_distinct_questions, 1)
        self.feat_embedding = nn.Embedding(
            nb_distinct_features, self.hidden_size)
        self.feat_bias = nn.Embedding(nb_distinct_features, 1)

    def forward(self, exercises, indices):
        # decoded = self.item_embedding(exercises)
        # decoded_bias = self.item_bias(exercises).squeeze()
        # print('metadata', metadata.shape)
        # print('indices', indices.shape)
        # print('indices view', indices.flatten().shape)
        # print('max indice', max(v for line in indices for v in line))
        # print(*indices.shape, -1)
        batch_metadata = (torch.index_select(metadata, 0, indices.flatten())
                               .view(*indices.shape, -1))
        decoded_bias = (batch_metadata @ self.feat_bias.weight).squeeze()
        decoded = batch_metadata @ self.feat_embedding.weight
        # sys.exit(0)
        return decoded, decoded_bias


class DKTM(nn.Module):
    def __init__(self, nb_distinct_actions, nb_distinct_questions,
                 hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        action_embedding = nn.Embedding(nb_distinct_actions, self.hidden_size)
        action_embedding.weight.requires_grad = False

        # print('user emb', user_embedding.device)
        # print('item emb', item_embedding.device)
        self.encoder = EncoderRNN(self.hidden_size, action_embedding)
        # Try factorization machines
        self.decoder = Decoder(self.hidden_size, nb_distinct_questions,
                               nb_distinct_features)
        # self.decoder = Decoder(item_embedding)

    def forward(self, actions, lengths, exercises, indices, hidden=None):
        # print('act', actions.device)
        # print('len', lengths.device)
        encoder_outputs, enc_hidden = self.encoder(actions, lengths, hidden)
        decoded, decoded_bias = self.decoder(exercises, indices)
        # print('bias', decoded_bias.shape)
        logits = torch.einsum('bsd,bsd->bs', encoder_outputs, decoded)
        # print('logits', logits.shape)
        if not options.no_bias:
            logits += decoded_bias
        return logits, enc_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.encoder.n_layers, batch_size,
                                self.encoder.hidden_size, device=device)


def sequence_mask(lengths, max_len):
    indexes = torch.arange(0, max_len).to(device)
    # print('dev', indexes.device)
    # print('dev', lengths.device)
    return (indexes < lengths.unsqueeze(1)).byte()


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

        pred0 = pred.masked_select(mask).cpu().numpy()
        target0 = target.masked_select(mask).cpu().numpy()
        acc = (pred0 == target0).mean()
        try:
            auc = roc_auc_score(target0, pred0)
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
            eval('train', logits, targets, mask)

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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=options.reg)

    # Prepare data
    metadata = torch.FloatTensor(metadata.todense()).to(device)

    train_actions = torch.LongTensor(train_actions).to(device)
    train_lengths = torch.LongTensor(train_lengths).to(device)
    train_exercises = torch.LongTensor(train_exercises).to(device)
    train_targets = torch.FloatTensor(train_targets).to(device)
    train_indices = torch.LongTensor(train_indices).to(device)

    test_actions = torch.LongTensor(test_actions).to(device)
    test_lengths = torch.LongTensor(test_lengths).to(device)
    test_exercises = torch.LongTensor(test_exercises).to(device)
    test_targets = torch.FloatTensor(test_targets).to(device)
    test_indices = torch.LongTensor(test_indices).to(device)

    for epoch in range(options.iter):  # Number of epochs
        print('Epoch', epoch)
        train(train_actions, train_lengths, train_exercises, train_targets,
              train_indices, optimizer)
        test(test_actions, test_lengths, test_exercises, test_targets,
             test_indices)


timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
with open('logs/output-{}.json'.format(timestamp), 'w') as f:
    f.write(json.dumps({
        'options': vars(options),
        'metrics': metrics
    }, indent=4))
