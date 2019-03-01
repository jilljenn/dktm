import torch
from torch import optim
import torch.nn as nn
from random import randint
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from prepare import fraction
import numpy as np
import sys
import json
import argparse


parser = argparse.ArgumentParser(description='Run DKTM')
parser.add_argument('--batch_size', type=int, nargs='?', default=250)
parser.add_argument('--bptt', type=int, nargs='?', default=2)
parser.add_argument('--iter', type=int, nargs='?', default=2)
parser.add_argument('--data', type=str, nargs='?', default='sim')
parser.add_argument('--d', type=int, nargs='?', default=20)
options = parser.parse_args()


hidden_size = 20
learning_rate = 0.005


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
else:
    # Load Fraction dataset (or Assistments)
    actions, lengths, exercises, targets = fraction()
    nb_students = len(actions)
    nb_questions = 20


nb_distinct_actions = 1 + max(max(actions[i]) for i in range(nb_students))
nb_distinct_questions = 1 + max(max(exercises[i]) for i in range(nb_students))


UNTIL_TRAIN = round(0.8 * nb_students) if nb_students > 5 else 3
train_actions = actions[:UNTIL_TRAIN]
train_lengths = lengths[:UNTIL_TRAIN]
train_exercises = exercises[:UNTIL_TRAIN]
train_targets = targets[:UNTIL_TRAIN]
test_actions = actions[UNTIL_TRAIN:]
test_lengths = lengths[UNTIL_TRAIN:]
test_exercises = exercises[UNTIL_TRAIN:]
test_targets = targets[UNTIL_TRAIN:]


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          batch_first=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(
        #     embedded, input_lengths, batch_first=True)
        # outputs, hidden = self.gru(packed, hidden)
        outputs, hidden = self.gru(embedded, hidden)
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #     outputs, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        return embedded


class DKTM(nn.Module):
    def __init__(self, nb_distinct_actions, nb_distinct_questions,
                 hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        user_embedding = nn.Embedding(nb_distinct_actions, self.hidden_size)
        item_embedding = nn.Embedding(nb_distinct_questions, self.hidden_size)
        self.encoder = EncoderRNN(self.hidden_size, user_embedding)
        # Try factorization machines
        self.decoder = Decoder(item_embedding)

    def forward(self, actions, lengths, exercises, hidden=None):
        encoder_outputs, enc_hidden = self.encoder(actions, lengths, hidden)
        decoded = self.decoder(exercises)
        logits = torch.einsum('bsd,bsd->bs', encoder_outputs, decoded)
        return logits, enc_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.encoder.n_layers, batch_size,
                                self.encoder.hidden_size)


def sequence_mask(seq_len):
    max_len = max(seq_len)
    indexes = torch.arange(0, max_len)
    return torch.ByteTensor((indexes < lengths.unsqueeze(1)))


def criterion(inp, target, mask=None):
    cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
    loss = cross_entropy(inp, target).masked_select(mask).mean()
    # print('loss', loss)
    loss = loss.to(device)
    return loss  # mask.sum()


def eval(mode, logits, target, mask=None):
    proba = 1/(1 + np.exp(-logits.detach().numpy()))
    pred = np.round(proba)
    target0 = target.detach().numpy()

    # Should also take lengths into account
    np_mask = mask.detach().numpy()
    acc = ((pred == target0) * np_mask).sum() / np_mask.sum()
    # (flat_pred == flat_target).mean()
    if mode == 'test':
        try:
            auc = roc_auc_score(target0, pred)
            print('acc={:f} auc={:f}'.format(acc, auc))
        except ValueError:
            print('acc={:f}'.format(acc))


def get_batch(actions, lengths, exercises, targets, pos, t):
    return (actions[pos:pos + options.batch_size, t:t + options.bptt],
            torch.clamp(lengths[pos:pos + options.batch_size] - t,
                        0, options.bptt),
            exercises[pos:pos + options.batch_size, t:t + options.bptt],
            targets[pos:pos + options.batch_size, t:t + options.bptt])


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(train_actions, train_lengths, train_exercises, train_targets,
          optimizer):
    model.train()

    print('# TRAIN')
    for pos in range(0, len(train_actions), options.batch_size):  # .size()
        actual_batch_size = min(options.batch_size, len(train_actions) - pos)
        hidden = model.init_hidden(actual_batch_size)
        for t in range(0, train_lengths[pos], options.bptt):
            # Get batch limited to current time window (bptt)
            actions, lengths, exercises, targets = get_batch(
                train_actions, train_lengths, train_exercises, train_targets,
                pos, t)
            mask = sequence_mask(lengths)

            hidden = repackage_hidden(hidden)  # Detach previous hidden state
            optimizer.zero_grad()
            # Predict
            logits, hidden = model(actions, lengths, exercises, hidden)
            eval('train', logits, target, mask)

            loss = criterion(logits, targets, mask)
            loss.backward()
            optimizer.step()


def test(actions, lengths, exercises, targets):
    model.eval()

    print('# TEST')
    hidden = None
    with torch.no_grad():
        logits, hidden = model(actions, lengths, exercises, hidden)
        eval('test', logits, targets, mask=sequence_mask(lengths))


if __name__ == '__main__':
    # Model
    model = DKTM(nb_distinct_actions, nb_distinct_questions, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare data
    input_var = torch.LongTensor(train_actions).to(device)
    lengths = torch.LongTensor(train_lengths).to(device)
    eval_var = torch.LongTensor(train_exercises).to(device)
    target = torch.FloatTensor(train_targets).to(device)

    test_input_var = torch.LongTensor(test_actions).to(device)
    test_lengths = torch.LongTensor(test_lengths).to(device)
    test_eval_var = torch.LongTensor(test_exercises).to(device)
    test_target = torch.FloatTensor(test_targets).to(device)

    for _ in range(options.iter):  # Number of epochs
        train(input_var, lengths, eval_var, target, optimizer)
    test(input_var, lengths, eval_var, target)
    test(test_input_var, test_lengths, test_eval_var, test_target)


print(json.dumps(vars(options), indent=4))
