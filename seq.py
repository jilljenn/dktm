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


NUM_WORDS = 50
HIDDEN_SIZE = 20
NB_EPOCHS = 50
learning_rate = 0.005


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print('device', device)


def sim(n, m):  # Should try simulated data of uneven length
    return [list(range(randint(0, 5), 5 * m + 1, randint(1, 5)))[:m]
            for _ in range(n)]


def gen_seq(n, m):
    Xy = np.array(sim(n, m + 1))
    X = Xy[:, :-1]
    y = Xy[:, -1]
    return X, y


def gen_int(n, m, max_val=NUM_WORDS):
    return np.random.randint(max_val, size=(n, m))


if options.data == 'sim':
    N = nb_students = 5
    M = nb_questions = 3
    actions = gen_int(N, M)
    lengths = [nb_questions] * nb_students
    exercises = gen_int(N, M)
    targets = gen_int(N, M, 2)
else:
    # Load Fraction dataset (or Assistments)
    actions, lengths, exercises, targets = fraction()
    nb_students = len(actions)


UNTIL_TRAIN = round(0.8 * nb_students)
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
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        return embedded


class DKTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        user_embedding = nn.Embedding(NUM_WORDS, HIDDEN_SIZE)
        item_embedding = nn.Embedding(NUM_WORDS, HIDDEN_SIZE)
        self.encoder = EncoderRNN(HIDDEN_SIZE, user_embedding)
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


def criterion(inp, target, mask=None):
    # nTotal = mask.sum()
    # crossEntropy = -torch.log(torch.gather(inp, 0, target).squeeze(1))
    cross_entropy = nn.BCEWithLogitsLoss()
    # loss = crossEntropy.masked_select(mask).mean()
    loss = cross_entropy(inp, target)
    loss = loss.to(device)
    return loss  # , nTotal.item()


def predict_and_eval(mode, model, actions, lengths, eval_var, target,
                     hidden=None):
    logits, hidden = model(actions, lengths, eval_var, hidden)

    proba = 1/(1 + np.exp(-logits.detach().numpy()))
    pred = np.round(proba)
    target0 = target.detach().numpy()

    acc = (pred == target0).astype(np.int32).sum() / len(target0.flatten())
    if mode == 'test':
        print('acc={:f} auc={:f}'.format(acc, roc_auc_score(target0, pred)))
    return logits, hidden, proba, pred, acc


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

            hidden = repackage_hidden(hidden)  # Detach previous hidden state
            optimizer.zero_grad()
            logits, hidden, proba, pred, acc = predict_and_eval(
                'train', model, actions, lengths, exercises, targets, hidden)

            loss = criterion(logits, targets)
            # print('loss', loss.detach().numpy())
            loss.backward()
            optimizer.step()


def test(actions, lengths, exercises, targets):
    model.eval()

    print('# TEST')
    with torch.no_grad():
        logits, hidden, proba, pred, acc = predict_and_eval(
            'test', model, actions, lengths, exercises, targets)


if __name__ == '__main__':
    # Model
    model = DKTM(HIDDEN_SIZE)
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
