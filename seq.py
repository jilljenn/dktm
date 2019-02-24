import torch
from torch import optim
import torch.nn as nn
from random import randint
from collections import defaultdict
import numpy as np
import sys
import json


NUM_WORDS = 50
HIDDEN_SIZE = 20
NB_EPOCHS = 50
learning_rate = 0.005
DATA = 'real'


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# print(device)


def sim(n, m):
    return [list(range(randint(0, 5), 5 * m + 1, randint(1, 5)))[:m]
            for _ in range(n)]


def gen_seq(n, m):
    Xy = np.array(sim(n, m + 1))
    X = Xy[:, :-1]
    y = Xy[:, -1]
    return X, y


def gen_int(n, m, max_val=NUM_WORDS):
    return np.random.randint(max_val, size=(n, m))


if DATA == 'sim':
    N = nb_students = 5
    M = nb_questions = 3
    actions = gen_int(N, M)
    exercises = gen_int(N, M)
    targets = gen_int(N, M, 2)
else:
    # Load data
    with open('/Users/jilljenn/code/qna/data/fraction.json') as f:
        answers = np.array(json.load(f)['student_data'], dtype=np.int32)
    nb_students, nb_questions = answers.shape
    with open('/Users/jilljenn/code/qna/data/qmatrix-fraction.json') as f:
        q = np.array(json.load(f)['Q'], dtype=np.int32)
    # Encode all actions
    skills = set()
    for line in q:
        code = ''.join(map(str, line))
        skills.add(code + '0')
        skills.add(code + '1')
    encode = dict(zip(skills, range(10000)))
    # Encode pairs
    encode_pair = {}
    for j in range(nb_questions):
        code = ''.join(map(str, q[j]))
        encode_pair[(j, 0)] = encode[code + '0']
        encode_pair[(j, 1)] = encode[code + '1']
    # Encode actions per user
    actions = [[] for _ in range(nb_students)]
    exercises = [[] for _ in range(nb_students)]
    targets = [[] for _ in range(nb_students)]
    for i in range(nb_students):
        for j in range(nb_questions - 1):
            actions[i].append(encode_pair[(j, answers[i][j])])
        exercises[i] = np.arange(nb_questions - 1)
        targets[i] = answers[i][1:]
    N, M = len(actions), len(actions[0])
    # print(len(exercises), len(exercises[0]))
    # print(len(targets), len(targets[0]))

UNTIL_TRAIN = round(0.8 * nb_students)
train_actions = actions[:UNTIL_TRAIN]
train_exercises = exercises[:UNTIL_TRAIN]
train_targets = targets[:UNTIL_TRAIN]
test_actions = actions[UNTIL_TRAIN:]
test_exercises = exercises[UNTIL_TRAIN:]
test_targets = targets[UNTIL_TRAIN:]


# print(actions)
# print(exercises)
# print(targets)


# sys.exit(0)


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
        # outputs = (outputs[:, :, :self.hidden_size] +
        #            outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        return embedded


def maskNLLLoss(inp, target, mask=None):
    # print('target', target.view(-1, 1).shape)

    # print('input', inp.shape)
    # nTotal = mask.sum()
    # crossEntropy = -torch.log(torch.gather(inp, 0, target).squeeze(1))
    cross_entropy = nn.BCEWithLogitsLoss()
    # loss = crossEntropy.masked_select(mask).mean()
    loss = cross_entropy(inp, target)
    loss = loss.to(device)
    return loss  # , nTotal.item()


def predict(input_var, lengths, target, encoder, decoded):
    # print('oh oh', input_var.shape, lengths.shape)
    encoder_outputs, encoder_hidden = encoder(input_var, lengths)
    # print('enc', encoder_outputs.shape)
    # print('dec', decoded.shape)

    logits = torch.einsum('bsd,bsd->bs', encoder_outputs, decoded)
    # print('logits', logits)
    proba = 1/(1 + np.exp(-logits.detach().numpy()))
    pred = np.round(proba)
    # print('proba', pred)
    target0 = target.detach().numpy()
    # print(pred.shape)
    # print(type(pred))
    # print(target0.shape)
    # print(type(target0))
    acc = (pred == target0).astype(np.int32).sum() / len(target0.flatten())
    print('acc', acc)
    # print('target', target)
    return logits, proba, pred, acc


def train(input_var, lengths, target, encoder, decoded,
          encoder_optimizer, decoder_optimizer):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_var.to(device)
    lengths.to(device)

    print('# TRAIN')
    logits, proba, pred, acc = predict(input_var, lengths, target,
                                       encoder, decoded)
    # sys.exit(0)

    loss = 0
    mask_loss = maskNLLLoss(logits, target)
    # print('loss', mask_loss.detach().numpy())
    loss += mask_loss

    loss.backward(retain_graph=True)

    encoder_optimizer.step()
    decoder_optimizer.step()


def test(input_var, lengths, target, encoder, decoded):
    input_var.to(device)
    lengths.to(device)

    print('# TEST')
    logits, proba, pred, acc = predict(input_var, lengths, target,
                                       encoder, decoded)


if __name__ == '__main__':
    user_embedding = nn.Embedding(NUM_WORDS, HIDDEN_SIZE)
    item_embedding = nn.Embedding(NUM_WORDS, HIDDEN_SIZE)
    encoder = EncoderRNN(HIDDEN_SIZE, user_embedding)
    decoder = Decoder(item_embedding)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    # Prepare data
    input_var = torch.LongTensor(train_actions)
    eval_var = torch.LongTensor(train_exercises)
    target = torch.FloatTensor(train_targets)
    test_input_var = torch.LongTensor(test_actions)
    test_eval_var = torch.LongTensor(test_exercises)
    test_target = torch.FloatTensor(test_targets)

    decoded = decoder(eval_var)
    test_decoded = decoder(test_eval_var)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    for _ in range(NB_EPOCHS):
        train(input_var, torch.tensor([M] * len(input_var)), target, encoder,
              decoded, encoder_optimizer, decoder_optimizer)
        test(test_input_var, torch.tensor([M] * len(test_input_var)),
             test_target, encoder, test_decoded)
