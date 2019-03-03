from collections import Counter
import pandas as pd
import numpy as np
import time
import json
import sys
import os


def fraction():
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
    lengths = [0] * nb_students
    exercises = [[] for _ in range(nb_students)]
    targets = [[] for _ in range(nb_students)]
    for i in range(nb_students):
        for j in range(nb_questions - 1):
            actions[i].append(encode_pair[(j, answers[i][j])])
        exercises[i] = np.arange(nb_questions - 1)
        lengths[i] = nb_questions - 1
        targets[i] = answers[i][1:]
    N, M = len(actions), len(actions[0])
    # print(len(exercises), len(exercises[0]))
    # print(len(targets), len(targets[0]))
    # sys.exit(0)
    return actions, lengths, exercises, targets


def assistments():
    # Load data
    if os.path.isfile('data/assistments09/data.npz'):
        snapshot = np.load('data/assistments09/data.npz')
        actions = snapshot['actions']
        lengths = snapshot['lengths']
        exercises = snapshot['exercises']
        targets = snapshot['targets']
        print('(nb_students, max_length) =', actions.shape,
              'min_length =', lengths.min())
        return actions, lengths, exercises, targets

    # df = pd.read_csv('/Users/jilljenn/code/seq/data/dummy/data.csv')
    dt = time.time()
    df = pd.read_csv('/Users/jilljenn/code/seq/data/assistments09/data.csv')
    print('loaded', time.time() - dt)

    dt = time.time()
    nb_samples_per_user = Counter(df['user'])
    max_length = max(nb_samples_per_user.values()) - 1
    print('max_length', max_length)
    nb_students = 1 + max(nb_samples_per_user.keys())
    # sys.exit(0)
    encode_pair = {}
    nb_codes = 1
    actions = np.zeros((nb_students, max_length), dtype=np.int32)
    lengths = np.array([-1] * nb_students)
    exercises = np.zeros((nb_students, max_length), dtype=np.int32)
    targets = np.zeros((nb_students, max_length), dtype=np.int32)

    data = np.array(df[['user', 'item', 'skill', 'correct']])
    for user, item, skill, correct in data:
        pair = (skill, correct)
        if pair not in encode_pair:
            encode_pair[pair] = nb_codes
            nb_codes += 1
        cursor = 1 + lengths[user]
        if cursor < nb_samples_per_user[user] - 1:
            actions[user][cursor] = encode_pair[pair]
        if cursor > 0:
            exercises[user][cursor - 1] = skill  # Shift exercises by 1
            targets[user][cursor - 1] = correct
        lengths[user] += 1
    # print(encode_pair)
    # print(actions, lengths, exercises, targets)
    print('computed', time.time() - dt)

    dt = time.time()
    np.save('data/assistments09/actions.npy', actions)
    np.save('data/assistments09/lengths.npy', lengths)
    np.save('data/assistments09/exercises.npy', exercises)
    np.save('data/assistments09/targets.npy', targets)
    np.savez('data/assistments09/data.npz', actions=actions, lengths=lengths,
             exercises=exercises, targets=targets)
    print('saved', time.time() - dt)
    # sys.exit(0)
    return actions, lengths, exercises, targets
