import numpy as np
import json


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
    lengths = [[] for _ in range(nb_students)]
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
    return actions, lengths, exercises, targets
