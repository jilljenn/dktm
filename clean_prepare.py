import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import json

"""
For each student i, actions[i] is the list of (question id, outcome) that this student made
For each question i, question_skills[i] is a binary vector of the skills of this question
"""
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]

def for_dktm(actions, question_skills=None):
    all_exercises = [
        [q] for student in actions for q, _ in student
    ]

    exercises = [
        [q for q, _ in student[1:]]
        for student in actions
    ]
    targets = [
        [int(o) for _, o in student[1:]]
        for student in actions
    ]

    if question_skills is not None:
        actions = [
            [(''.join(map(str, question_skills[q])), o) for (q, o) in student]
            for student in actions
        ]

    uniq_actions = set().union(*actions)
    uniq_actions = dict(zip(sorted(uniq_actions), range(len(uniq_actions))))

    actions = [
        [uniq_actions[action] for action in student[:-1]]
        for student in actions
    ]
    lengths = [len(student) for student in actions]
    indices = []
    x = 1
    for student in actions:
        y = len(student)
        indices.append(list(range(x, x+y)))
        x += y + 1

    # flatten_exercises = [[a] for l in exercises for a in l]
    # print('all actions', actions)
    # flatten_exercises += [[0]]

    max_length = max(lengths)
    for i in range(len(actions)):
        for liste in [actions, exercises, targets, indices]:
            liste[i].extend([0] * (max_length - len(liste[i])))

    enc = OneHotEncoder()
    metadata = enc.fit_transform(all_exercises)
    print('wut metadata', metadata[:1], metadata[:1].sum())
    print('test', metadata.shape, len(actions) * (len(actions[0]) + 1))
    '''for liste in [actions, lengths, exercises, targets, indices]:
                    print(namestr(liste, locals()), liste[:3], len(liste))'''

    actions, lengths, exercises, targets, indices = map(np.array, [actions, lengths, exercises, targets, indices])

    # Everything has same length
    assert len(set(map(len, [actions, lengths, exercises, targets, indices]))) == 1



    return actions, lengths, exercises, targets, metadata, indices

def fraction(decoder):
    # Load data
    with open('data/fraction/fraction.json') as f:
        answers = np.array(json.load(f)['student_data'], dtype=np.int32)
    nb_students, nb_questions = answers.shape
    with open('data/fraction/qmatrix-fraction.json') as f:
        q = np.array(json.load(f)['Q'], dtype=np.int32)
        q_sp = csr_matrix(q)

    actions, lengths, exercises, targets, _, indices = for_dktm(
        list(map(lambda x: list(zip(range(len(x)), x)), answers)),
        q
    )

    features = []
    bonus_features = []
    for i in range(nb_students):
        w = 0 * q[0]
        f = 0 * q[0]
        for j in range(nb_questions):
            features.append(j)
            bonus_features.append(np.concatenate((w, f)))
    features = np.array(features).reshape(-1, 1)
    bonus_features = np.array(bonus_features)
    # Encode metadata
    enc = OneHotEncoder()
    enc.fit(features)
    if decoder == 'i':
        metadata = enc.transform(features)
    elif decoder == 's':
        metadata = q_sp[features.squeeze()]
    elif decoder == 'is':
        metadata = enc.transform(features)
        metadata = hstack((metadata, q_sp[features.squeeze()])).tocsr()
    elif decoder == 'swf':
        metadata = hstack((q_sp[features.squeeze()], bonus_features)).tocsr()
    elif decoder == 'iswf':
        metadata = enc.transform(features)
        metadata = hstack((metadata, q_sp[features.squeeze()], bonus_features)).tocsr()

    print(len(actions), len(actions[0]), metadata.shape, len(indices), 536 * 20)

    return np.array(actions), np.array(lengths), np.array(exercises), np.array(targets), metadata, np.array(indices)

def attempts(decoder):
    df = pd.read_csv('data/attempts/attempts.csv')
    nb_students = df['student'].nunique()

    df_grouped = pd.DataFrame({
        'problems': df.groupby('student')['problem'].apply(list),
        'outcomes': df.groupby('student')['solved'].apply(list)
    })
    # print(df_grouped.head())

    actions, lengths, exercises, targets, metadata, indices = for_dktm(
        df_grouped.apply(lambda x: list(zip(x.problems, x.outcomes)), axis=1).tolist()
    )

    return actions, lengths, exercises, targets, metadata, indices



if __name__ == "__main__":
    actions, lengths, exercises, targets, metadata, indices = fraction('i')

    # actions, _, exercises, _, _, _ = attempts('s')
    print(len(actions), len(actions[0]), metadata.shape, indices[:2])
    print(max(indices[-1]))
    print(exercises)
