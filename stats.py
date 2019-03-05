import numpy as np
import re

r1 = re.compile('Test ACC: (.*)')
r2 = re.compile('Test AUC: (.*)')

lines = []

for model in ['ktm', 'pfa', 'ktm-big', 'pfa-big']:
    name = 'assistments' if 'big' in model else 'fraction'
    with open('{}.txt'.format(model)) as f:
        corpus = f.read()

    acc = []
    for line in r1.findall(corpus):
        acc.append(float(line))

    auc = []
    for line in r2.findall(corpus):
        auc.append(float(line))
    lines.append((name, model, 'acc =', np.mean(acc[:5]).round(3), 'auc =', np.mean(auc[:5]).round(3)))
    if len(acc) > 5:
        lines.append(('berkeley', model, 'acc =', np.mean(acc[5:]).round(3), 'auc =', np.mean(auc[5:]).round(3)))

s = re.compile('test acc=(.*) auc=(.*)')

for model in ['dkt2', 'ours', 'ours400', 'dkt-big', 'dkt50-big', 'ours-big', 'dkt-no-dec', 'dkt-no-dec-big', 'ktm-no-enc', 'ktm-no-enc-big', 'dkt-no-dec50']:
    name = 'assistments' if 'big' in model else 'fraction'
    with open('{}.txt'.format(model)) as f:
        corpus = f.read()

    accs = []
    aucs = []
    for acc, auc in s.findall(corpus):
        accs.append(float(acc))
        aucs.append(float(auc))

    lines.append((name, model, 'acc =', np.mean(accs[:5]).round(3), 'auc =', np.mean(aucs[:5]).round(3)))
    if len(accs) > 5:
        lines.append(('berkeley', model, 'acc =', np.mean(accs[5:]).round(3), 'auc =', np.mean(aucs[5:]).round(3)))

for line in sorted(lines):
    print(line)
