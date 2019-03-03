import matplotlib.pyplot as plt
import json
import sys


with open(sys.argv[1]) as f:
    data = json.load(f)

for legend in data['metrics']:
    plt.plot(data['metrics'][legend], label=legend)
plt.legend()
plt.show()
