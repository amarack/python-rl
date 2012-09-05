
import numpy, csv, matplotlib.pyplot as plt
import sys

data = []
with open(sys.argv[1], "r") as f:
    for row in csv.reader(f):
        data += map(float, row)

data = numpy.array(data)

smoothed = numpy.zeros(data.shape)
for i in range(len(smoothed)):
    smoothed[i] = data[i:i+5000].mean()

plt.plot(smoothed)
plt.title("MSPBE")
plt.xlabel("Timesteps")
plt.ylabel("Estimated MSPBE")
plt.show()
