
import numpy, csv, matplotlib.pyplot as plt
import sys

data = []
with open(sys.argv[1], "r") as f:
    for row in csv.reader(f):
        data += map(float, row)

data = numpy.array(data)


buf = 5000
smoothed = numpy.zeros((len(data)-buf,))
for i in range(len(smoothed)):
    smoothed[i] = data[i:i+buf].mean()

plt.plot(smoothed)
#plt.yscale('log')
#plt.ylim([0,50])
plt.title("MSPBE")
plt.xlabel("Timesteps")
plt.ylabel("Estimated MSPBE")
plt.savefig("mspbe.png")
#plt.show()
