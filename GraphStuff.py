import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#lrdq = [0.05,309.0,  0.01,131.0,   0.05,2648.0,  0.1,5000,  0.5,642]
#gatedq = {"Leaky Relu":259.0, "Sigmoid":1423.0, "Tanh":911.0, "Elu":5000.0}
gscores = [92.0,88.0,99.0,437.0,932.0,920.0]
glens = [250,500,1000,2000,4000,6000]

df = pd.read_csv("results_temp.csv")

lrdq = df.iloc[8:13]
print(lrdq["l_rate"])


gatedq = df.iloc[13:17]
print(gatedq["activation"])

gscores = df.iloc[17:23]
print(gscores["games_played"])

dqdf = pd.read_csv("doubleQResults.csv")
m1 = dqdf.iloc[5]
m2 = dqdf.iloc[8]

s1 = list(map(int,list(str.split(m1["steps"][1:-1],","))))
s2 = list(map(int,list(str.split(m2["steps"][1:-1],","))))
s3 = list(map(int,list(str.split((df.iloc[16])["steps"][1:-1],","))))
s4 = list(map(int,list(str.split((df.iloc[5])["steps"][1:-1],","))))

ds = 10
bt = np.zeros((int(len(s1)/ds)))
for j in range(len(bt)):
  bt[j] = np.mean(s1[int(j*ds):int((j+1)*ds)])

plt.plot(bt)

bt = np.zeros((int(len(s2)/ds)))
for j in range(len(bt)):
  bt[j] = np.mean(s2[int(j*ds):int((j+1)*ds)])

plt.plot(bt)

bt = np.zeros((int(len(s3)/ds)))
for j in range(len(bt)):
  bt[j] = np.mean(s3[int(j*ds):int((j+1)*ds)])

plt.plot(bt)

bt = np.zeros((int(len(s4)/ds)))
for j in range(len(bt)):
  bt[j] = np.mean(s4[int(j*ds):int((j+1)*ds)])

plt.plot(bt)
#plt.plot(s2)
#plt.plot(s3)
#plt.plot(s4)
#plt.plot(gscores["games_played"],gscores["test_util"])
plt.legend(["DDQN Large", "DDQN Small","DQN LRelu", "DQN Elu"])
plt.xlabel("Number of Games Played")
plt.ylabel("Total Utility in Test (n_steps)")
plt.title("Train Time on DQN")
plt.show()