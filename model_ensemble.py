import pandas as pd
import numpy as np
pred = []
for index in range(5):
    tmp = pd.read_csv('results/result_of_test' + str(index + 1) + '.csv', delimiter = ',', header = None)
    pred.append(tmp)
all_data = pd.concat(pred, axis = 1)
#all_data.to_csv("test.csv", index = False, header = None)
all_data = all_data.values
ans = []
for i in range(12500):
    x = 0
    y = 0
    tmp = 0
    for j in range(5):
        if all_data[i][j] > 0.5:
            x += 1
        else :
            y += 1
    if x == 0 :
        tmp = 0
    elif x == 5:
        tmp = 1
    else :
        for j in range(5):
            tmp += all_data[i][j]
        tmp /= 5
    if tmp > 0.2 and tmp < 0.8:
        tmp = 0.5
    #tmp = min(tmp,0.995)
    #tmp = max(tmp,0.005)
    ans.append(tmp)
    #print(tmp)
#print(ans)
np.savetxt('test' + '.csv', ans, delimiter=',')
