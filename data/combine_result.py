
import pandas as pd
import glob

dics = dict()
for res in glob.glob('result_*.csv'):
    dics[res] = pd.read_csv(res)
print('We have %d result' %len(dics.keys()))

com = open('result_comb.csv','w')
com.write('id,target\n')
results = list(dics.values())
cnt = 0
for i, j, k in zip(results[0].iterrows(), results[1].iterrows(), results[2].iterrows()):
    idx, v1 = i
    _, v2 = j
    _, v3 = k
    L = [v1['target'], v2['target'], v3['target']]
    
    com.write('%s,%s\n' %(v1['id'], max(set(L), key = L.count)))
    cnt += 1
print('%d rows saved' %cnt)
com.close()
