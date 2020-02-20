import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

testSize = 0.4 #训练集占0.6  测试集占0.4


fileName = "API_classify_data(Programweb).csv"
data = pd.read_csv(fileName,encoding='utf-8')
# ID      = data[0]
APIName = data['APIName']
descr   = data['descr']
tags    = data['tags2']



# APIName : pass
descr   = descr  #[i.lower() for i in descr]
tags    = [i.lower().split("###")[:-1] for i in tags]

# vocabulary
vocab = None

# make category
catgy = {}
count = 0  # is 115
for i in tags:
    for j in i:
        if j not in catgy.keys():
            count += 1
            catgy[j] = count


#make data
data = []
for i in range(0,len(descr)):
    temp = {'text':descr[i],'Id':str(i+1),'catgy':[catgy[j] for j in tags[i]]}
    data.append(temp)

# split data
train , test = train_test_split(data, test_size = testSize, random_state = 1)

with open('API_classify_data(Programweb).p','wb') as f:
    pickle.dump([train,test,vocab,catgy],f)
