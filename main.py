import pickle
import numpy as np
from SKNN import SKNN
from VSKNN import VSKNN
from STAN import STAN
import torch
import math
import time

# load data [id,x,t,y]
# train_data = pickle.load(open('datasets/retailrocket/train_session_3.txt', 'rb'))
train_data = pickle.load(open('datasets/diginetica/train_session_3.txt', 'rb'))
# train_data = pickle.load(open('datasets/yoochoose/train_1_4_session_3.txt', 'rb'))
train_id = train_data[0]
train_session = train_data[1]
train_timestamp = train_data[2]
train_predict = train_data[3]

# 把train_session和train_predict合并
for i, s in enumerate(train_session):
    train_session[i] += [train_predict[i]]

# test_data = pickle.load(open('datasets/retailrocket/test_session_3.txt', 'rb'))
test_data = pickle.load(open('datasets/diginetica/test_session_3.txt', 'rb'))
# test_data = pickle.load(open('datasets/yoochoose/test_1_4_session_3.txt', 'rb'))
test_id = test_data[0]
test_session = test_data[1]
test_timestamp = test_data[2]
test_predict = test_data[3]

# model = SKNN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500)
# model = VSKNN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500)
model = STAN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500,
             factor1=True, l1=2, factor2=True, l2=80 * 24 * 3600, factor3=True, l3=8.5)

testing_size = len(test_session)
# testing_size = 10

R_5 = 0
R_10 = 0
R_20 = 0

MRR_5 = 0
MRR_10 = 0
MRR_20 = 0

NDCG_5 = 0
NDCG_10 = 0
NDCG_20 = 0
for i in range(testing_size):
    if i % 1000 == 0:
        print("%d/%d" % (i, testing_size))
        # print("MRR@20: %f" % (MRR_20 / (i + 1)))

    score = model.predict(session_id=test_id[i], session_items=test_session[i], session_timestamp=test_timestamp[i],
                          k=20)
    # for s in score:
    #     print(s)
    # print(test_predict[i])
    # print("-----------------------------------")
    # print("-----------------------------------")
    items = [x[0] for x in score]
    # if len(items) == 0:
    #     print("!!!")
    if test_predict[i] in items:
        rank = items.index(test_predict[i]) + 1
        # print(rank)
        MRR_20 += 1 / rank
        R_20 += 1
        NDCG_20 += 1 / math.log(rank + 1, 2)

        if rank <= 5:
            MRR_5 += 1 / rank
            R_5 += 1
            NDCG_5 += 1 / math.log(rank + 1, 2)

        if rank <= 10:
            MRR_10 += 1 / rank
            R_10 += 1
            NDCG_10 += 1 / math.log(rank + 1, 2)

MRR_5 = MRR_5 / testing_size
MRR_10 = MRR_10 / testing_size
MRR_20 = MRR_20 / testing_size
R_5 = R_5 / testing_size
R_10 = R_10 / testing_size
R_20 = R_20 / testing_size
NDCG_5 = NDCG_5 / testing_size
NDCG_10 = NDCG_10 / testing_size
NDCG_20 = NDCG_20 / testing_size

print("MRR@5: %f" % MRR_5)
print("MRR@10: %f" % MRR_10)
print("MRR@20: %f" % MRR_20)
print("R@5: %f" % R_5)
print("R@10: %f" % R_10)
print("R@20: %f" % R_20)
print("NDCG@5: %f" % NDCG_5)
print("NDCG@10: %f" % NDCG_10)
print("NDCG@20: %f" % NDCG_20)
print("training size: %d" % len(train_session))
print("testing size: %d" % testing_size)
