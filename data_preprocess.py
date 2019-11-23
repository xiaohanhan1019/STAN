import pickle
import operator
import time

dataset = 'diginetica'

# store data into map {session_id:seq} and {session_id:timestamp}
ds = './datasets/retailrocket/events_train_full.0.txt'
if dataset == 'retailrocket':
    ds = './datasets/retailrocket/events_train_full.0.txt'
elif dataset == 'yoochoose':
    ds = './datasets/yoochoose/yoochoose-clicks.dat'
elif dataset == 'diginetica':
    ds = './datasets/diginetica/train-item-views.csv'

with open(ds, "r") as f:
    if dataset == 'retailrocket' or dataset == 'diginetica':
        lines = f.readlines()[1:]  # skip the first line
    else:
        lines = f.readlines()
    sess_clicks = {}
    sess_date = {}
    for line in lines:
        if dataset == 'retailrocket':
            data = [int(x) for x in line.split()]
            sess_id = data[3]
            item = data[2]
            timestamp = data[0]
        elif dataset == 'yoochoose':
            data = [x for x in line.split(",")]
            sess_id = int(data[0])
            item = int(data[2])
            timestamp = time.mktime(time.strptime(data[1][:19], '%Y-%m-%dT%H:%M:%S'))
        elif dataset == 'diginetica':
            data = [x for x in line.split(";")]
            sess_id = int(data[0])
            timestamp = time.mktime(time.strptime(data[4][:-1], '%Y-%m-%d'))
            timeframe = int(data[3])
            item = (int(data[2]), timestamp, timeframe)

        if sess_id in sess_clicks:
            sess_clicks[sess_id] += [item]
            sess_date[sess_id] = timestamp  # timestamp取最后一个item的timestamp
        else:
            sess_clicks[sess_id] = [item]
            sess_date[sess_id] = timestamp  # timestamp取最后一个item的timestamp

    if dataset == 'diginetica':
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            # sorted_clicks = sorted(sess_clicks[i], key=lambda x: (x[1], x[2]))  # 以日期为第一关键字,timeframe为第二关键字
            sess_clicks[i] = [c[0] for c in sorted_clicks]
print("length of session clicks: %d" % len(sess_clicks))

filter_len = 1
# filter out length 1 or 2 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) <= filter_len:
        del sess_clicks[s]
        del sess_date[s]
print("after filter out length of %d, length of session clicks: %d" % (filter_len, len(sess_clicks)))

# Count number of times each item appears
item_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in item_counts:
            item_counts[iid] += 1
        else:
            item_counts[iid] = 1

sorted_counts = sorted(item_counts.items(), key=operator.itemgetter(1))

# 去掉次数小于5的item后 session内item数量小于等于filter_len的删掉
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: item_counts[i] >= 5, curseq))
    if len(filseq) <= filter_len:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq
print("after item<5 , length of session clicks: %d" % len(sess_clicks))

# split out train set and test set based on date
dates = list(sess_date.items())
max_timestamp = dates[0][1]

for _, date in dates:
    max_timestamp = max(max_timestamp, date)  # latest date

# split_timestamp为latest date当天00:00的时间戳
# max_date = time.strftime("%Y-%m-%d", time.gmtime(max_timestamp))
# split_timestamp = time.mktime(time.strptime(max_date, "%Y-%m-%d"))
# 再往前推6天
# split_timestamp = split_timestamp - 6 * 86400
if dataset == 'retailrocket' or dataset == 'diginetica':
    split_timestamp = max_timestamp - 7 * 86400
else:
    split_timestamp = max_timestamp - 86400

print("split timestamp: %d" % split_timestamp)
train_session = filter(lambda x: x[1] < split_timestamp, dates)
test_session = filter(lambda x: x[1] > split_timestamp, dates)

# Sort sessions by date
train_session = sorted(train_session, key=operator.itemgetter(1))
test_session = sorted(test_session, key=operator.itemgetter(1))

item_dict = {}


# convert training sessions to sequences and renumber the items
def get_train():
    tra_sid = []
    tra_seq = []
    tra_timestamp = []
    item_cnt = 1
    for s, t in train_session:
        seq = sess_clicks[s]
        outseq = []
        if len(seq) < filter_len:  # Doesn't occur
            continue
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                item_dict[i] = item_cnt
                outseq += [item_dict[i]]
                item_cnt += 1
        tra_sid += [s]
        tra_timestamp += [t]
        tra_seq += [outseq]
    print('item_num: %d' % (item_cnt - 1))
    return tra_sid, tra_timestamp, tra_seq


# Convert test sessions to sequences, ignoring items that do not appear in training set
def get_test():
    tes_sid = []
    tes_seq = []
    tes_timestamp = []
    for s, t in test_session:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < filter_len:
            continue
        tes_sid += [s]
        tes_timestamp += [t]
        tes_seq += [outseq]
    return tes_sid, tes_timestamp, tes_seq


train_sid, train_timestamp, train_seq = get_train()
test_sid, test_timestamp, test_seq = get_test()
print("training: %d" % len(train_sid))
print("testing: %d" % len(test_sid))


# train_data = (train_sid, train_seq, train_timestamp)
# pickle.dump(train_data, open('./datasets/retailrocket/train_seq_2.txt', 'wb'))


# split sequence
def split_seq(sid, timestamp, seq):
    x = []
    t = []
    y = []
    s_id = []
    for sid, seq, timestamp in zip(sid, seq, timestamp):
        if filter_len == 2:
            temp = len(seq) - 1
        else:
            temp = len(seq)
        for i in range(1, temp):
            y += [seq[-i]]
            x += [seq[:-i]]
            t += [timestamp]
            s_id += [sid]
    return x, t, y, s_id


def split_seq_train(sid, timestamp, seq):
    x = []
    t = []
    s_id = []
    for sid, seq, timestamp in zip(sid, seq, timestamp):
        if len(seq) > filter_len:
            x += [seq]
            t += [timestamp]
            s_id += [sid]
        if filter_len == 2:
            temp = len(seq) - 1
        else:
            temp = len(seq)
        for i in range(1, temp):
            x += [seq[:-i]]
            t += [timestamp]
            s_id += [sid]
    return x, t, s_id


tr_seq, tr_timestamp, tr_predict, tr_sid = split_seq(train_sid, train_timestamp, train_seq)
te_seq, te_timestamp, te_predict, te_sid = split_seq(test_sid, test_timestamp, test_seq)

print("after split,training: %d" % len(tr_seq))
print("after split,testing: %d" % len(te_seq))

train_data = (tr_sid, tr_seq, tr_timestamp, tr_predict)
test_data = (te_sid, te_seq, te_timestamp, te_predict)

# retailrocket
# pickle.dump(train_data, open('./datasets/retailrocket/train_session_2.txt', 'wb'))  # 749947 item 48989
# pickle.dump(test_data, open('./datasets/retailrocket/test_session_2.txt', 'wb'))  # 28445

# pickle.dump(train_data, open('./datasets/retailrocket/train_session_3.txt', 'wb'))  # 433648 item 36968
# pickle.dump(test_data, open('./datasets/retailrocket/test_session_3.txt', 'wb'))  # 15132

# ------------------------------------------------------------------------------------
# yoochoose session>=2
# pickle.dump(test_data, open('./datasets/yoochoose/test_1_4_session_2.txt', 'wb'))  # 55898
# pickle.dump(test_data, open('./datasets/yoochoose/test_1_64_session_2.txt', 'wb'))  # 55898

# split4, split64 = int(len(tr_seq) / 4), int(len(tr_seq) / 64)
# split4 = split4 + 1

# train_4 = (tr_sid[-split4:], tr_seq[-split4:], tr_timestamp[-split4:], tr_predict[-split4:])  # 5917746
# pickle.dump(train_4, open('./datasets/yoochoose/train_1_4_session_2.txt', 'wb'))

# sequence
# train_data = (train_sid[-split4:], train_seq[-split4:], train_timestamp[-split4:])
# pickle.dump(train_data, open('./datasets/yoochoose/train_1_4_seq_2.txt', 'wb'))

# train_64 = (tr_sid[-split64:], tr_seq[-split64:], tr_timestamp[-split64:], tr_predict[-split64:])  # 369859
# pickle.dump(train_64, open('./datasets/yoochoose/train_1_64_session_2.txt', 'wb'))

# sequence
# train_data = (train_sid[-split64:], train_seq[-split64:], train_timestamp[-split64:])
# pickle.dump(train_data, open('./datasets/yoochoose/train_1_64_seq_2.txt', 'wb'))

# ------------------------------------------------------------------------------------
# yoochoose session>=3
# pickle.dump(test_data, open('./datasets/yoochoose/test_1_4_session_3.txt', 'wb'))  # 40548
# pickle.dump(test_data, open('./datasets/yoochoose/test_1_64_session_3.txt', 'wb'))  # 40548

# split4, split64 = int(len(tr_seq) / 4), int(len(tr_seq) / 64)

# train_4 = (tr_sid[-split4:], tr_seq[-split4:], tr_timestamp[-split4:], tr_predict[-split4:])  # 3924324
# pickle.dump(train_4, open('./datasets/yoochoose/train_1_4_session_3.txt', 'wb'))

# sequence
# train_data = (train_sid[-split4:], train_seq[-split4:], train_timestamp[-split4:])
# pickle.dump(train_data, open('./datasets/yoochoose/train_1_4_seq_3.txt', 'wb'))

# train_64 = (tr_sid[-split64:], tr_seq[-split64:], tr_timestamp[-split64:], tr_predict[-split64:])  # 245270
# pickle.dump(train_64, open('./datasets/yoochoose/train_1_64_session_3.txt', 'wb'))

# sequence
# train_data = (train_sid[-split64:], train_seq[-split64:], train_timestamp[-split64:])
# pickle.dump(train_data, open('./datasets/yoochoose/train_1_64_seq_3.txt', 'wb'))

# ------------------------------------------------------------------------------------
# diginetica session>=2
# pickle.dump(train_data, open('./datasets/diginetica/train_session_2.txt', 'wb'))
# pickle.dump(test_data, open('./datasets/diginetica/test_session_2.txt', 'wb'))

# diginetica session>=3
# pickle.dump(train_data, open('./datasets/diginetica/train_session_3.txt', 'wb'))
# pickle.dump(test_data, open('./datasets/diginetica/test_session_3.txt', 'wb'))

print("finish")

# 例子[1,2,3]
# session>=2 保留[1,2,3],[1,2]
# session>=3 保留[1,2,3]
