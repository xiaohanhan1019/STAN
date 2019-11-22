import pickle

train_data = pickle.load(open('datasets/diginetica/train_session_3.txt', 'rb'))
test_data = pickle.load(open('datasets/diginetica/test_session_3.txt', 'rb'))

sid = train_data[0] + test_data[0]
session = train_data[1] + test_data[1]
predict = train_data[3] + test_data[3]

clicks = 0

iid = -1
for i, s in enumerate(session):
    if iid != sid[i]:
        clicks += len(s) + 1
        iid = sid[i]

item = set()
for s in train_data[1]:
    for i in s:
        item.add(i)
for s in train_data[3]:
    item.add(s)
train_items = len(item)

item.clear()
for s in test_data[1]:
    for i in s:
        item.add(i)
for s in test_data[3]:
    item.add(s)
test_items = len(item)

item.clear()
for s in session:
    for i in s:
        item.add(i)
for s in predict:
    item.add(s)
all_items = len(item)

print("# of training sessions: %d" % len(train_data[0]))
print("# of testing sessions: %d" % len(test_data[0]))
print("# of clicks: %d" % clicks)
print("# of items in training sessions: %d" % train_items)
print("# of items in testing sessions: %d" % test_items)
print("# of items in all sessions: %d" % all_items)