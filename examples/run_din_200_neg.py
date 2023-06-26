'''
在 run_din.py的基础上
不让每一条数据随机为正或者负
而是根据每条数据生成一个正和一个负
'''
import sys

sys.path.insert(0, '..')
import pickle as pkl
import pandas as pd
import random
import numpy as np
import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN

RAW_DATA_FILE = '../data/taobao_data/UserBehavior.csv'
DATASET_PKL = '../data/taobao_data/dataset.pkl'
Test_File = "../data/taobao_data/taobao_test.txt"
Train_File = "../data/taobao_data/taobao_train.txt"
# Train_handle = open(Train_File, 'w')
# Test_handle = open(Test_File, 'w')
# Feature_handle = open("../data/taobao_data/taobao_feature.pkl", 'wb')



def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    return df


def remap(df, feature_columns, MAX_LEN_ITEM):
    # 特征id化 顺序：item user cate btag
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))

    df['iid'] = df['iid'].map(lambda x: item_map[x])

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(btag_len)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])

    print("remap completed")
    print(item_len, user_len, cate_len, btag_len)

    feature_columns += [SparseFeat('user', user_len, embedding_dim=16),
                        SparseFeat('item', item_len + 1, embedding_dim=16),
                        SparseFeat('item_cate', cate_len + 1, embedding_dim=16)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', item_len + 1, embedding_dim=16), MAX_LEN_ITEM, length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_item_cate', cate_len + 1, embedding_dim=16), MAX_LEN_ITEM, length_name="seq_length")]

    return df, item_len, user_len + item_len + cate_len + btag_len + 1  # +1 is for unknown target btag


def gen_user_item_group(df):
    # 根据uid、time排序， uid分组
    # 根据iid、time排序， iid分组
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')

    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, item_cnt, feature_size, feature_columns, MAX_LEN_ITEM):
    # uid + target_item + target_item_cate + label + item_list + cat_list
    uid_array = []
    iid_array = []
    icate_array = []
    label_array = []
    hist_iid_array = []
    hist_icate_array = []
    behavior_length = []

    test_uid_array = []
    test_iid_array = []
    test_icate_array = []
    test_label_array = []
    test_hist_iid_array = []
    test_hist_icate_array = []
    test_behavior_length = []

    train_sample_list = []
    test_sample_list = []
    print("the number of user:" + str(len(user_df)))

    # get each user's last touch point time
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])

    # 按时间划分训练集测试集 7：3
    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    cnt = 0
    for uid, hist in user_df:
        cnt += 1
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()

        target_item_time = hist['time'].tolist()[-1]
        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        target_item_btag = feature_size  # unknown btag
        test = (target_item_time > split_time)
        label = 1

        # neg sampling
        neg_label = 0
        neg_target_item = target_item
        while neg_target_item == item_hist[-1]:
            neg_target_item = random.randint(0, item_cnt - 1)
            neg_target_item_cate = item_df.get_group(neg_target_item)['cid'].tolist()[0]
            neg_target_item_btag = feature_size

        # the item history part of the sample
        item_part = []
        for i in range(len(item_hist) - 1):
            item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        item_part.append([uid, target_item, target_item_cate, target_item_btag])
        # item_part_len = min(len(item_part), MAX_LEN_ITEM)

        # choose the item side information: which user has clicked the target item
        # padding history with 0
        if len(item_part) <= MAX_LEN_ITEM:
            item_part_pad = [[0] * 4] * (MAX_LEN_ITEM - len(item_part)) + item_part
        else:
            item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]

        # gen sample
        # sample = (label, item_part_pad, item_part_len, user_part_pad, user_part_len)

        if test:
            # test_set.append(sample)
            cat_list = []
            item_list = []
            # btag_list = []
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
                # cat_list.append(item_part_pad[i][0])
            test_sample_list.append(
                str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(
                    map(str, item_list)) + "\t" + ",".join(map(str, cat_list)) + "\n")
            test_uid_array.append(uid)
            test_iid_array.append(target_item)
            test_icate_array.append(target_item_cate)
            test_label_array.append(label)
            test_hist_iid_array.append(item_list)
            test_hist_icate_array.append(cat_list)
            test_behavior_length.append(min(len(item_part), MAX_LEN_ITEM))

            # neg
            test_uid_array.append(uid)
            test_iid_array.append(neg_target_item)
            test_icate_array.append(neg_target_item_cate)
            test_label_array.append(neg_label)
            test_hist_iid_array.append(item_list)
            test_hist_icate_array.append(cat_list)
            test_behavior_length.append(min(len(item_part), MAX_LEN_ITEM))
        else:
            cat_list = []
            item_list = []
            # btag_list = []
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            train_sample_list.append(
                str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(
                    map(str, item_list)) + "\t" + ",".join(map(str, cat_list)) + "\n")
            uid_array.append(uid)
            iid_array.append(target_item)
            icate_array.append(target_item_cate)
            label_array.append(label)
            hist_iid_array.append(item_list)
            hist_icate_array.append(cat_list)
            behavior_length.append(min(len(item_part), MAX_LEN_ITEM))

            # neg
            uid_array.append(uid)
            iid_array.append(neg_target_item)
            icate_array.append(neg_target_item_cate)
            label_array.append(neg_label)
            hist_iid_array.append(item_list)
            hist_icate_array.append(cat_list)
            behavior_length.append(min(len(item_part), MAX_LEN_ITEM))

    uid_array = np.array(uid_array)
    iid_array = np.array(iid_array)
    icate_array = np.array(icate_array)
    label_array = np.array(label_array)
    hist_iid_array = np.array(hist_iid_array)
    hist_icate_array = np.array(hist_icate_array)
    behavior_length = np.array(behavior_length)

    test_uid_array = np.array(test_uid_array)
    test_iid_array = np.array(test_iid_array)
    test_icate_array = np.array(test_icate_array)
    test_label_array = np.array(test_label_array)
    test_hist_iid_array = np.array(test_hist_iid_array)
    test_hist_icate_array = np.array(test_hist_icate_array)
    test_behavior_length = np.array(test_behavior_length)

    feature_dict = {'user': uid_array, 'item': iid_array, 'item_cate': icate_array,
                    'hist_item': hist_iid_array, 'hist_item_cate': hist_icate_array,
                    "seq_length": behavior_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = label_array

    test_feature_dict = {'user': test_uid_array, 'item': test_iid_array, 'item_cate': test_icate_array,
                         'hist_item': test_hist_iid_array, 'hist_item_cate': test_hist_icate_array,
                         "seq_length": test_behavior_length}
    test_x = {name: test_feature_dict[name] for name in get_feature_names(feature_columns)}
    test_y = test_label_array
    print("train and test sample completed")
    return x, y, test_x, test_y



def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat('gender', 2, embedding_dim=8),
                       SparseFeat('item', 3 + 1, embedding_dim=8), SparseFeat('item_gender', 2 + 1, embedding_dim=8),
                       DenseFeat('score', 1)]

    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), 4, length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), 4, length_name="seq_length")]
    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
    behavior_length = np.array([3, 3, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score,
                    "seq_length": behavior_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])

    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    random.seed(19)
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:2'
    MAX_LEN_ITEM = 200

    feature_columns = []
    behavior_feature_list = ["item", "item_cate"]
    df = to_df(RAW_DATA_FILE)
    df, item_cnt, feature_size = remap(df, feature_columns, MAX_LEN_ITEM)
    user_df, item_df = gen_user_item_group(df)
    x, y, test_x, test_y = gen_dataset(user_df, item_df, item_cnt, feature_size, feature_columns, MAX_LEN_ITEM)

    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['auc', 'logloss'])
    history = model.fit(x, y, batch_size=256, epochs=5, verbose=2, validation_data=(test_x, test_y))




