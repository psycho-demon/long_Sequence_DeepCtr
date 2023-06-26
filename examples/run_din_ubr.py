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
    item_map = dict(zip(item_key, range(1, item_len + 1)))

    df['iid'] = df['iid'].map(lambda x: item_map[x])

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(1, user_len + 1)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(1, cate_len + 1)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(1, btag_len + 1)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])

    print("remap completed")
    print(item_len, user_len, cate_len, btag_len)

    feature_columns += [SparseFeat('user', user_len + 1, embedding_dim=16),
                        SparseFeat('item', item_len + 1, embedding_dim=16),
                        SparseFeat('item_cate', cate_len + 1, embedding_dim=16)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', item_len + 1, embedding_dim=16), MAX_LEN_ITEM,
                                         length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_item_cate', cate_len + 1, embedding_dim=16), MAX_LEN_ITEM,
                                         length_name="seq_length")]

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
    train_uid_array = []
    train_iid_array = []
    train_icate_array = []
    train_label_array = []
    train_hist_iid_array = []
    train_hist_icate_array = []
    train_behavior_length = []

    valid_uid_array = []
    valid_iid_array = []
    valid_icate_array = []
    valid_label_array = []
    valid_hist_iid_array = []
    valid_hist_icate_array = []
    valid_behavior_length = []

    test_uid_array = []
    test_iid_array = []
    test_icate_array = []
    test_label_array = []
    test_hist_iid_array = []
    test_hist_icate_array = []
    test_behavior_length = []

    print("the number of user:" + str(len(user_df)))

    # # get each user's last touch point time
    # user_last_touch_time = []
    # for uid, hist in user_df:
    #     user_last_touch_time.append(hist['time'].tolist()[-1])
    #
    # # 按时间划分训练集测试集 7：3
    # user_last_touch_time_sorted = sorted(user_last_touch_time)
    # split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    for uid, hist in user_df:
        if len(hist) < 4:
            continue

        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()


        # ----------------------------------- train data
        train_target_item = item_hist[-3]
        train_target_item_cate = cate_hist[-3]
        train_neg_item, train_neg_cate = neg_sample(train_target_item, item_cnt, item_df)
        # the item history part of the sample
        train_item_part = []
        for i in range(len(item_hist) - 3):
            train_item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        if len(train_item_part) <= MAX_LEN_ITEM:
            train_item_part_pad = [[0] * 4] * (MAX_LEN_ITEM - len(train_item_part)) + train_item_part
        else:
            train_item_part_pad = train_item_part[len(train_item_part) - MAX_LEN_ITEM:len(train_item_part)]
        train_cat_list = []
        train_item_list = []
        for i in range(len(train_item_part_pad)):
            train_item_list.append(train_item_part_pad[i][1])
            train_cat_list.append(train_item_part_pad[i][2])
        train_uid_array.append(uid)
        train_iid_array.append(train_target_item)
        train_icate_array.append(train_target_item_cate)
        train_label_array.append(1)
        train_hist_iid_array.append(train_item_list)
        train_hist_icate_array.append(train_cat_list)
        train_behavior_length.append(min(len(train_item_part), MAX_LEN_ITEM))

        train_uid_array.append(uid)
        train_iid_array.append(train_neg_item)
        train_icate_array.append(train_neg_cate)
        train_label_array.append(0)
        train_hist_iid_array.append(train_item_list)
        train_hist_icate_array.append(train_cat_list)
        train_behavior_length.append(min(len(train_item_part), MAX_LEN_ITEM))


        # ------------------------------------ valid data
        valid_target_item = item_hist[-2]
        valid_target_item_cate = cate_hist[-2]
        valid_neg_item, valid_neg_cate = neg_sample(valid_target_item, item_cnt, item_df)
        # the item history part of the sample
        valid_item_part = []
        for i in range(len(item_hist) - 2):
            valid_item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        if len(valid_item_part) <= MAX_LEN_ITEM:
            valid_item_part_pad = [[0] * 4] * (MAX_LEN_ITEM - len(valid_item_part)) + valid_item_part
        else:
            valid_item_part_pad = valid_item_part[len(valid_item_part) - MAX_LEN_ITEM:len(valid_item_part)]
        valid_cat_list = []
        valid_item_list = []
        for i in range(len(valid_item_part_pad)):
            valid_item_list.append(valid_item_part_pad[i][1])
            valid_cat_list.append(valid_item_part_pad[i][2])
        valid_uid_array.append(uid)
        valid_iid_array.append(valid_target_item)
        valid_icate_array.append(valid_target_item_cate)
        valid_label_array.append(1)
        valid_hist_iid_array.append(valid_item_list)
        valid_hist_icate_array.append(valid_cat_list)
        valid_behavior_length.append(min(len(valid_item_part), MAX_LEN_ITEM))

        valid_uid_array.append(uid)
        valid_iid_array.append(valid_neg_item)
        valid_icate_array.append(valid_neg_cate)
        valid_label_array.append(0)
        valid_hist_iid_array.append(valid_item_list)
        valid_hist_icate_array.append(valid_cat_list)
        valid_behavior_length.append(min(len(valid_item_part), MAX_LEN_ITEM))

        
        # ------------------------------------- test data
        test_target_item = item_hist[-1]
        test_target_item_cate = cate_hist[-1]
        test_neg_item, test_neg_cate = neg_sample(test_target_item, item_cnt, item_df)
        # the item history part of the sample
        test_item_part = []
        for i in range(len(item_hist) - 1):
            test_item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        if len(test_item_part) <= MAX_LEN_ITEM:
            test_item_part_pad = [[0] * 4] * (MAX_LEN_ITEM - len(test_item_part)) + test_item_part
        else:
            test_item_part_pad = test_item_part[len(test_item_part) - MAX_LEN_ITEM:len(test_item_part)]
        test_cat_list = []
        test_item_list = []
        for i in range(len(test_item_part_pad)):
            test_item_list.append(test_item_part_pad[i][1])
            test_cat_list.append(test_item_part_pad[i][2])
        test_uid_array.append(uid)
        test_iid_array.append(test_target_item)
        test_icate_array.append(test_target_item_cate)
        test_label_array.append(1)
        test_hist_iid_array.append(test_item_list)
        test_hist_icate_array.append(test_cat_list)
        test_behavior_length.append(min(len(test_item_part), MAX_LEN_ITEM))

        test_uid_array.append(uid)
        test_iid_array.append(test_neg_item)
        test_icate_array.append(test_neg_cate)
        test_label_array.append(0)
        test_hist_iid_array.append(test_item_list)
        test_hist_icate_array.append(test_cat_list)
        test_behavior_length.append(min(len(test_item_part), MAX_LEN_ITEM))


    train_uid_array = np.array(train_uid_array)
    train_iid_array = np.array(train_iid_array)
    train_icate_array = np.array(train_icate_array)
    train_label_array = np.array(train_label_array)
    train_hist_iid_array = np.array(train_hist_iid_array)
    train_hist_icate_array = np.array(train_hist_icate_array)
    train_behavior_length = np.array(train_behavior_length)
    train_feature_dict = {'user': train_uid_array, 'item': train_iid_array, 'item_cate': train_icate_array,
                    'hist_item': train_hist_iid_array, 'hist_item_cate': train_hist_icate_array,
                    "seq_length": train_behavior_length}
    train_x = {name: train_feature_dict[name] for name in get_feature_names(feature_columns)}
    train_y = train_label_array

    valid_uid_array = np.array(valid_uid_array)
    valid_iid_array = np.array(valid_iid_array)
    valid_icate_array = np.array(valid_icate_array)
    valid_label_array = np.array(valid_label_array)
    valid_hist_iid_array = np.array(valid_hist_iid_array)
    valid_hist_icate_array = np.array(valid_hist_icate_array)
    valid_behavior_length = np.array(valid_behavior_length)
    valid_feature_dict = {'user': valid_uid_array, 'item': valid_iid_array, 'item_cate': valid_icate_array,
                          'hist_item': valid_hist_iid_array, 'hist_item_cate': valid_hist_icate_array,
                          "seq_length": valid_behavior_length}
    valid_x = {name: valid_feature_dict[name] for name in get_feature_names(feature_columns)}
    valid_y = valid_label_array

    test_uid_array = np.array(test_uid_array)
    test_iid_array = np.array(test_iid_array)
    test_icate_array = np.array(test_icate_array)
    test_label_array = np.array(test_label_array)
    test_hist_iid_array = np.array(test_hist_iid_array)
    test_hist_icate_array = np.array(test_hist_icate_array)
    test_behavior_length = np.array(test_behavior_length)
    test_feature_dict = {'user': test_uid_array, 'item': test_iid_array, 'item_cate': test_icate_array,
                          'hist_item': test_hist_iid_array, 'hist_item_cate': test_hist_icate_array,
                          "seq_length": test_behavior_length}
    test_x = {name: test_feature_dict[name] for name in get_feature_names(feature_columns)}
    test_y = test_label_array

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def neg_sample(pos_item, item_cnt, item_df):
    target_item = pos_item
    target_item_cate = 0
    while target_item == pos_item:
        target_item = random.randint(1, item_cnt + 1)
        target_item_cate = item_df.get_group(target_item)['cid'].tolist()[0]
    return target_item, target_item_cate


if __name__ == "__main__":
    random.seed(19)
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:2'
    MAX_LEN_ITEM = 16

    feature_columns = []
    behavior_feature_list = ["item", "item_cate"]
    df = to_df(RAW_DATA_FILE)
    df, item_cnt, feature_size = remap(df, feature_columns, MAX_LEN_ITEM)
    user_df, item_df = gen_user_item_group(df)
    train_x, train_y, valid_x, valid_y, test_x, test_y = gen_dataset(user_df, item_df, item_cnt, feature_size, feature_columns, MAX_LEN_ITEM)

    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['auc', 'logloss'])
    history = model.fit(train_x, train_y, batch_size=256, epochs=1, verbose=2, validation_data=(valid_x, valid_y))

    print("test_data: " + str(model.evaluate(test_x, test_y)))




