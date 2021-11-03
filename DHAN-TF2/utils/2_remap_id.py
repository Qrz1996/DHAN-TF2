import random
import pickle
import numpy as np

random.seed(1234)

with open(r'D:\datasets\Amazon_electronic\DHAN/reviews_electronics.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
with open(r'D:\datasets\Amazon_electronic\DHAN/meta_electronics.pkl', 'rb') as f:
    meta_df = pickle.load(f)
    meta_df = meta_df[['asin', 'categories']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

# 这里和DIN的处理有些不同，只保留有超过300个物品的种类
valid_set = []
for categories, hist in meta_df.groupby('categories'):
    pos_list = hist['asin'].tolist()
    if (len(pos_list) > 299):  # 1110
        print(categories, len(pos_list))
        valid_set.append(categories)

meta_df = meta_df[meta_df["categories"].isin(valid_set)]        # 删掉物品数量少于300的种类
asin_set = meta_df['asin'].values.tolist()                      # 所有物品的id，不重复
reviews_df = reviews_df[reviews_df["asin"].isin(asin_set)]      # 删掉记录中不再在meta中的数据


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))             # todo 这个map是从0开始的啊
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


asin_map, asin_key = build_map(meta_df, 'asin')         # 给元数据物品id重新编号
cate_map, cate_key = build_map(meta_df, 'categories')   # 给元数据物品类别重新编号
revi_map, revi_key = build_map(reviews_df, 'reviewerID')

user_count, item_count, cate_count, example_count = \
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]    # len(asin_map) = meta_df.shape[0]
cate_list = np.array(cate_list, dtype=np.int32)     # index是物品编号，值是物品类别编号

with open(r'D:\datasets\Amazon_electronic\DHAN/remap_electronics.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
    pickle.dump((user_count, item_count, cate_count, example_count),
                f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
