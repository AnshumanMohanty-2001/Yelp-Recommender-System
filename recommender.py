"""
Method Description
I have worked on my Model-Based Recommender System, leveraging XGBoost to predict user ratings for businesses using the Yelp Reviews Dataset.

To enhance the model, I increased the number of features extracted from user.json, business.json, tip.json, and photo.json. While I considered using features from review_train.json, I ultimately decided against it since the validation set lacked corresponding review-related features. To address the cold-start problem, I assigned average rating values of 3.76 for new users and 3.75 for new businesses, based on overall dataset averages. This adjustment reduced the RMSE by 0.0035 compared to my HW-3 results. Additionally, including the state feature led to a further RMSE reduction of 0.0007. After consolidating these features into a NumPy array, I applied MinMax normalization to scale them effectively. These improvements brought the RMSE down from 0.9852 to 0.9810 in this project.

To optimize the model further, I employed Optuna for hyperparameter tuning with an extensive search space and 2-fold cross-validation. This approach reduced the RMSE to 0.9753 on the validation set. I also experimented with a Gradient Boosting Regressor model using the same preprocessing steps and explored hybrid approaches by combining item-based collaborative filtering with the XGBoost model. Despite these attempts, the XGBoost solution provided the best results for the final submission.

Error Distribution
>=0 and <1: 102682                                                                                                                                                          
>=1 and <2: 32446                           
>=2 and <3: 6090            
>=3 and <4: 824            
>=4: 2 

RMSE
0.9753088256169121

Duration:
1067.643090724945s
"""

import os
import csv
from datetime import datetime
from sys import argv

from json import loads
from pyspark import SparkContext, SparkConf
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

# os.environ['PYSPARK_PYTHON'] = 'C:\\Users\\anshu\\USC\\Fall-24\\553\\Assigments\\venv-553\\Scripts\\python.exe'
# os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:\\Users\\anshu\\USC\\Fall-24\\553\\Assigments\\venv-553\\Scripts\\python.exe'

conf = SparkConf().setAppName('Competition')

ob_sp = SparkContext(conf=conf)

init_time = time.time()

dset_dir, val_file, op_file = argv[1], argv[2], argv[3]

ip_rdd_train, ip_rdd_val = ob_sp.textFile(os.path.join(dset_dir, 'yelp_train.csv')), ob_sp.textFile(val_file)
col_name_train, col_name_val = ip_rdd_train.first(), ip_rdd_val.first()

rdd_proc_train = ip_rdd_train.filter(lambda rec: rec != col_name_train).map(lambda rec: rec.split(','))
rdd_proc_val = ip_rdd_val.filter(lambda rec: rec != col_name_val).map(lambda rec: rec.split(',')).cache()


def get_json_rdd(loc):
    temp_ob = ob_sp.textFile(loc)
    return temp_ob.mapPartitions(lambda part: map(loads, part))


## (uid: (fans, avg_stars, rev_count, average_years, average_compliment, friend_count, elite_count, useful, funny, cool))
rdd_us_js = (get_json_rdd(os.path.join(dset_dir, 'user.json'))
.map(lambda rec: (rec['user_id'], (int(rec['fans']), float(rec['average_stars']), int(rec['review_count']),
(datetime.now() - datetime.strptime(rec['yelping_since'],"%Y-%m-%d")).days / 365,(rec['compliment_hot'] + rec['compliment_more'] + rec['compliment_profile'] + rec['compliment_cute'] +
rec['compliment_list'] + rec['compliment_note'] + rec['compliment_plain'] + rec['compliment_cool'] + rec['compliment_funny'] + rec['compliment_writer'] + rec['compliment_photos']) / 11,
0 if rec['friends'] == "None" else len(rec['friends'].split(',')), 0 if rec['elite'] == "None" else len(rec['elite'].split(',')), rec['useful'], rec['funny'], rec['cool']))).cache().collectAsMap())


## (bid: (rev_count, total_stars, is_open, categories, attributes, state) )
state_rdd = get_json_rdd(os.path.join(dset_dir, 'business.json')).map(lambda rec: rec['state']).distinct()
state_label = state_rdd.zipWithIndex().collectAsMap()


rdd_biz_js = (get_json_rdd(os.path.join(dset_dir, 'business.json'))
.map(lambda rec: (rec['business_id'], (int(rec['review_count']), float(rec['stars']), rec['is_open'], 0 if rec['categories'] is None else len(rec['categories'].split(',')),
0 if rec['attributes'] is None else len(rec['attributes']), state_label[rec['state']]) )).cache().collectAsMap())


## ( (user_id, business_id): (like, total))
rdd_tip_js = (get_json_rdd(os.path.join(dset_dir, 'tip.json'))
.map(lambda rec: ((rec['user_id'], rec['business_id']), (rec['likes'], 1))).aggregateByKey((0, 0), lambda rec1, rec2: (rec1[0] + rec2[0], rec1[1] + rec2[1]),lambda rec1, rec2: (rec1[0] + rec2[0],
rec1[1] + rec2[1])).cache().collectAsMap())

## (business_id: (label_num, photo_num) )
rdd_photo_js = (get_json_rdd(os.path.join(dset_dir, 'photo.json'))
.map(lambda rec: ((rec['business_id']), (rec['label'], 1))).aggregateByKey((set(), 0),lambda rec1, rec2: ( rec1[0].union([rec2[0]]),rec1[1] + rec2[1]),lambda rec1, rec2: (rec1[0].union(rec2[0]),
rec1[1] + rec2[1])).map(lambda kv: (kv[0], (len(kv[1][0]), kv[1][1]))).cache().collectAsMap())

## (uid, bid, (fans, avg_stars, rev_count, average_years, average_compliment, friend_count, elite_count, useful, funny, cool),
# (rev_count, total_stars, is_open, categories, attributes), (like, total), (label_num, photo_num) )->
## (uid, bid, (fans, avg_stars, rev_count, average_years, average_compliment, friend_count, elite_count, useful, funny, cool,
# rev_count, total_stars, is_open, categories, attributes, like, total, label_num, photo_num)

X_train = (rdd_proc_train.map(lambda rec: (
rec[0], rec[1], rdd_us_js.get(rec[0], (0, 3.76, 0, None, 0, 0, 0, None, None, None)),
rdd_biz_js.get(rec[1], (0, 3.75, None, None, None, -1)), rdd_tip_js.get((rec[0], rec[1]), (0, 0)),
rdd_photo_js.get(rec[1], (0, 0)))).map(lambda rec: (rec[2][0], rec[2][1], rec[2][2], rec[2][3],
                                                    rec[2][4], rec[2][5], rec[2][6], rec[2][7], rec[2][8], rec[2][9],

                                                    rec[3][0], rec[3][1], rec[3][2], rec[3][3], rec[3][4], rec[3][5], rec[4][0],
                                                    rec[4][1], rec[5][0], rec[5][1])).collect())

X_test = (rdd_proc_val.map(lambda rec: (
rec[0], rec[1], rdd_us_js.get(rec[0], (0, 3.76, 0, None, 0, 0, 0, None, None, None)),
rdd_biz_js.get(rec[1], (0, 3.75, None, None, None, -1)), rdd_tip_js.get((rec[0], rec[1]), (0, 0)),
rdd_photo_js.get(rec[1], (0, 0)))).map(lambda rec: (rec[2][0], rec[2][1], rec[2][2], rec[2][3],
                                                    rec[2][4], rec[2][5], rec[2][6], rec[2][7], rec[2][8], rec[2][9],

                                                    rec[3][0], rec[3][1], rec[3][2], rec[3][3], rec[3][4], rec[3][5], rec[4][0],
                                                    rec[4][1], rec[5][0], rec[5][1])).collect())

y_test = rdd_proc_val.map(lambda rec: float(rec[2])).collect()

y_train = rdd_proc_train.map(lambda rec: float(rec[2])).collect()

X_train_np = np.array(X_train)
y_train_np = np.array(y_train).reshape(-1, 1)
combined_train_data = np.hstack((X_train_np, y_train_np))

X_test_np = np.array(X_test)
y_test_np = np.array(y_test).reshape(-1, 1)
combined_val_data = np.hstack((X_test_np, y_test_np))


state_idx = 15
X_train_state = combined_train_data[:, state_idx] 
X_train_features = np.delete(combined_train_data[:, :-1], state_idx, axis=1)

X_test_state = combined_val_data[:, state_idx]
X_test_features = np.delete(combined_val_data[:, :-1], state_idx, axis=1)

sc = MinMaxScaler()
X_train_features_scaled = sc.fit_transform(X_train_features)
X_test_features_scaled = sc.transform(X_test_features)

# Recombine scaled features with the state column
X_train_scaled = np.hstack((X_train_features_scaled, X_train_state.reshape(-1, 1)))
X_test_scaled = np.hstack((X_test_features_scaled, X_test_state.reshape(-1, 1)))



val_proc_lst = rdd_proc_val.map(lambda rec: (rec[0], rec[1])).collect()


params = {'learning_rate': 0.031603392636448006, 'n_estimators': 1000, 'max_depth': 8, 'min_child_weight': 17, 'subsample': 0.8985070339768877, 'colsample_bytree': 0.727203100222523, 'gamma': 3.5310638001304353, 'reg_alpha': 3.085063223179588, 'reg_lambda': 5.7734872264759587, 'max_features': 'sqrt'}
model = XGBRegressor(**params)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
result_list = list()

for i in range(len(y_pred)):
    result_list.append((val_proc_lst[i][0], val_proc_lst[i][1], str(y_pred[i])))

    

header = ["user_id", "business_id", "prediction"]
with open(argv[3], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(result_list)


ob_sp.stop()
print('Duration: ', time.time() - init_time)
