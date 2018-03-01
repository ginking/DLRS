import sys
import os
import numpy as np
import pandas as pd
import torch
import aiohttp
import asyncio
import json
import requests
from utils import get_gpu_name, get_number_processors, get_gpu_memory, get_cuda_version
from parameters import *
from load_test import run_load_test

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)
print("Number of CPU processors: ", get_number_processors())
print("GPU: ", get_gpu_name())
print("GPU memory: ", get_gpu_memory())
print("CUDA: ", get_cuda_version())

%matplotlib inline
%load_ext autoreload
%autoreload 2

%%time
%run ./DeepRecommender/data_utils/netflix_data_convert.py $NF_PRIZE_DATASET $NF_DATA

nf_3m_valid = os.path.join(NF_DATA, 'N3M_VALID', 'n3m.valid.txt')
df = pd.read_csv(nf_3m_valid, names=['CustomerID','MovieID','Rating'], sep='\t')
print(df.shape)
df.head()

nf_3m_test = os.path.join(NF_DATA, 'N3M_TEST', 'n3m.test.txt')
df2 = pd.read_csv(nf_3m_test, names=['CustomerID','MovieID','Rating'], sep='\t')
print(df2.shape)
df2.head()

%run ./DeepRecommender/run.py --gpu_ids $GPUS \
    --path_to_train_data $TRAIN \
    --path_to_eval_data $EVAL \
    --hidden_layers $HIDDEN \
    --non_linearity_type $ACTIVATION \
    --batch_size $BATCH_SIZE \
    --logdir $MODEL_OUTPUT_DIR \
    --drop_prob $DROPOUT \
    --optimizer $OPTIMIZER \
    --lr $LR \
    --weight_decay $WD \
    --aug_step $AUG_STEP \
    --num_epochs $EPOCHS
    
%run ./DeepRecommender/infer.py \
    --path_to_train_data $TRAIN \
    --path_to_eval_data $TEST \
    --hidden_layers $HIDDEN \
    --non_linearity_type $ACTIVATION \
    --save_path  $MODEL_PATH \
    --drop_prob $DROPOUT \
    --predictions_path $INFER_OUTPUT

%run ./DeepRecommender/compute_RMSE.py --path_to_predictions=$INFER_OUTPUT

titles = pd.read_csv(MOVIE_TITLES, names=['MovieID','Year','Title'], encoding = "latin")
titles.head()

target = df2[df2['CustomerID'] == 0]
target

df_customer = pd.merge(target, titles, on='MovieID', how='left', suffixes=('_',''))
df_customer.drop(['Title_','Year'], axis=1, inplace=True)
df_customer


df_query = df_customer.drop(['CustomerID','Title'], axis=1).set_index('MovieID')
dict_query = df_query.to_dict()['Rating']
dict_query

end_point = 'http://127.0.0.1:5000/'
end_point_recommend = "http://127.0.0.1:5000/recommend"

!curl $end_point

headers = {'Content-type':'application/json'}
res = requests.post(end_point_recommend, data=json.dumps(dict_query), headers=headers)
print(res.ok)
print(json.dumps(res.json(), indent=2))


NUM = 10
CONCURRENT = 2
VERBOSE = True
payload = {13:5.0, 191:5.0, 209:5.0}
payload_list = [payload]*NUM

%%time
with aiohttp.ClientSession() as session:
    loop = asyncio.get_event_loop()
    calc_routes = loop.run_until_complete(run_load_test(end_point_recommend, payload_list, session, CONCURRENT, VERBOSE))
