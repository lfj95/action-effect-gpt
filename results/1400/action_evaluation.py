# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 00:15:25 2022

@author: 19002
"""
import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from copy import deepcopy
import sacrebleu
import bert_score
import sys
sys.path.append('D:/lfj/dataset/PIGLET/piglet-main/')



split='test'
ref_keys = ['anno_0','anno_1','anno_2','anno_3','anno_4','anno_5','anno_6','anno_7','anno_8','anno_9']
all_qs = []
with open('D:/lfj/dataset/Paper/few_shot/few-shot-learning-main/data/action/1400_test.json', 'r') as f:
    for i, l in enumerate(f):
        all_qs.append(json.loads(l))
        all_qs[-1]['id']= i

refs = [tuple([x[k] for k in ref_keys]) for x in all_qs]
refs2 = [[x[k] for x in all_qs] for k in ref_keys]    



models = {
        'baseline': 'D:/lfj/dataset/PIGLET/piglet-main/data/result_1400/action_baseline.csv',
        'knowrob': 'D:/lfj/dataset/PIGLET/piglet-main/data/result_1400/action_knowrob.csv',
       'comet': 'D:/lfj/dataset/PIGLET/piglet-main/data/result_1400/action_comet.csv'
   }
all_preds = {}


for model, fn in models.items():
    
    with open(fn, 'r', encoding="utf-8") as f:
        all_preds[model] = pd.read_csv(f)['post_pred'].tolist()
    
        # if any([not isinstance(x, str) for x in all_preds[model]]):
        #     nn = len([x for x in all_preds[model] if not isinstance(x, str)])
        #     print(f"NANs with {model}: {nn}")
        #     all_preds[model] = [x if isinstance(x, str) else dset[i]['annot']['postcondition_language'] for i, x in enumerate(all_preds[model])]



# with open('D:/lfj/dataset/PIGLET/piglet-main/data/result/conceptnet.csv', 'r', encoding="utf-8") as f:
#     all_preds['conceptnet'] = pd.read_csv(f)['postcondition_language'].tolist()

df = []
for model, res in all_preds.items():
        item = {
            'model': model,
        }
        bleu_score = sacrebleu.corpus_bleu(sys_stream=res, ref_streams=refs2, lowercase=True)
        item['BLEU'] = bleu_score.score
        P, R, F1 = bert_score.score(res, refs, verbose=False, model_type='bert-large-uncased')
        item['BERTScore'] = float(F1.mean())
        item['nex'] = len(res)
        df.append(item)
df=pd.DataFrame(df).set_index('model', drop=True)
        
# df_val = get_eval_for_a_split('val')
# df_test = get_eval_for_a_split('test', is_zs=None) 
    
 
out_txt = 'Model & \multicolumn{2}{c}{BLEU} & \multicolumn{2}{c}{BERTScore}'
out_txt += '\\\\ \n'
out_txt += '     & Val & Test               & Val & Test                    & Test'
# for model in ['t5small', 'baseline', 'ours', 'human']:
for model in ['baseline', 'comet', 'knowrob']:

    out_txt += '\\\\ \n'

    # res = [df_val.loc[model, 'BLEU'], df_test.loc[model, 'BLEU'],
    #        100 * df_val.loc[model, 'BERTScore'], 100 * df_test.loc[model, 'BERTScore'],
    #        df_val.loc[model, 'Human'], df_test.loc[model, 'Human'],
    #        ]
    res = [df.loc[model, 'BLEU'],
           100 * df.loc[model, 'BERTScore'], 
           # df.loc[model, 'Human'], 
           ]

    res_txt = [model] + ['{:.1f}'.format(x) for x in res]
    out_txt += '&'.join(['{:>10s}'.format(a) for a in res_txt])
    print(res_txt)