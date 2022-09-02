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
from data.zeroshot_lm_setup.encoder import get_encoder, Encoder


def get_statechange_dataset(seed=123456, zs_in_train=False, daxify=False):
    """
    Retrieve datasets from the ZSLM test corpora file
    :param seed: random seed, sets it globally but that's probably ok for now
    :return:
    """
    all_qs = []
    with open(os.path.join( 'D:/lfj/dataset/PIGLET/piglet-main/', 'data', 'annotations.jsonl'), 'r') as f:
        for i, l in enumerate(f):
            all_qs.append(json.loads(l))
            all_qs[-1]['id'] = i

    sets = {k: [v for v in all_qs if v['split'] == k] for k in ['train', 'val', 'test']}

    if not zs_in_train:
        sets['test'] += [x for x in sets['train'] if x['is_zs']]
        sets['train'] = [x for x in sets['train'] if not x['is_zs']]

    np.random.seed(seed)
    for k in ['train', ]:
        inds_perm = np.random.permutation(len(sets[k])).tolist()
        sets[k] = [sets[k][i] for i in inds_perm]

    # tf.compat.v1.logging.info(" ".join(['{}: {}'.format(k, len(v)) for k, v in sets.items()]))

    if daxify:
        from data.concept_utils import concept_df
        from data.concept_utils import get_concepts_in_text

        zs_objs = concept_df[concept_df['is_zeroshot']]['thor_name'].unique().tolist()
        ids = np.random.permutation(len(zs_objs))
        obj_to_name = {obj: f'blicket{id}' for obj, id in zip(zs_objs, ids)}

        for split in ['train', 'val', 'test']:
            for item in sets[split]:
                for sent in ['precondition', 'action', 'postcondition']:
                    k = f'{sent}_language'
                    sent_k = item['annot'][k]
                    print(item['annot'][k])

                    all_concepts = get_concepts_in_text(sent_k)
                    replace_dict = {}
                    for v in all_concepts:
                        concept_item = concept_df.iloc[v['id']]
                        if concept_item['is_zeroshot']:
                            replace_dict[sent_k[v['start_idx']:v['end_idx']]] = obj_to_name[concept_item['thor_name']]
                    print(replace_dict)
                    for k, v in replace_dict.items():
                        sent_k = sent_k.replace(k, v)
                    item['annot'][k] = sent_k
                    print(item['annot'][k])
                    print('---')
    return sets



all_dsets = get_statechange_dataset()

split='test'
ref_keys = ['annot', 'extra_annot0']
hyp_model = 'extra_annot1'
all_qs = []
with open('D:/lfj/dataset/PIGLET/piglet-main/data/annotations.jsonl', 'r') as f:
    for i, l in enumerate(f):
        all_qs.append(json.loads(l))
        all_qs[-1]['id']= i

dset = all_dsets[split]

to_include = [True for item in dset]
models = {
        'baseline': 'D:/lfj/dataset/PIGLET/piglet-main/data/result_piglet/piglet_baseline.csv',
        'knowrob': 'D:/lfj/dataset/PIGLET/piglet-main/data/result_piglet/piglet_knowrob.csv',
       'conceptnet': 'D:/lfj/dataset/PIGLET/piglet-main/data/result_piglet/piglet_concept.csv'
   }
all_preds = {}
all_preds['human'] = [x[hyp_model]['postcondition_language'] for x in dset]

for model, fn in models.items():
    
    with open(fn, 'r', encoding="utf-8") as f:
        all_preds[model] = pd.read_csv(f)['post_pred'].tolist()
    
        # if any([not isinstance(x, str) for x in all_preds[model]]):
        #     nn = len([x for x in all_preds[model] if not isinstance(x, str)])
        #     print(f"NANs with {model}: {nn}")
        #     all_preds[model] = [x if isinstance(x, str) else dset[i]['annot']['postcondition_language'] for i, x in enumerate(all_preds[model])]



dset_filtered = [x for x, ti in zip(dset, to_include) if ti]
refs = [tuple([x[k]['postcondition_language'] for k in ref_keys]) for x in dset_filtered]
refs2 = [[x[k]['postcondition_language'] for x in dset_filtered] for k in ref_keys]


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
for model in ['baseline', 'conceptnet', 'knowrob', 'human']:

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