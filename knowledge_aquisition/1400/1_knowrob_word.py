import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import json
sys.path.append(os.getcwd())
import itertools

import tqdm
import os
import openai
import time
import spacy
nlp = spacy.load('en_core_web_sm')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()




def get_content_words(text, nlp):
    """
    Return all the adjectives, nouns and verbs in the text.
    """
    doc = nlp(text)
    content_words = [wnl.lemmatize(t.text, pos=t.pos_.lower()[0]) for t in doc if t.pos_ in {"VERB", "NOUN", "ADJ"}]
    return list(set(map(str.lower, content_words)))


def split_multi_keyword(keywords, sep=','):
    answers = keywords.strip().split(sep)
    split_answers = []
    for answer in answers:
        answer = answer.strip()
        if len(answer):
            split_answers.append(answer)
    return split_answers


if __name__ == "__main__":
    
    keywords_all = []
    for split in ['train', 'val', 'test']:      
        num_lines = sum(1 for _ in open(f'D:/lfj/dataset/Paper/self_talk-master/data/PIGLET/{split}_sents.json'))
        with open(f'D:/lfj/dataset/Paper/self_talk-master/data/PIGLET/{split}_sents.json', 'r') as f_in:
             with open(f'D:/lfj/dataset/Paper/self_talk-master/data/PIGLET/{split}_sents_word.json', "w") as f_out:
                for line in tqdm.tqdm(f_in, total=num_lines):
                    fields = json.loads(line.strip())
                    precondition = fields["precondition"]  #The robot is holding a pair of boots.
                    action = fields["action"]   #The robot puts the boots on the floor.
    
                    
                    # Texts: any pair of content word from the context + choice
                    precondition_context_words =  get_content_words(precondition, nlp)
                    action_context_words =  get_content_words(action, nlp) #get_content_words(choice, nlp): ['holds', 'robot', 'laptop']
                    
                    if '\'s' in precondition_context_words:
                        precondition_context_words.remove('\'s')
                    if '\'s' in action_context_words:
                        action_context_words.remove('\'s')
                    
                    
                    context_words = list(set(precondition_context_words + action_context_words))
                    
                    # for i in range(len(precondition_context_words)):
                    #     if len(precondition_context_words[i])==1:
                    #         print(precondition)
                    #         print(precondition_context_words[i])
    
                    # for i in range(len(action_context_words)):
                    #     if len(action_context_words[i])==1:
                    #         print(action)
                    #         print(action_context_words[i])
                    keywords_all = list(set(keywords_all + context_words))
                    fields['action_precondition'] = context_words
                    fields['action_context_words'] = action_context_words
                    fields['precondiction_context_words'] = precondition_context_words

                    f_out.write(json.dumps(fields) + '\n')
                    f_out.flush()

                    
                    
    
    word_list = keywords_all

    
    knowrob_data = pd.read_csv(f'data_knowrob.csv')
    # verb_annotate = ['open','bake','boil','chop','close','cook','crack','cut','mix','stir']
    # noun_annotate = ['bag', 'box', 'chair', 'cup', 'door', 'drawer', 'egg', 'meat', 'soup']
    verb_annotate = knowrob_data['verb'].tolist()
    verb_common_annotate = knowrob_data['verb_ontology'].tolist()
    noun_annotate = knowrob_data['noun'].tolist()[0:-1]
    noun_common_annotate = knowrob_data['noun_ontology'].tolist()[0:-1]
    word_annotate = verb_annotate + noun_annotate
    word_common_annotate = verb_common_annotate + noun_common_annotate
    
    

    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    
    word_unannotate= [i for i in word_list if i not in word_annotate]
    
    prompt_prefix = ""
    # prompt_prefix = "I am a highly intelligent robot with common sense. If you give me a verb, I will give you the common sense explanation of that verb.\n\n"
    # q_prefix = "Q: What's the common sense knowledge for verb "    
    q_prefix = "Q: What's the common sense knowledge for "
    q_subfix = "?\n"
    a_prefix = "A: "
    a_subfix = "\n\n"
    
    for (i,j) in zip(word_annotate,word_common_annotate):
        prompt_prefix  +=  q_prefix + i + q_subfix + a_prefix + j +a_subfix 

        
    all_raw_answers = []
    all_unannotate_answers=[]
    prompt_list=[]
    
    print(len(word_unannotate))
    
    openai.api_key = "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"
    for i in range(0,len(word_unannotate),1):
        prompt = prompt_prefix + q_prefix + word_unannotate[i] + q_subfix
        prompt_list.append(prompt)
        resp = openai.Completion.create(
          engine="text-davinci-002",
          prompt= prompt,
          temperature=0,
          max_tokens=25,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=["Q: "]
        )
        for answer_id, answer in enumerate(resp['choices']):
            all_raw_answers.append(answer)
        print(i)
        time.sleep(1)   
    
    
    
    for resp in all_raw_answers:
        all_unannotate_answers.append(resp['text'].replace('\n', ''))
    
    all_unannotate_answers = [c.replace('A: ','') for c in all_unannotate_answers]
    # all_unannotate_answers = [c.replace('A:','') for c in all_unannotate_answers]
    all_word = word_annotate + word_unannotate
    all_annotation = word_common_annotate + all_unannotate_answers
    # all_verb_annotation = [c.replace('..','.') for c in all_verb_annotation]
     
    all_word_unique = []
    all_annotation_unique = []
    for i in range(len(all_word)):
        if all_word[i] not in all_word_unique:
            all_word_unique.append(all_word[i])
            all_annotation_unique.append(all_annotation[i])
    c={"word" : all_word_unique, "annotation" : all_annotation_unique}
    data_annotation = pd.core.frame.DataFrame(c)
    
    data_annotation.to_csv('D:/lfj/dataset/Paper/self_talk-master/data/PIGLET/knowrob_gpt3_annotation.csv',index=False)
    
    knowrob_data = pd.read_csv(f'D:/lfj/dataset/Paper/self_talk-master/data/PIGLET/knowrob_gpt3_annotation.csv',index_col=0)


    for split in ['train', 'val', 'test']:      
        num_lines = sum(1 for _ in open(f'D:/lfj/dataset/Paper/self_talk-master/data/PIGLET/{split}_sents_word.json'))
        with open(f'D:/lfj/dataset/Paper/self_talk-master/data/PIGLET/{split}_sents_word.json', 'r') as f_in:
             with open(f'D:/lfj/dataset/Paper/self_talk-master/data/PIGLET/{split}_sents_knowrob.json', "w") as f_out:
                for line in tqdm.tqdm(f_in, total=num_lines):
                    fields = json.loads(line.strip())
                    precondition_context_words = fields['action_context_words']
                    a_anno = ''
                    for v in precondition_context_words:
                        if pd.isnull(v)==False:
                            anno = v  + ' means '+  knowrob_data.at[ v,'annotation'] +' '
                            anno = anno.capitalize()
                            a_anno += anno

                        else:
                            a_anno = ''
                
                    fields['knowrob'] = a_anno

                    f_out.write(json.dumps(fields) + '\n')
                    f_out.flush()
    