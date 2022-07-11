# %% Imports
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# %% Path definition
input_path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/' \
    '05_spy_project_NEWS_v2/00_data/00_raw/2018_body_labeled.feather'

output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/' \
    '05_spy_project_NEWS_v2/00_data/01_preprocessed'

# %% Global initialization
args = {'selected_label': 'label1',
        'model_name': 'ProsusAI/finbert',
        'seq_len': 256,
        'test_size': 0.2}


# %% Function definitions
# BERT tokenization
def BERT_tokenize_f(strings, **kwargs):
    token_ids = []
    token_types = []
    att_masks = []
    bert_tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name'],
                                                   use_fast = True)
    
    for string in tqdm(strings, desc='tokenizing'):
        bert_out = bert_tokenizer(string,
                                  return_tensors = 'pt',
                                  padding = 'max_length',
                                  truncation = True,
                                  max_length = kwargs['seq_len'])
    
        token_ids.append(bert_out['input_ids'].squeeze(0).type(torch.LongTensor))
        token_types.append(bert_out['token_type_ids'].squeeze(0).type(torch.LongTensor))
        att_masks.append(bert_out['attention_mask'].squeeze(0).type(torch.LongTensor))
     
    return token_ids, token_types, att_masks   

# Conversion to multilabel
def to_multilabel_f(labels):
    multilabels = []
    for label in tqdm(labels, desc='converting labels'):
        if label == 0:
            multilabels.append(torch.tensor([1,0,0]).float())
        elif label == 1:
            multilabels.append(torch.tensor([0,1,0]).float())
        elif label == 2:
            multilabels.append(torch.tensor([0,0,1]).float())
            
    return multilabels

# %% Data load
data = pd.read_feather(input_path)

# %% Count duplicated news
print(f'Number of duplicated news = {sum(data.text.duplicated())}')

# %% Preprocess dataset
strings = [(' ').join(list(x)) for x in data.tokens]
data['strings'] = strings
data = data[['strings', args['selected_label']]]
data = data.rename(columns = {args['selected_label']: 'Y'})

# %% Check dataset size and label balance
print(f'Shape full dataset = {data.shape}')
_ = pd.value_counts(data['Y']).plot(kind='bar')

# %% Train - test split
train_set, test_set = train_test_split(data,
                                       test_size=args['test_size'],
                                       shuffle=False)

# %% Check sizes
train_pct = len(train_set)/len(data) 
test_pct = len(test_set)/len(data) 

print(f'Shape train dataset = {train_set.shape}\t{train_pct*100:.2f}%')
print(f'Shape test dataset = {test_set.shape}\t\t{test_pct*100:.2f}%')

# %% Check train label balance
_ = pd.value_counts(train_set['Y']).plot(kind='bar')

# %% Resample train set
ros = RandomOverSampler(random_state=1234)
train_X = pd.DataFrame(train_set['strings'])
train_Y = train_set['Y']

train_X_res, train_Y_res = ros.fit_sample(train_X, train_Y)

train_set = pd.concat([train_X_res, train_Y_res], axis=1)

# %% Check train label balance
_ = pd.value_counts(train_set['Y']).plot(kind='bar')

# %% Check sizes
len_data_res = len(train_set) + len(test_set)
train_pct = len(train_set)/len_data_res
test_pct = len(test_set)/len_data_res

print(f'Shape resampled train dataset = {train_set.shape}\t{train_pct*100:.2f}%')
print(f'Shape resampled test dataset = {test_set.shape}\t{test_pct*100:.2f}%')

# %% Tokenize train set
token_ids, token_types, att_masks = BERT_tokenize_f(train_set['strings'],
                                                    **args)
train_set['token_ids'] = token_ids
train_set['token_types'] = token_types
train_set['att_masks'] = att_masks

# %% Tokenize test set
token_ids, token_types, att_masks = BERT_tokenize_f(test_set['strings'],
                                                    **args)
test_set['token_ids'] = token_ids
test_set['token_types'] = token_types
test_set['att_masks'] = att_masks

# %% Convert labels to multilabels
train_set['Y'] = to_multilabel_f(train_set['Y'])
test_set['Y'] = to_multilabel_f(test_set['Y'])

# %% Reorder outputs
train_set = train_set[['strings', 'token_ids', 'token_types',
                      'att_masks', 'Y']]
test_set = test_set[['strings', 'token_ids', 'token_types',
                     'att_masks', 'Y']]

# %% Save outputs
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)

train_path = os.path.join(output_folder, 'model_train.pkl')
test_path = os.path.join(output_folder, 'model_test.pkl')

print(f'Saving datasets to {output_folder}')
train_set.to_pickle(train_path)
test_set.to_pickle(test_path)
print('Done')
