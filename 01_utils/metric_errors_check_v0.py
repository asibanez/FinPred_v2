import pandas as pd
import random
from sklearn.metrics import classification_report

#%%
path = 'C:/Users/siban/Dropbox/Desktop/bow_predictions_max_features_2000_lr_0.0002_dropout_0.3.csv'
data_raw = pd.read_csv(path)

#%%
data = data_raw.drop(columns = ['Unnamed: 0', 'dscd', 'label1'])

#%% EDA
pd.value_counts(data.predicted)
pd.value_counts(data.target)

#%% Accuracy
num_correct = sum(data.predicted == data.target)
num_total = len(data.predicted)
accuracy = num_correct / num_total

#%% Classification report
print(f'accuracy = {accuracy:.2f}')
#%%
print('\nModel')
print(classification_report(data.target, data.predicted, digits = 4))

#%% Random classifier
pred_random = []
for idx in range(0, num_total):
    pred_random.append(random.randint(0,2))

#%% Classification report
print(f'accuracy = {accuracy:.2f}')
#%%
print('\nRandom classifier')
print(classification_report(data.target, pred_random))


#%%
print('\nModel')
print(classification_report(data.target, data.predicted, digits = 4))
print('Random classifier')
print(classification_report(data.target, pred_random))