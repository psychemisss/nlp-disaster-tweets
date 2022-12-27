import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from catboost.text_processing import Tokenizer

# Load data
train_data = pd.read_csv('data/train.csv')
train_labels = train_data['target']
test_data = pd.read_csv('data/test.csv')

# Drop unnecessary columns
train_data.drop(['text'], axis=1, inplace=True)

# Tokenize keywords and location columns using catboost Tokenizer
train_data['keyword'] = train_data['keyword'].fillna('None')
train_data['location'] = train_data['location'].fillna('None')

keywords = Tokenizer(lowercasing=True, separator_type='BySense', token_types=['Word', 'Number']).tokenize(train_data['keyword'])
locations = Tokenizer(lowercasing=True, separator_type='BySense', token_types=['Word', 'Number']).tokenize(train_data['location'])

print(keywords, locations)

# Create pools
train_pool = Pool(train_data, train_labels)
test_pool = Pool(test_data)

# Initialize CatBoostClassifier
model = CatBoostClassifier(devices='0:1', loss_function='F1', verbose=True)

# Searching for the best parameters
grid_search_params = {
    'depth': [4, 7, 10],
    'learning_rate' : [0.03, 0.1, 0.15],
    'l2_leaf_reg': [1,4,9],
    'border_count': [32,5,10,20,50,100,200],
    'iterations': [100, 250, 500, 1000]
}

best_params = model.grid_search(grid_search_params, train_pool)
print(best_params)

# Fit model
# model.fit(train_pool)

# Get predictions
# preds_class = model.predict(test_pool)
# preds_proba = model.predict_proba(test_pool)
# preds_raw = model.predict(test_pool, prediction_type='RawFormulaVal')

# Save predictions to file
# np.savetxt('catboost_baseline.csv', preds_class, delimiter=',', fmt='%d')

# Save model
# model.save_model('catboost_baseline.cbm', format="cbm", export_parameters=None, pool=None)