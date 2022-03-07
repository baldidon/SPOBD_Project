import scipy.io as scp
import numpy as np
import imblearn
import pandas as pd



data = pd.read_csv('DATASET/dataset.csv')
target = data['stroke']
print(target.shape)
data.drop(['stroke'],axis=1,inplace=True)


from imblearn.over_sampling import SMOTE

oversample = SMOTE(random_state=2)
data, target = oversample.fit_resample(data, target)


data['isFemale'] =  np.floor(data['isFemale'])
data['heart_disease'] =  np.floor(data['heart_disease'])
data['ever_married'] =  np.floor(data['ever_married'])
data['ever_smoked'] =  np.floor(data['ever_smoked'])
data['hypertension'] =  np.floor(data['hypertension'])
data['public_job'] =  np.floor(data['public_job'])
data['private_job'] =  np.floor(data['private_job'])
data['never_worked'] =  np.floor(data['never_worked'])
data['age'] =  np.floor(data['age'])

data['target'] = target

# EXPORT
data.to_csv('DATASET/dataset_balanced', index=False)



