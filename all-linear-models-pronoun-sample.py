import numpy as np 
import pandas as pd 

# dir_data="/Users/alicehwu/Dropbox/EJR_project/data-pp/Wu_data_P20181101_latest/" # specify the directory to data files 
# dir_outputs="/Users/alicehwu/Dropbox/EJR_project/data-pp/Wu_data_P20181101_latest/lasso/outputs/" # where the outputs are saved 

dir_data="/scratch/public/alicehwu/EJR_data/Wu_data_P20181101_latest/" # specify the directory to data files 
dir_outputs="/scratch/public/alicehwu/EJR_data/Wu_data_P20181101_latest/lasso/outputs/" # where the outputs are saved 

### (1) Identify training/test samples 
posts=pd.read_csv(dir_data+"gendered_posts.csv") 
keys_X=pd.read_csv(dir_data+'keys_to_X.csv') # in the same order as rows in matrix x

# additional step to make sure the order is the consistent with the matrix "X" of word counts 
# (This step may be unnecessary if you have sorted posts by title_id and post_id)
keys_merged=pd.merge(keys_X,posts,on=['title_id','post_id'],how="left") 

# note: "non-duplicate" posts contain only female or only male classifiers
i_train=np.where(keys_merged['training_pronoun']==1) # 75% of non-duplicate posts as training sample
i_test0=np.where(keys_merged['training_pronoun']==0) # 25% of non-duplicate posts as test sample for selecting optimal probability threshold
i_test1=np.where((keys_merged['fem_pronoun']>0) & (keys_merged['male_pronoun']>0)) # duplicate posts that include both female and male classifiers; To be reclassified 

# an array of unambiguous gender in the training sample 
y_train=keys_merged.loc[i_train[0],'female_pronoun'].as_matrix() 

### (2) Bring in word count matrix X
word_counts=np.load(dir_data+"X_word_count.npz",encoding='latin1')
X=word_counts['X'][()] 
X_train=X[i_train[0],:]
X_test0=X[i_test0[0],:]
X_test1=X[i_test1[0],:]


### (3) Select Predictors: most frequent 10K excluding gender classifiers & additional last names 
vocab10K=pd.read_csv(dir_data+"vocab10K.csv")
vocab10K['exclude'].sum() 
exclude_vocab=vocab10K.loc[vocab10K['exclude']==1,:]
i_exclude=exclude_vocab['index']-1 # indexing in Python starts from 0, while the indices for vocab are 1 to 10,000

i_columns=range(10000)
i_keep_columns=list(set(i_columns)-set(i_exclude)) 
np.savetxt(dir_outputs+"i_keep_columns.txt",i_keep_columns) # later this can be merged by estimated coefficients (in the same order as these indices) 

X_train=X_train[:,i_keep_columns] 
print(X_train.shape)              
X_test0=X_test0[:,i_keep_columns] 
print(X_test0.shape)              
X_test1=X_test1[:,i_keep_columns] 
print(X_test1.shape)              



################################################################################################################
											### All Logistic Models ###
											# (0, l1, l2) 
# linear Lasso, using previous estimates
# don't need to look at coef_path here 									
################################################################################################################
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

# #----------------------------------------------------------------------------------------------------------# 
# # (0) linear
# model0=LinearRegression().fit(X_train,y_train) 
# # large Cs in list ~ inverse of lambda=0
# # Cs integer ==> search over grid [1e-4, 1e4]

# coef=model0.coef_ # does not include intercept, check intercept_
# len(coef)  
# np.savetxt(dir_outputs+"linear_coef.txt",coef) 

# # predicted probability for a post being Female 
# ypred_train=model0.predict(X_train) # Pr(female=1)
# ypred_test0=model0.predict(X_test0)
# ypred_test1=model0.predict(X_test1)

# # "ypred_pronoun" in the files below have been brought back into "genderd_posts.csv"
# np.savetxt(dir_outputs+"linear_ypred_pronoun_train.txt",ypred_train)
# np.savetxt(dir_outputs+"linear_ypred_pronoun_test0.txt",ypred_test0)
# np.savetxt(dir_outputs+"linear_ypred_pronoun_test1.txt",ypred_test1) 



#----------------------------------------------------------------------------------------------------------# 
# (2) linear Ridge 
print('Model 2: Linear Ridge')

model2=RidgeCV(fit_intercept=True,cv=5).fit(X_train,y_train)

coef=model2.coef_ # intercept?
len(coef)  
np.savetxt(dir_outputs+"linear_ridge_coef.txt",coef) 

# predicted probability for a post being Female 
ypred_train=model2.predict(X_train) # Pr(female=1)
ypred_test0=model2.predict(X_test0)
ypred_test1=model2.predict(X_test1)

# "ypred_pronoun" in the files below have been brought back into "genderd_posts.csv"
np.savetxt(dir_outputs+"linear_ridge_ypred_pronoun_train.txt",ypred_train)
np.savetxt(dir_outputs+"linear_ridge_ypred_pronoun_test0.txt",ypred_test0)
np.savetxt(dir_outputs+"linear_ridge_ypred_pronoun_test1.txt",ypred_test1) 

print('done with model 2')


#----------------------------------------------------------------------------------------------------------# 
# (1) linear LASSO Model 
print('Model 1: Linear Lasso')

model1=LassoCV(cv=5).fit(X_train,y_train)

coef=model1.coef_
print(type(coef))
len(coef)
np.savetxt(dir_outputs+"linear_lasso_coef.txt",coef) 

# predicted probability for a post being Female 
ypred_train=model1.predict(X_train) # Pr(female=1)
ypred_test0=model1.predict(X_test0)
ypred_test1=model1.predict(X_test1)


np.savetxt(dir_outputs+"linear_lasso_ypred_train.txt",ypred_train)
np.savetxt(dir_outputs+"linear_lasso_ypred_test0.txt",ypred_test0)
np.savetxt(dir_outputs+"linear_lasso_ypred_test1.txt",ypred_test1)

print('done with model 1')




