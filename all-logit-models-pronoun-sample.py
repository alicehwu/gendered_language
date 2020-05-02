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
# np.savetxt(dir_outputs+"i_keep_columns.txt",i_keep_columns) # later this can be merged by estimated coefficients (in the same order as these indices) 

X_train=X_train[:,i_keep_columns] 
print(X_train.shape)              
X_test0=X_test0[:,i_keep_columns] 
print(X_test0.shape)              
X_test1=X_test1[:,i_keep_columns] 
print(X_test1.shape)              



################################################################################################################
											### All Logistic Models ###
											# (0, l1, l2) 
################################################################################################################
from sklearn.linear_model import LogisticRegressionCV # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
# from sklearn.linear_model import LogisticRegression # no cross validation

#----------------------------------------------------------------------------------------------------------# 
# (0) logistic 
# print('*** Model 0: Logistic, no penalty ***')
# model0=LogisticRegressionCV(Cs=[1e50],cv=5,penalty='l2',solver='liblinear',refit=True).fit(X_train,y_train) 
# # large Cs in list ~ inverse of lambda=0
# # Cs integer ==> search over grid [1e-4, 1e4]

# coef=model0.coef_ # does not include intercept, check intercept_
# len(coef[0])  
# np.savetxt(dir_outputs+"logit_coef.txt",coef[0]) 

# # coef path
# coef_path=model0.coefs_paths_[1] # dict with dict_keys([1.0]) # coef_path=coef_path[1.0]
# coef_path.shape # (n_folds, n_cs, n_features + 1)

# # optimal coef path: lambda=0 here! 
# opt_coef_path=coef_path[:,0,:] # ~ 3D to 2D array
# opt_coef_path=np.transpose(opt_coef_path)
# np.savetxt(dir_outputs+'logit_coef_path.csv', opt_coef_path, delimiter=",")


# # predicted probability for a post being Female 
# ypred_train=model0.predict_proba(X_train)[:,1] # Pr(female=1)
# ypred_test0=model0.predict_proba(X_test0)[:,1]
# ypred_test1=model0.predict_proba(X_test1)[:,1]

# # "ypred_pronoun" in the files below have been brought back into "genderd_posts.csv"
# np.savetxt(dir_outputs+"logit_ypred_pronoun_train.txt",ypred_train)
# np.savetxt(dir_outputs+"logit_ypred_pronoun_test0.txt",ypred_test0)
# np.savetxt(dir_outputs+"logit_ypred_pronoun_test1.txt",ypred_test1) 

# print('done with model 0')

#----------------------------------------------------------------------------------------------------------# 
# # (1) logistic LASSO Model 
# print('*** Model 1: Logistic Lasso ***')
# model1=LogisticRegressionCV(Cs=20,cv=5,penalty='l1',solver='liblinear',refit=True,max_iter=1e5).fit(X_train,y_train)

# coef=model1.coef_ # intercept?
# coef=coef[0]
# len(coef)  
# np.savetxt(dir_outputs+"logit_lasso_coef.txt",coef) 

# # scores
# s=model1.scores_[1]  # s=s[1.0] # (cv,n_Cs)
# mean_scores=np.mean(s,axis=0) # means of each column, (n_CS,) mean scores
# i_opt=np.where(mean_scores==max(mean_scores)) # tuple of length 1
# i_opt=i_opt[0][0] # get the index

# # optimal tuning paramter: maximize accuracy scores across folds 
# opt_lambda=1/model1.Cs_[i_opt]
# # lambda = 1/Cs # tuning_lambda=1/model1.Cs_
# print('check if lambda %s = 1/optimal C' %opt_lambda)
# print(opt_lambda==1/model1.C_[0])

# # coef path across five folds, at the same opt_lambda
# coef_path=model1.coefs_paths_[1] # coef_path=coef_path[1.0] , dict with dict_keys([1.0])
# coef_path.shape # (n_folds, n_cs, n_features + 1)
# opt_coef_path=coef_path[:,i_opt,:] # n_cv by nwords
# opt_coef_path=np.transpose(opt_coef_path)
# np.savetxt(dir_outputs+'logit_lasso_coef_path.csv', opt_coef_path, delimiter=",")


# # predicted probability for a post being Female 
# ypred_train=model1.predict_proba(X_train)[:,1] # Pr(female=1)
# ypred_test0=model1.predict_proba(X_test0)[:,1]
# ypred_test1=model1.predict_proba(X_test1)[:,1]

# # "ypred_pronoun" in the files below have been brought back into "genderd_posts.csv"
# np.savetxt(dir_outputs+"logit_lasso_ypred_pronoun_train.txt",ypred_train)
# np.savetxt(dir_outputs+"logit_lasso_ypred_pronoun_test0.txt",ypred_test0)
# np.savetxt(dir_outputs+"logit_lasso_ypred_pronoun_test1.txt",ypred_test1) 

# print('done with model 1')

#----------------------------------------------------------------------------------------------------------# 
# (2) logistic Ridge 
print('*** Model 2: Logistic Ridge ***')
model2=LogisticRegressionCV(Cs=20,cv=5,penalty='l2',solver='liblinear',refit=True,max_iter=1e5).fit(X_train,y_train)

coef=model2.coef_ # intercept?
coef=coef[0]
len(coef)  
np.savetxt(dir_outputs+"logit_ridge_coef.txt",coef) 

# scores
s=model2.scores_[1]  # s=s[1.0] # (cv,n_Cs)
mean_scores=np.mean(s,axis=0) # means of each column, (n_CS,) mean scores
i_opt=np.where(mean_scores==max(mean_scores)) # tuple of length 1
i_opt=i_opt[0][0] # get the index

# optimal tuning paramter: maximize accuracy scores across folds 
opt_lambda=1/model2.Cs_[i_opt]
# lambda = 1/Cs # tuning_lambda=1/model2.Cs_
print('check if lambda %s = 1/optimal C' %opt_lambda)
print(opt_lambda==1/model2.C_[0])

# coef path across five folds, at the same opt_lambda
coef_path=model2.coefs_paths_[1] # coef_path=coef_path[1.0] , dict with dict_keys([1.0])
coef_path.shape # (n_folds, n_cs, n_features + 1)
opt_coef_path=coef_path[:,i_opt,:] # n_cv by nwords
opt_coef_path=np.transpose(opt_coef_path)
np.savetxt(dir_outputs+'logit_ridge_coef_path.csv', opt_coef_path, delimiter=",")


# predicted probability for a post being Female 
ypred_train=model2.predict_proba(X_train)[:,1] # Pr(female=1)
ypred_test0=model2.predict_proba(X_test0)[:,1]
ypred_test1=model2.predict_proba(X_test1)[:,1]

# "ypred_pronoun" in the files below have been brought back into "genderd_posts.csv"
np.savetxt(dir_outputs+"logit_ridge_ypred_pronoun_train.txt",ypred_train)
np.savetxt(dir_outputs+"logit_ridge_ypred_pronoun_test0.txt",ypred_test0)
np.savetxt(dir_outputs+"logit_ridge_ypred_pronoun_test1.txt",ypred_test1) 

print('done with model 2')
