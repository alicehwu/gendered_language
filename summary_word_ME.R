# summarize marginal effects of each word across models
# Nov 2019 

dir_data='/Users/alicehwu/Dropbox/EJR_project/data-pp/Wu_data_P20181101_latest/'
dir_plots='/Users/alicehwu/Dropbox/EJR_project/EJR_2018/plots/'

library(stargazer)
library(reshape2)
library(ggplot2)


# datasets
vocab10K=read.csv(paste0(dir_data,"vocab10K.csv")) 
i_keep_columns=read.csv(paste0(dir_data,"lasso/outputs/i_keep_columns.txt"),header=FALSE) # min=0

# coefficients from Logit Models 
coef_m0=read.csv(paste0(dir_data,'lasso/outputs/logit_coef.txt'),header=FALSE)
coef_m1=read.csv(paste0(dir_data,'lasso/outputs/logit_lasso_coef.txt'),header=FALSE)
coef_m2=read.csv(paste0(dir_data,'lasso/outputs/logit_ridge_coef.txt'),header=FALSE)
coef_m0_2=read.csv(paste0(dir_data,'lasso/outputs/linear_coef.txt'),header=FALSE)
coef_m1_2=read.csv(paste0(dir_data,'lasso/outputs/linear_lasso_coef.txt'),header=FALSE)
coef_m2_2=read.csv(paste0(dir_data,'lasso/outputs/linear_ridge_coef.txt'),header=FALSE)

# avg marginal effects 
ypred_m0=read.csv(paste0(dir_data,'lasso/outputs/logit_ypred_pronoun_train.txt'),header=FALSE)
x0=mean(ypred_m0$V1 * (1-ypred_m0$V1)) # [1] 0.1654892
ypred_m1=read.csv(paste0(dir_data,'lasso/outputs/logit_lasso_ypred_pronoun_train.txt'),header=FALSE)
x1=mean(ypred_m1$V1 * (1-ypred_m1$V1)) # [1] 0.1617864
ypred_m2=read.csv(paste0(dir_data,'lasso/outputs/logit_ridge_ypred_pronoun_train.txt'),header=FALSE)
x2=mean(ypred_m2$V1 * (1-ypred_m2$V1)) # [1] 0.1688919

ME_m0=coef_m0$V1 * x0
ME_m1=coef_m1$V1 * x1
ME_m2=coef_m2$V1 * x2

all_models=data.frame(index=i_keep_columns$V1+1,coef_logit=coef_m0$V1, coef_logit_l1=coef_m1$V1,coef_logit_l2=coef_m2$V1,
           ME_logit=ME_m0,ME_logit_l1=ME_m1,ME_logit_l2=ME_m2,
           coef_linear=coef_m0_2$V1,coef_linear_l1=coef_m1_2$V1,coef_linear_l2=coef_m2_2$V1)

vocab10K=merge(vocab10K,all_models,by="index",all.x=T)

# lasso vs ridge
head(vocab10K[order(vocab10K$ME_logit_l1,decreasing=T),c("word","ME_logit_l1","ME_logit_l2")])



# Tab A1: Logit models -> top F words
tab_A1=head(vocab10K[order(vocab10K$ME_logit,decreasing=T),c("word","ME_logit")],10)
tab_A1=cbind(tab_A1,head(vocab10K[order(vocab10K$ME_logit_l1,decreasing=T),c("word","ME_logit_l1")],10))
tab_A1=cbind(tab_A1,head(vocab10K[order(vocab10K$ME_logit_l2,decreasing=T),c("word","ME_logit_l2")],10))
stargazer(tab_A1,summary = FALSE,ndigits=3)

# linear models
tab_A1_2=head(vocab10K[order(vocab10K$coef_linear,decreasing=T),c("word","coef_linear")],10)
tab_A1_2=cbind(tab_A1_2,head(vocab10K[order(vocab10K$coef_linear_l1,decreasing=T),c("word","coef_linear_l1")],10))
tab_A1_2=cbind(tab_A1_2,head(vocab10K[order(vocab10K$coef_linear_l2,decreasing=T),c("word","coef_linear_l2")],10))
stargazer(tab_A1_2,summary = FALSE,ndigits=3)

# Tab A2: Logit models -> top M words
vocab10K=vocab10K[vocab10K$word!='15,000',]
tab_A2=head(vocab10K[order(vocab10K$ME_logit,decreasing=F),c("word","ME_logit")],10)
tab_A2=cbind(tab_A2,head(vocab10K[order(vocab10K$ME_logit_l1,decreasing=F),c("word","ME_logit_l1")],10))
tab_A2=cbind(tab_A2,head(vocab10K[order(vocab10K$ME_logit_l2,decreasing=F),c("word","ME_logit_l2")],10))
stargazer(tab_A2,summary = FALSE,ndigits=3)

# linear models
tab_A2_2=head(vocab10K[order(vocab10K$coef_linear,decreasing=F),c("word","coef_linear")],10)
tab_A2_2=cbind(tab_A2_2,head(vocab10K[order(vocab10K$coef_linear_l1,decreasing=F),c("word","coef_linear_l1")],10))
tab_A2_2=cbind(tab_A2_2,head(vocab10K[order(vocab10K$coef_linear_l2,decreasing=F),c("word","coef_linear_l2")],10))
stargazer(tab_A2_2,summary = FALSE,ndigits=3)


# grouped bar plots (graphic comparison of magnitude)
# https://www.r-graph-gallery.com/48-grouped-barplot-with-ggplot2.html
words=c('hot','marry','attractive','nobel','adviser','handsome')
# tab_A3=vocab10K[vocab10K$word %in% words,c("word","ME_logit","ME_logit_l1","ME_logit_l2")]
tab_A3=vocab10K[vocab10K$word %in% words,c("word","ME_logit","ME_logit_l1","ME_logit_l2",
                                           "coef_linear","coef_linear_l1","coef_linear_l2")]
tab_A3=melt(tab_A3,id.vars = 'word')
tab_A3=tab_A3[order(tab_A3$word),]

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
plot_F=ggplot(tab_A3[tab_A3$value>0,],aes(x=word,y=value))+
  geom_bar(position='dodge',stat='identity',aes(fill=variable),alpha=0.6,width=0.5)+ 
  #scale_fill_manual(values=c("#56B4E9","#CC79A7","#009E73"),labels=c("Logit","Logit, Lasso","Logit,Ridge"))+ 
  scale_fill_manual(values=cbPalette,labels=c("Logit","Logit, Lasso","Logit,Ridge","Linear","Linear, Lasso","Linear, Ridge"))+ 
  coord_flip() +
  theme(legend.title=element_blank(),legend.text=element_text(size=8),
        panel.background = element_blank(),
        axis.line = element_line(colour = "gray"),axis.title=element_text(size=9),
        axis.text.y=element_text(size=12))
ggsave(paste0(dir_plots,'top_F_across_models.pdf'),plot_F,width=6,height=3)

plot_M=ggplot(tab_A3[tab_A3$value<0,],aes(x=word,y=value))+
  geom_bar(position='dodge',stat='identity',aes(fill=variable),alpha=0.6,width=0.5)+ 
  # scale_fill_manual(values=c("#56B4E9","#CC79A7","#009E73"),labels=c("Logit","Logit, Lasso","Logit,Ridge"))+ 
  scale_fill_manual(values=cbPalette,labels=c("Logit","Logit, Lasso","Logit,Ridge","Linear","Linear, Lasso","Linear, Ridge"))+
  coord_flip() +
  theme(legend.title=element_blank(),legend.text=element_text(size=8),
        panel.background = element_blank(),
        axis.line = element_line(colour = "gray"),axis.title=element_text(size=9),
        axis.text.y=element_text(size=12))
ggsave(paste0(dir_plots,'top_M_across_models.pdf'),plot_M,width=6,height=3)






### Model Performance ### 
posts=read.csv(paste0(dir_data,"gendered_posts.csv"),stringsAsFactors = FALSE)
y_test0=0+(posts[posts$training_pronoun==0 & !is.na(posts$training_pronoun),]$fem_all>0)

models=c('logit','logit_lasso','logit_ridge','linear','linear_lasso','linear_ridge')
cutoffs=seq(0.2,0.8,by=0.05)
for (i in models){
  pred_test0=read.csv(paste0(dir_data,'lasso/outputs/',i,'_ypred_pronoun_test0.txt'),header=FALSE)
  accuracy=c()
  for (j in cutoffs){
    ypred_test0=0+(pred_test0>=j)
    accuracy=c(accuracy,mean(y_test0==ypred_test0))
  }
  print(paste0(i,'cutoff=',cutoffs[which.max(accuracy)],' score=',max(accuracy)))
}

# [1] "logitcutoff=0.35 score=0.75119349705389"
# [1] "logit_lassocutoff=0.35 score=0.750698894671197"
# [1] "logit_ridgecutoff=0.4 score=0.750053761128554"
# [1] "linearcutoff=0.4 score=0.732377102060126"
# [1] "linearcutoff=0.4 score=0.732377102060126"
# [1] "linear_lassocutoff=0.35 score=0.743666939056385"
# [1] "linear_ridgecutoff=0.35 score=0.746290482129801"

# names(vocab10K)[14]='coef_linear_l1' # from 'linear_coef' as of Jan 2018
list_ME=c('ME_logit','ME_logit_l1','ME_logit_l2','coef_linear','coef_linear_l1','coef_linear_l2')
for (i in list_ME){
  print(summary(vocab10K[,i],na.rm=T))
  print(sd(vocab10K[,i],na.rm=T))
  print(sum(vocab10K[,i]!=0,na.rm=T))
}




### Coefficient Stability across 5 Folds ### 
coef_path_l1=read.csv(paste0(dir_data,'lasso/outputs/logit_lasso_coef_path.csv'),header=FALSE)
coef_path_l1=coef_path_l1[1:nrow(coef_path_l1)-1,]
names(coef_path_l1)=paste0('coef_logit_l1_',c(1:5))
for (i in 1:5){
  coef_path_l1[,paste0('ME_logit_l1_',i)]=coef_path_l1[,paste0('coef_logit_l1_',i)] * x1
}
coef_path_l1$index=i_keep_columns$V1+1
coef_path_l1=merge(coef_path_l1,vocab10K[,c("index","word","ME_logit_l1")],all.x=T)

coef_path_l1$sd_ME=apply(coef_path_l1[,7:11],1,function(x) sd(x))
mean(coef_path_l1$ME_logit_l1) # [1] -0.001251561
mean(coef_path_l1$sd_ME) # [1] 0.006157636
mean(coef_path_l1[coef_path_l1$ME_logit_l1==0,]$sd_ME) # [1] 0.001432239
mean(coef_path_l1[coef_path_l1$ME_logit_l1!=0,]$sd_ME) # [1] 0.01457518

coef_path_l1=coef_path_l1[order(coef_path_l1$ME_logit_l1,decreasing=T),]
mean(coef_path_l1[1:50,]$ME_logit_l1) # [1] 0.1692379
mean(coef_path_l1[1:50,]$sd_ME) # 0.02822516
coef_path_l1=coef_path_l1[order(coef_path_l1$ME_logit_l1,decreasing=F),]
mean(coef_path_l1[1:50,]$ME_logit_l1) # [1] -0.1432403
mean(coef_path_l1[1:50,]$sd_ME) # [1] 0.0281295


# graphic illustration for top words (Lasso model only)
tab_coef_path=coef_path_l1[coef_path_l1$word %in% words,]
tab_coef_path=melt(tab_coef_path[,c("word","ME_logit_l1_1","ME_logit_l1_2","ME_logit_l1_3","ME_logit_l1_4","ME_logit_l1_5")],id.vars = 'word')
tab_coef_path=tab_coef_path[order(tab_coef_path$word),]

plot_F_cv=ggplot(tab_coef_path[tab_coef_path$value>0,],aes(x=word,y=value))+
  geom_bar(position='dodge',stat='identity',aes(fill=variable),alpha=0.6,width=0.5)+ 
  scale_fill_manual(values=cbPalette,labels=c("1st Fold","2nd","3rd","4th","5th"))+ 
  coord_flip() +
  theme(legend.title=element_blank(),legend.text=element_text(size=8),
        panel.background = element_blank(),
        axis.line = element_line(colour = "gray"),axis.title=element_text(size=9),
        axis.text.y=element_text(size=12))
ggsave(paste0(dir_plots,'top_F_across_cv.pdf'),plot_F_cv,width=6,height=3)

plot_M_cv=ggplot(tab_coef_path[tab_coef_path$value<0,],aes(x=word,y=value))+
  geom_bar(position='dodge',stat='identity',aes(fill=variable),alpha=0.6,width=0.5)+ 
  scale_fill_manual(values=cbPalette,labels=c("1st Fold","2nd","3rd","4th","5th"))+ 
  coord_flip() +
  theme(legend.title=element_blank(),legend.text=element_text(size=8),
        panel.background = element_blank(),
        axis.line = element_line(colour = "gray"),axis.title=element_text(size=9),
        axis.text.y=element_text(size=12))
ggsave(paste0(dir_plots,'top_M_across_cv.pdf'),plot_M_cv,width=6,height=3)


# check coef_path under Logit and Logit-Ridge
models=c("logit","logit_ridge")
x=c(x0,x2)
for (l in 1:length(models)){
  coef_path=read.csv(paste0(dir_data,'lasso/outputs/',models[l],'_coef_path.csv'),header=FALSE)
  coef_path=coef_path[1:nrow(coef_path)-1,]
  names(coef_path)=paste0('coef_logit_',c(1:5))
  for (i in 1:5){
    coef_path[,paste0('ME_logit_',i)]=coef_path[,paste0('coef_logit_',i)] * x[l]
  }
  coef_path$index=i_keep_columns$V1+1
  coef_path=merge(coef_path,vocab10K[,c("index","word","ME_logit","ME_logit_l1","ME_logit_l2")],all.x=T)
  
  coef_path$sd_ME=apply(coef_path[,7:11],1,function(x) sd(x))
  
  if (l==1){var='ME_logit'}else{var='ME_logit_l2'}
  
  print("All Words")
  print(mean(coef_path[,var])) # [1] -0.001251561
  print(mean(coef_path$sd_ME)) # [1] 0.006157636
  print(mean(coef_path[coef_path[,var]==0,]$sd_ME)) # [1] 0.001432239
  print(mean(coef_path[coef_path[,var]!=0,]$sd_ME)) # [1] 0.01457518
  
  print("Top Female Words")
  coef_path=coef_path[order(coef_path$ME_logit_l1,decreasing=T),]
  print(mean(coef_path[1:50,var])) # [1] 0.1692379
  print(mean(coef_path[1:50,]$sd_ME)) # 0.02822516
  print("Top Male Words")
  coef_path=coef_path[order(coef_path$ME_logit_l1,decreasing=F),]
  print(mean(coef_path[1:50,var])) # [1] -0.1432403
  print(mean(coef_path[1:50,]$sd_ME)) # [1] 0.0281295
}
