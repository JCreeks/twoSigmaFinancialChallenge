rm(list=ls(all=TRUE)) 
cat("\014") 

source("http://bioconductor.org/biocLite.R")
biocLite("rhdf5")

library(rhdf5)
h5ls("/Users/jingguo/Desktop/KaggleCompetition/input/train.h5")
#mydata <- h5read("/Users/jingguo/Desktop/KaggleCompetition/input/train.h5",
#                 'train/')

df=data.frame(t(h5read("/Users/jingguo/Desktop/KaggleCompetition/input/train.h5",
                       'train/block0_values')),
              t(h5read("/Users/jingguo/Desktop/KaggleCompetition/input/train.h5",
                     'train/block1_values')))
colnames(df)=c(h5read("/Users/jingguo/Desktop/KaggleCompetition/input/train.h5",
                      'train/block0_items'),
               h5read("/Users/jingguo/Desktop/KaggleCompetition/input/train.h5",
                      'train/block1_items')
               )
