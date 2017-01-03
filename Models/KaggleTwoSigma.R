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

library(nlme)
autoCorr<-function(ID) {
  array=(df$y)[df$id==ID]
  cor(array[-length(array)],array[-1])
}

acfVec=sapply(unique(df$id),autoCorr)

plot(unique(df$id),acfVec)
highACF_ID=unique(df$id)[(abs(acfVec)>.4)]
highACF_ID=highACF_ID[!is.na(highACF_ID)]

allNACols<-function(ID) {
  colName=colnames(df)[2:length(colnames(df))]
  isNACol=sapply(colName,function(x) {sum(!is.na(df[df$id==ID,x]))})
  #print(paste("id=",ID))
  #print(colName[isNACol==0])
  colName[isNACol==0]
}

sapply(highACF_ID, allNACols)

set=allNACols(highACF_ID[1])

for (i in highACF_ID) {
  set=intersect(set,allNACols(i))
}

sapply(highACF_ID,function(x) {
  print(paste('id=',x))
  print(df$timestamp[df$id==x])
  return}
  )

(unique(df$id))[sapply(unique(df$id),function(x) {
  tmp=df$timestamp[df$id==x]
  (max(tmp)-max(tmp))<20
})]

(unique(df$id))[sapply(unique(df$id),function(x) {
  out=FALSE
  for (col in set) {
    out=out||(sum(!is.na(df$col[df$id==x])))
  }
  !out
})]

