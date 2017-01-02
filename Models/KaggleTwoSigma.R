source("http://bioconductor.org/biocLite.R")
biocLite("rhdf5")

library(rhdf5)
h5ls("Desktop/train.h5")
mydata <- h5read("Desktop/train.h5", "foo/A")
str(mydata)