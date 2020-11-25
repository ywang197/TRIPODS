file <- "D:/Research/Discussion/Dan Nettleton/Subsampling/Code/Concrete_Data.csv"
data <- read.csv(file, sep = ",")

data <- data[,c(9,1:8)]
names(data)[1] <- "y"

data[,-1] <- scale(data[,-1])
data <- as.data.frame(data)

set.seed(915)

# adaptive wasserstein subsampling
library(cluster)
library(caret)
n = dim(data)[1]
n.train = round(0.8*n)
n.test = n - n.train
n.pretrain = 200

# different realizations could be applied here
train.index = sample(n,n.train)
train = data[train.index,]
test = data[-train.index,]

n.sampling.step = 20
k = round((n.train-n.pretrain)*0.25/n.sampling.step)
m = k*n.sampling.step

step.data = train[,-1]
record.data = train
test.rmse = vector()
subdata = data.frame()

for(i in 1:(n.sampling.step+1)){
  # OT to find k best
  if(1==i){
    subdata = rbind(subdata, record.data[1:n.pretrain,])
    fit = train(y = subdata[,1], x = subdata[,-1], tuneGrid = data.frame(mtry = 1:5), method = "rf",
                ntree = 1000,
                trControl = trainControl(method = "oob"))
    test.rmse[i] = sqrt(mean((predict(fit,test)-test[,1])^2))
    print(test.rmse[i])
    ymax = max(record.data[1:n.pretrain,]$y)
    record.data = record.data[-(1:n.pretrain),]
    step.pred = predict(fit, record.data)
    step.data = cbind(step.pred, record.data[,-1])    # in step.data, use predicted y and X
  }
  pamdata = pam(step.data, k = k, metric = "manhattan")
  subdata = rbind(subdata, record.data[pamdata$id.med,])   # all available data up to this round

  fit = train(y = subdata[,1], x = subdata[,-1], tuneGrid = data.frame(mtry = 1:5), method = "rf",
              ntree = 1000, trControl = trainControl(method = "oob"))
  test.rmse[i] = sqrt(mean((predict(fit,test)-test[,1])^2))
  print(test.rmse[i])
  record.data = record.data[-pamdata$id.med,]
  step.pred = predict(fit, record.data) #/ymax
  step.data = cbind(step.pred, record.data[,-1])    # in step.data, use predicted y and X
}


OT.data = train[pam(train[,-1], k = (m+n.pretrain), metric = "manhattan")$id.med,]
#OT.fit = lm(y~., data = OT.data)
OT.fit = train(y = OT.data[,1], x = OT.data[,-1], tuneGrid = data.frame(mtry = 1:5), method = "rf",
               ntree = 1000, trControl = trainControl(method = "oob"))
OT.rmse = sqrt(mean((predict(OT.fit,test)-test[,1])^2))
OT.rmse

rand.data = train[sample(n.train,m+n.pretrain),]
#rand.fit = lm(y~., data = rand.data)
rand.fit = train(y = rand.data[,1], x = rand.data[,-1], tuneGrid = data.frame(mtry = 1:5), method = "rf",
                 ntree = 1000, trControl = trainControl(method = "oob"))
rand.rmse = sqrt(mean((predict(rand.fit,test)-test[,1])^2))
rand.rmse

plot(test.rmse, ylim=c(5,max(test.rmse)+0.5),pch=19,xlab="number of steps",ylab="RMSE",
     main=paste0("n.train=",n.train,",","n.test=",n.test,",", "n.pretrain=", n.pretrain,",","m=",m))
abline(h=OT.rmse,col="red")
abline(h=rand.rmse,col="blue")
legend("topright",col=c("red","blue"),legend=c("global Wasserstein based on X","random"),lty=1)
