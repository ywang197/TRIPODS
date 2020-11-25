file <- "D:/Research/Discussion/Dan Nettleton/Subsampling/Code/Concrete_Data.csv"
data <- read.csv(file, sep = ",")

data <- data[,c(9,1:8)]
names(data)[1] <- "y"

data[,-1] <- scale(data[,-1])
data <- as.data.frame(data)

# adaptive wasserstein subsampling
library(cluster)
library(caret)
n = dim(data)[1]
n.train = round(0.8*n)
n.test = n - n.train

# different realizations could be applied here
train.index = sample(n,n.train)
train = data[train.index,]
test = data[-train.index,]

n.sampling.step = 50
k = round(n.train*0.25/n.sampling.step)
m = k*n.sampling.step

step.data = train[,-1]
record.data = train
test.rmse = vector()
subdata = data.frame()
#subdata = subdata[F,]
for(i in 1:n.sampling.step){
  # OT to find k best
  pamdata = pam(exp((scale(step.data)))^(n.sampling.step/i), k = k, metric = "manhattan")
  subdata = rbind(subdata, record.data[pamdata$id.med,])   # all available data up to this round

  fit = train(y = subdata[,1], x = subdata[,-1], tuneGrid = data.frame(mtry = 1:5), method = "rf",
              ntree = 1000,
              trControl = trainControl(method = "oob"))
  test.rmse[i] = sqrt(mean((predict(fit,test)-test[,1])^2))
  print(test.rmse[i])
  record.data = record.data[-pamdata$id.med,]
  step.pred = predict(fit, record.data)
  step.data = cbind(step.pred, record.data[,-1])
}


OT.data = train[pam(train[,-1], k = m)$id.med,]
OT.fit = train(y = OT.data[,1], x = OT.data[,-1], tuneGrid = data.frame(mtry = 1:5), method = "rf",
            ntree = 1000, trControl = trainControl(method = "oob"))
OT.rmse = sqrt(mean((predict(OT.fit,test)-test[,1])^2))
OT.rmse

rand.data = train[sample(n.train,m),]
rand.fit = train(y = rand.data[,1], x = rand.data[,-1], tuneGrid = data.frame(mtry = 1:5), method = "rf",
            ntree = 1000, trControl = trainControl(method = "oob"))
rand.rmse = sqrt(mean((predict(rand.fit,test)-test[,1])^2))
rand.rmse

plot(test.rmse, ylim=c(5,max(test.rmse)+0.5),pch=19,xlab="number of steps",ylab="RMSE",
     main=paste0("n.train=",n.train,",","n.test=",n.test,",","m=",m))
abline(h=OT.rmse,col="red")
abline(h=rand.rmse,col="blue")
legend("topright",col=c("red","blue"),legend=c("global Wasserstein based on X","random"),lty=1)
