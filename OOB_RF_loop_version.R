file = "D:/Research/Discussion/Dan Nettleton/Subsampling/Code/Concrete_Data.csv"
data = read.csv(file, sep = ",")
data = data[,c(9,1:8)]
names(data)[1] = "y"
data[,-1] = scale(data[,-1])
data = as.data.frame(data)

set.seed(915)

library(cluster)
library(caret)
library(tidyverse)

n = dim(data)[1]
n.train = round(0.8*n)
n.test = n - n.train

RMSE = list()
for(tt in 1:20){
b = 0.025 * tt

B = round(b*n)    # B < n.train should be satisfied
n.rnd.smpl = round(0.5 * B)
n.seq.smpl = B - n.rnd.smpl

test.rmse = vector()
test.rmse.rand = vector()

for(t in 1:100){
print(t)
train.index = sample(n,n.train)
train = data[train.index,]
rnd.smpl = train[(1:n.rnd.smpl),]
seq.pool = train[-(1:n.rnd.smpl),] # to draw n.seq.smpl points from seq.pool
test = data[-train.index,]

# initial RF -> OOB error -> new reponse RF -> assign weight to unseen data for sampling
fit.rnd.smpl = train(y=rnd.smpl[,1], x=rnd.smpl[,-1], tuneGrid=data.frame(mtry=1:8), method="rf",
                     ntree=1000, trControl=trainControl(method="oob"))

# check OOB error for each point in rnd.smpl and set new response
response = (rnd.smpl[,1] - predict(fit.rnd.smpl,rnd.smpl))^2
res.fit.rnd.smpl = train(y=response, x=rnd.smpl[,-1], tuneGrid=data.frame(mtry=1:8), method="rf",
                         ntree=1000, trControl=trainControl(method="oob"))
alpha = 1
OOB.weight = (predict(res.fit.rnd.smpl, seq.pool))^alpha
OOB.weight = OOB.weight/sum(OOB.weight)
hist(OOB.weight)

# stochastic sampling
sum.weight = vector()
sum.weight[1] = (OOB.weight[1])^alpha
for(i in 2:(n.train-n.rnd.smpl)){
  sum.weight[i] = sum.weight[i-1] + (OOB.weight[i])^alpha
}
OOB.sample = data.frame(index=rownames(seq.pool), weight=OOB.weight, s.weight=sum.weight, state=0)
while(sum(OOB.sample$state)<n.seq.smpl){
  s = OOB.sample[1,]
  rn = runif(1)
  if(0==s$state & rn<s$s.weight){OOB.sample[1,]$state=1}
  for(i in 2:dim(OOB.sample)[1]){
    rn = runif(1)
    s = OOB.sample[i,]
    s1w = OOB.sample[i-1,]$s.weight
    if(0==s$state & rn<s$s.weight & rn>s1w){OOB.sample[i,]$state=1}
  }
}
hist(OOB.sample[OOB.sample$state==1,]$weight, 
     xlab="subsampled OOB.weight", main=paste0("Histogram of subsampled OOB.weight, alpha=",alpha))

# creat new dataset and run RF
seq.smpl = train[OOB.sample[OOB.sample$state==1,]$index,]
d = rbind(rnd.smpl,seq.smpl)
fit = train(y=d[,1], x=d[,-1], tuneGrid=data.frame(mtry=1:8), method="rf",
            ntree=1000, trControl=trainControl(method="oob"))
#test.rmse = sqrt(mean((predict(fit,test)-test[,1])^2))
test.rmse[t] = sqrt(mean((predict(fit,test)-test[,1])^2))
#test.rmse[t]

# benchmark, uniformly random sampling
rand.index = sample(n.train,B)
rand = train[rand.index,]
fit.rand = train(y=rand[,1], x=rand[,-1], tuneGrid=data.frame(mtry=1:8), method="rf",
                 ntree=1000, trControl=trainControl(method="oob"))
#test.rmse.rand = sqrt(mean((predict(fit.rand,test)-test[,1])^2))
test.rmse.rand[t] = sqrt(mean((predict(fit.rand,test)-test[,1])^2))
#test.rmse.rand[t]
}


rmse = data.frame(test.rmse = test.rmse, test.rmse.rand = test.rmse.rand)
plot(rmse$test.rmse - rmse$test.rmse.rand,ylab="RMSE_OOB - RMSE_rand", pch=20)
abline(h=0,col="red")
RMSE[[tt]] = rmse
}

plot(0.025*c(1:20)/0.8,unlist(lapply(RMSE,function(x){sum(x[,1]<x[,2])}))/100,
     xlab="B/n.train",ylab="Prob(RMSE_weighted < RMSE_rand)",pch=20,
     main="weighting by using OOB error, 0.5*B uniformly sampled points, 100 realizations")
abline(h=0.5,col="red")

plot(0.025*c(1:20)/0.8,unlist(lapply(RMSE,function(x){mean(x[,1])})), pch=19,col="red",
     xlab="B/n.train", ylab="")
points(0.025*c(1:20)/0.8,unlist(lapply(RMSE,function(x){mean(x[,2])})),pch=15,col="blue")
legend("topright",pch=c(19,15), col=c("red","blue"), legend=c("mean RMSE_OOB","mean RMSE_rand"))

plt = ggplot(data = RMSE[[1]], aes(x = variable, y = value))
plt + geom_boxplot() + theme_minimal() + labs(x = "Title", y = "x")

boxplot(c(RMSE[[1]],RMSE[[5]],RMSE[[10]],RMSE[[15]],RMSE[[20]]), col=c("red","blue"))
