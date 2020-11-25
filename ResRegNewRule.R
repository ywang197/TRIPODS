set.seed(915)
N = 100*5   # number of train + valid points
n = 80*5   # number of train points
m = 30*5   # number of test points
p = 1500   # number of all features
p0 = 150   # number of selected features 
#beta = rep(1,p)   # true coefficients
beta = 0.1*rnorm(n=p)
#beta = rep(0,p)
beta[sample(p,5)] = 1 


X.full = matrix(rnorm(n=(N+m)*p), ncol=p)
y.full = X.full %*% beta + rnorm(n=(N+m),mean=0,sd=0.5)   # y = X * beta + rnorm: true model
X.full = data.frame(X.full)
colnames(X.full) = paste0("x",c(1:p))
data = data.frame(y=y.full, X.full)   # all data

d = data[1:N,]   # train + valid data
d.test = data[(N+1):(N+m),]   # test data
library(hdlm)
ols = hdlm(y~.+0, data=d, alpha=0.05)   # alpha->0:ridge; alpha=1:lasso
b.ols = ols$coefficients
dif = mean((b.ols-beta)^2)
test.err = mean((as.matrix(d.test[,-1])%*%b.ols-d.test$y)^2)

b = rep(0,p)   # initial value of b
X = d[,-1]     # X for train + valid
y = d[,1]      # y for train + valid

# loop
dif = vector()   
err = vector() 
valid.err = vector()
for(i in 100001:1000000){
  row.ind = sample(N,n)   # randomly selected rows
  col.ind = sample(p,p0)  # randomly selected cols
  X.train = X[row.ind,col.ind]
  y.train = y[row.ind]   # regression based on this is not as good as using residuals 
  y.res = y.train - as.matrix(X.train) %*% b[col.ind]   # residual obtained by using previous b
  
  X.valid = X[-row.ind,]
  y.valid = y[-row.ind]
  
  
  b.res = rep(0,p)
  train.res = data.frame(y.res, X.train)
  
  lm = lm(y.res~.+0, data=train.res)
  b.res[col.ind] = lm$coefficients
  b.temp = b.res + b
  train.err.b = mean(y.res^2)
  train.err.b.temp = mean((y.train - as.matrix(X.train) %*% b.temp[col.ind])^2)
  valid.err.b = mean((as.matrix(X.valid) %*% b - y.valid)^2)
  valid.err.b.temp = mean((as.matrix(X.valid) %*% (b.temp) - y.valid)^2)
  cond1 = valid.err.b.temp < valid.err.b
  cond2 = train.err.b.temp < train.err.b 
  if(cond1 & cond2){
    b = b + 0.2*b.res  #/log(1+i)   
  }
  dif[i] = mean((b-beta)^2)   # record diff between b and beta
  valid.err[i] = mean((as.matrix(X.valid) %*% b - y.valid)^2)
  err[i] = mean((as.matrix(d.test[,-1]) %*% b - d.test$y)^2) # record prediction error on test set
  if(i %% 10 == 0){print(c(i,dif[i],err[i]))}
}
#plot(dif, ylim=c(0.8*ols.dif,dif[1]*1.1))
#abline(h=ols.dif,col="red")
plot(err,ylim=c(15,25))
abline(h=test.err,col="red")
text(100000,17,"test error by using elastic net on the whole training set.", col="red")
#plot(diff)
plot(valid.err,ylim=c(0,30))
abline(h=test.err,col="red")
text(100000,20,"test error by using elastic net on the whole training set.", col="red")
#plot(err,ylim=c(0,5),ylab="test.err")
#abline(h=test.err,col="red")
#text(5000,0.5,"test error by using lasso on the whole training set.", col="red")
# = data.frame()
#result = rbind(result, data.frame(dif=dif, dif.noupdate=dif.noupdate, err=err, p0=p0))
library(tidyverse)
if(-1>0){
  #ggplot(result, aes(x=rep((1:1000),6), y=dif, colour=p0, group=p0)) + geom_line() + labs(x="step")
  ggplot(result, aes(x=rep((1:1000),6), y=err, colour=p0, group=p0)) + geom_line() + labs(x="step")
  #ggplot(result, aes(x=rep((1:1000),6), y=dif.noupdate, colour=p0, group=p0)) + geom_line() + labs(x="step")
  
  #compare = aggregate(result[,c(1,2)], list(result$p0), mean)
  #colnames(compare)[1] = "p0"
  #ggplot(compare, aes(p0)) + 
  #  geom_point(aes(y = dif, colour = "dif")) + 
  #  geom_line(aes(y = dif, colour = "dif")) +
  #  geom_point(aes(y = dif.noupdate, colour = "dif.noupdate")) +
  #  geom_line(aes(y = dif.noupdate, colour = "dif.noupdate")) + labs(y="")
}


