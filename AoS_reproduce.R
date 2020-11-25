set.seed(915)
N = 100*5   # number of train + valid points
n = 100*5   # number of train points
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
for(i in 1:10000){
  row.ind = sample(N,n)   # randomly selected rows
  y.train = y[row.ind]   # regression based on this is not as good as using residuals 
  X.valid = X[row.ind,]
  y.valid = y[row.ind]
  permu.p = sample(p)     # shuffle columns
  times.draw.p0 = floor(p/p0) # split the shuffled column indices into [floor(p/p0)*p0 + left], suppose p%%p0=0 
  #col.ind = sample(p,p0)  # randomly selected cols for once
  b.temp = matrix(rep(0,times.draw.p0*p),ncol=times.draw.p0)
  valid.err.b.temp = vector()
  for(j in (1:times.draw.p0)){
    col.ind = permu.p[((j-1)*p0+1):(j*p0)]
    #print(col.ind)
    X.train = X[row.ind,col.ind]
    y.res = y.train - as.matrix(X.train) %*% b[col.ind]   # residual obtained by using previous b
    
    b.res = rep(0,p)
    train.res = data.frame(y.res, X.train)
    
    lm = lm(y.res~.+0, data=train.res)
    b.res[col.ind] = lm$coefficients
    b.temp[,j] = b.res + b
    valid.err.b.temp[j] = mean((as.matrix(X.valid) %*% as.matrix(b.temp[,j]) - y.valid)^2)
  }
  valid.err.b = mean((as.matrix(X.valid) %*% b - y.valid)^2)
  cond = max(valid.err.b - valid.err.b.temp)>0
  j0 = which.max(valid.err.b - valid.err.b.temp)
  #print(valid.err.b - valid.err.b.temp[j0])
  b = 0.8*b + 0.2*b.temp[,j0] 

  #dif[i] = mean((b-beta)^2)   # record diff between b and beta
  valid.err[i] = mean((as.matrix(X.valid) %*% b - y.valid)^2)  # valid error using updated b
  err[i] = mean((as.matrix(d.test[,-1]) %*% b - d.test$y)^2) # record prediction error on test set
  if(i %% 10 == 0){print(c(i,valid.err[i],err[i]))}
}
plot(err)
abline(h=test.err,col="red")
text(1000,20,"test error by using elastic net on the whole training set.", col="red")
plot(valid.err)
abline(h=test.err,col="red")
text(1000,15,"test error by using elastic net on the whole training set.", col="red")

library(tidyverse)
if(-1>0){
  ggplot(result, aes(x=rep((1:1000),6), y=err, colour=p0, group=p0)) + geom_line() + labs(x="step")
}


