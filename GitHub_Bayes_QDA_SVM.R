## Bayes' Classifier and SVM for Gaussian distribution classification, SVM and QDA for USPS Handwritten data classification ##

##########################
#### Data generation #####
##########################

library(MASS)

# mixture of Gaussian densities: 2 components
set.seed(99)
p1 = 0.4                               
p2 = 0.6
n = 200                                

s1 = c(2,2)                       
s2 = c(3,2)
s = rbind(s1,s2)                    

m1 = c(1,2)                            
m2 = c(2,-0.5)
m  = rbind(m1,m2)                   

indx = sample(c(1,2),size=n,prob=c(p1,p2),replace=TRUE)

X = cbind (rnorm (n,m[indx,1],s[indx,1]), rnorm(n,m[indx,2],s[indx,2]))

theLabel = rep(1,n)

theLabel [indx==2] <- 2

plot(X,col = theLabel, xlim = c(min(X[,1]-1),max(X[,1]+1)), ylim = c(min(X[,2]-1),max(X[,2]+1)),xlab = 'samples[,1]',ylab = 'samples[,2]')

rm(indx)

################################################
##### Boundaries of the Bayes Classifier #######
################################################

len <- 50

x_tst <- seq(min(X[,1]),max(X[,1]), length=len)
y_tst <- seq(min(X[,2]),max(X[,2]), length=len)

tst_grid <- expand.grid(z1=x_tst,z2=y_tst)

pdf <- p1 * dnorm(tst_grid[,1],m[1,1],s[1,1])*dnorm(tst_grid[,2],m[1,2],s[1,2])
pdf <- cbind (pdf, p2 * dnorm(tst_grid[,1],m[2,1],s[2,1])*dnorm(tst_grid[,2],m[2,2],s[2,2]))

pdfDiff <- pdf[,1] - pdf[,2]

par(new = TRUE)
contour(x_tst, y_tst, matrix(pdfDiff, len), add=TRUE, levels=0, drawlabels=FALSE,
        xlim = c(min(X[,1]-1),max(X[,1]+1)), ylim = c(min(X[,2]-1),max(X[,2]+1)))

###########################################################################################################
## Support Vector Machine, experiment with several values of the regularization parameter C = 0.1, 1, 10 ##
###########################################################################################################
library(e1071)
theClass <- as.factor(theLabel) 
X = as.matrix(X)

model1of10 = svm(X[1:100,1:2], theClass, kernel = 'linear', cost = 0.1) 
predict_grid1of10 = predict(model1of10, data.frame(tst_grid))
par(new = TRUE)
contour(x_tst, y_tst, matrix(as.numeric(predict_grid1of10), length(x_tst), length(y_tst)), add=TRUE,levels = 1.5)

model1 = svm(X[1:100,1:2], theClass, kernel = 'linear', cost = 1) 
predict_grid1 = predict(model1, data.frame(tst_grid))
par(new = TRUE)
contour(x_tst, y_tst, matrix(as.numeric(predict_grid1), length(x_tst), length(y_tst)), add=TRUE,levels=1.5)

model10 = svm(X[1:100,1:2], theClass, kernel = 'linear', cost = 10) 
predict_grid10 = predict(model10, data.frame(tst_grid))
par(new = TRUE)
contour(x_tst, y_tst, matrix(as.numeric(predict_grid10), length(x_tst), length(y_tst)), add=TRUE,levels=1.5)


###########################################################################
##### Parameter tuning using cross-validation, optimizing parameter C #####
###########################################################################
the_samples = X[,1:2]
Nfolds = 10
foldIndex = sample(1:Nfolds,size=n,prob=rep(1,Nfolds)/Nfolds,replace=TRUE)

c = X[,3]
M = 50;
the_risk = rep(0,M)                   
regul = seq(0.05,10,length.out=M)     

for (k in 1:M) {                        
  
  print(k)                           
  
  for (fold_i in 1:Nfolds) {                   
    
    train.x <- the_samples[foldIndex!=fold_i,] 
    train.c <- theLabel[foldIndex!=fold_i]
    
    test.x = the_samples[foldIndex==fold_i,] 
    test.c = theLabel[foldIndex==fold_i]
    
    
    the_data = data.frame(varY = as.factor(train.c), varX = train.x);
    
    model = svm(varY ~., data = the_data, kernel = 'radial', cost = regul[k])
    
    
    predict_class = predict(model,data.frame(varX = test.x))
    
    the_risk[k] = the_risk[k] + sum (abs (as.numeric(predict_class) - test.c))
  }
}

par(new=FALSE)
plot(regul,the_risk)


regulOpt = which.min(the_risk)
                     

#Bayes Classifier and SVM with optimal C
optimal = svm(X[,1:2], theClass, kernel = 'radial', cost = 0.456) 
predict_optimal = predict(optimal, data.frame(tst_grid))
par(new=TRUE)
contour(x_tst, y_tst, matrix(as.numeric(predict_optimal), length(x_tst), length(y_tst)), add=TRUE,levels = 1.5)

#######################################################################
#### USPS handwritten digit, QDA and SVM, optimize the parameter C ####
#######################################################################

load('zip.train.RData')
load('zip.test.RData')

#Linear discriminant classifier
test = zip.test
train = zip.train
thelabel = train[,1]
theclass <- as.factor(thelabel)

Xj = apply(train[,2:257], 2, jitter)
Xj = cbind(train[,1],Xj)

Xj.lda = lda(Xj[,2:257], theclass)

usps_guess = predict(Xj.lda, test[,-1])

cm = table(test[,1], usps_guess$class)


#Support Vector Machine
the_samples = train[,2:257]
Nfolds = 10
foldIndex = sample(1:Nfolds,size=n,prob=rep(1,Nfolds)/Nfolds,replace=TRUE)


M = 10;
the_risk = rep(0,M)                    
regul = seq(0.05,10,length.out=M)      

for (k in 1:M) {                        
  
  print(k)                           
  
  for (fold_i in 1:Nfolds) {                
    
    train.x = the_samples[foldIndex!=fold_i,] 
    train.c = thelabel[foldIndex!=fold_i]
    
    test.x = the_samples[foldIndex==fold_i,] 
    test.c = thelabel[foldIndex==fold_i]
    
    
    the_data = data.frame(varY = as.factor(train.c), varX = train.x);
    
    model = svm(varY ~., data = the_data, kernel = 'radial', cost = regul[k])
    
    
    predict_class = predict(model,data.frame(varX = test.x))
    
    
    the_risk[k] = the_risk[k] + sum (abs(as.numeric(predict_class) - test.c))
  }
}

plot(regul,the_risk)

regulOpt = which.min(the_risk)




