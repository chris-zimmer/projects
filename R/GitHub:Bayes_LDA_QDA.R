#Using Bayes' Classifier, Linear Discriminant Anlaysis, Qudratic Discriminant Analysis to classify synthetic data from a Gaussian distribution###
###########################################################
### Generating three multivariate Gaussin distributions ###
mu1 = cbind(c(1, 2))
mu2 = cbind(c(6, 6))
mu3 = cbind(c(6, -2))

cov1 = cbind(c(1, 0), c(0, 4))
cov2 = cbind(c(9, 0), c(0, 1))
cov3 = cbind(c(2.25, 0), c(0, 4))

pi1 = 0.2
pi2 = 0.3
pi3 = 0.5


f = (0.2*f1 + 0.3*f2 + 0.5*f3)
X = f
plot(X[,1],X[,2])

f1 = mvrnorm(20,mu1,cov1)
f2 = mvrnorm(30,mu2,cov2)
f3 = mvrnorm(50,mu3,cov3)

X = rbind(f1, f2, f3)
dev.new()
plot(f1, col='red', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X[1,]',ylab = 'X[,1]')
par(new=TRUE)
plot(f2, col='blue', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X[1,]',ylab = 'X[,1]')
par(new=TRUE)
plot(f3, col='magenta', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X[1,]',ylab = 'X[,1]')

## Creating the Bayes' Classifier boundary, graphing contours on the three distributions generated above ###
xmin = min(X[,1])
xmax = max(X[,1])
ymin = min(X[,2])
ymax = max(X[,2])
Xgx = vector()
Xgy = vector()
for (i in 1:50){
  Xgx[i] = xmin + i*(xmax - xmin)/50
  Xgy[i] = ymin + i*(ymax - ymin)/50
}

f1def = (1/(sqrt(pi^2)*det(cov1)))*exp(-1*t(x-mu1)%*%solve(cov1)%*%(x-mu1))
f2def = (1/(sqrt(pi^2)*det(cov2)))*exp(-1*t(x-mu2)%*%solve(cov2)%*%(x-mu2))
f3def = (1/(sqrt(pi^2)*det(cov3)))*exp(-1*t(x-mu3)%*%solve(cov3)%*%(x-mu3))

x = vector()
g1 = matrix(nrow = 50, ncol = 50)
g2 = matrix(nrow = 50, ncol = 50)
g3 = matrix(nrow = 50, ncol = 50)
for (i in 1:50){
  for (j in (1:50)){
    x = c(Xgx[i], Xgy[j])
    g1[i,j] = pi1*(1/(sqrt(pi^2)*det(cov1)))*exp(-1*t(x-mu1)%*%solve(cov1)%*%(x-mu1)) - max(pi2*(1/(sqrt(pi^2)*det(cov2)))*exp(-1*t(x-mu2)%*%solve(cov2)%*%(x-mu2)),pi3*(1/(sqrt(pi^2)*det(cov3)))*exp(-1*t(x-mu3)%*%solve(cov3)%*%(x-mu3)))
    g2[i,j] = pi2*(1/(sqrt(pi^2)*det(cov2)))*exp(-1*t(x-mu2)%*%solve(cov2)%*%(x-mu2)) - max(pi1*(1/(sqrt(pi^2)*det(cov1)))*exp(-1*t(x-mu1)%*%solve(cov1)%*%(x-mu1)),pi3*(1/(sqrt(pi^2)*det(cov3)))*exp(-1*t(x-mu3)%*%solve(cov3)%*%(x-mu3)))
    g3[i,j] = pi3*(1/(sqrt(pi^2)*det(cov3)))*exp(-1*t(x-mu3)%*%solve(cov3)%*%(x-mu3)) - max(pi1*(1/(sqrt(pi^2)*det(cov1)))*exp(-1*t(x-mu1)%*%solve(cov1)%*%(x-mu1)),pi2*(1/(sqrt(pi^2)*det(cov2)))*exp(-1*t(x-mu2)%*%solve(cov2)%*%(x-mu2)))
  }
}

plot(f1, col='red', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X[1,]',ylab = 'X[,1]')
par(new = TRUE)
plot(f2, col='blue', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X[1,]',ylab = 'X[,1]')
par(new = TRUE)
plot(f3, col='magenta', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X[1,]',ylab = 'X[,1]')
par(new = TRUE)
contour(Xgx, Xgy, g1,xlim = c(-2, 10), ylim = c(-8,10), levels = 0)
par(new = TRUE)
contour(Xgx, Xgy, g2,xlim = c(-2, 10), ylim = c(-8,10), levels = 0)
par(new = TRUE)
contour(Xgx, Xgy, g3,xlim = c(-2, 10), ylim = c(-8,10), levels = 0)

### Linear Discriminant Analysis boundary ###

newcol = vector()
for (i in 1:20){
  newcol[i] = 1
}
for (i in 21:50){
  newcol[i] = 2
}
for (i in 51:100){
  newcol[i] = 3
}
X = cbind(X, newcol)

X = as.data.frame(X)
colnames(X) = c("x", "y", "ClassLabel")


TheClass = factor(X$ClassLabel[1:100])

X = as.matrix(X)
classifier = lda(X[1:100,1:2], TheClass)

tst_grid <- expand.grid(x = Xgx, y= Xgy)
guess = predict(classifier,tst_grid)
predict_class = guess$class
proba_class = guess$posterior

g1lda = vector()
g2lda = vector()
g3lda = vector()
for (i in 1:2500){
  g1lda[i] = proba_class[i,1] - max(proba_class[i,2], proba_class[i,3])
  g2lda[i] = proba_class[i,2] - max(proba_class[i,1], proba_class[i,3])
  g3lda[i] = proba_class[i,3] - max(proba_class[i,1], proba_class[i,2])
}
g1lda_m = matrix(as.matrix(g1lda), nrow = 50, ncol = 50)
g2lda_m = matrix(as.matrix(g2lda), nrow = 50, ncol = 50)
g3lda_m = matrix(as.matrix(g3lda), nrow = 50, ncol = 50)


plot(X,xlim = c(0.5,4.0), ylim = c(-3.5,2.5))
par(new=TRUE)
contour(Xgx, Xgy, g1lda_m, xlim = c(-2, 10), ylim = c(-8,10),levels = 0)
par(new=TRUE)
contour(Xgx, Xgy, g2lda_m, xlim = c(-2, 10), ylim = c(-8,10),levels = 0)
par(new=TRUE)
contour(Xgx, Xgy, g3lda_m, xlim = c(-2, 10), ylim = c(-8,10),levels = 0)


##Using Bayes' classifier and Quadratic discriminant analysis to classify data from Gaussian distributions with the same covariance matrix##
grad_cov1 = cbind(c(2.5, 0), c(0, 2.5))
grad_cov2 = cbind(c(2.5, 0), c(0, 2.5))
grad_cov3 = cbind(c(2.5, 0), c(0, 2.5))
grad_pi1 = 0.25
grad_pi2 = 0.4
grad_pi3 = 0.35

grad_f1 = mvrnorm(25, mu1, grad_cov1)
grad_f2 = mvrnorm(40, mu2, grad_cov2)
grad_f3 = mvrnorm(35, mu3, grad_cov3)

X_grad = rbind(grad_f1, grad_f2, grad_f3)
xmin = min(X_grad[,1])
xmax = max(X_grad[,1])
ymin = min(X_grad[,2])
ymax = max(X_grad[,2])
Xgx = vector()
Xgy = vector()
for (i in 1:50){
  Xgx[i] = xmin + i*(xmax - xmin)/50
  Xgy[i] = ymin + i*(ymax - ymin)/50
}

f1_grad_def = (1/(sqrt(pi^2)*det(grad_cov1)))*exp(-1*t(x-mu1)%*%solve(grad_cov1)%*%(x-mu1))
f2_grad_def = (1/(sqrt(pi^2)*det(grad_cov2)))*exp(-1*t(x-mu2)%*%solve(grad_cov2)%*%(x-mu2))
f3_grad_def = (1/(sqrt(pi^2)*det(grad_cov3)))*exp(-1*t(x-mu3)%*%solve(grad_cov3)%*%(x-mu3))

x = vector()
g1_grad = matrix(nrow = 50, ncol = 50)
g2_grad = matrix(nrow = 50, ncol = 50)
g3_grad = matrix(nrow = 50, ncol = 50)
for (i in 1:50){
  for (j in (1:50)){
    x = c(Xgx[i], Xgy[j])
    g1_grad[i,j] = grad_pi1*(1/(sqrt(pi^2)*det(grad_cov1)))*exp(-1*t(x-mu1)%*%solve(grad_cov1)%*%(x-mu1)) - max(grad_pi2*(1/(sqrt(pi^2)*det(grad_cov2)))*exp(-1*t(x-mu2)%*%solve(grad_cov2)%*%(x-mu2)),grad_pi3*(1/(sqrt(pi^2)*det(grad_cov3)))*exp(-1*t(x-mu3)%*%solve(grad_cov3)%*%(x-mu3)))
    g2_grad[i,j] = grad_pi2*(1/(sqrt(pi^2)*det(grad_cov2)))*exp(-1*t(x-mu2)%*%solve(grad_cov2)%*%(x-mu2)) - max(grad_pi1*(1/(sqrt(pi^2)*det(grad_cov1)))*exp(-1*t(x-mu1)%*%solve(grad_cov1)%*%(x-mu1)),grad_pi3*(1/(sqrt(pi^2)*det(grad_cov3)))*exp(-1*t(x-mu3)%*%solve(grad_cov3)%*%(x-mu3)))
    g3_grad[i,j] = grad_pi3*(1/(sqrt(pi^2)*det(grad_cov3)))*exp(-1*t(x-mu3)%*%solve(grad_cov3)%*%(x-mu3)) - max(grad_pi1*(1/(sqrt(pi^2)*det(grad_cov1)))*exp(-1*t(x-mu1)%*%solve(grad_cov1)%*%(x-mu1)),grad_pi2*(1/(sqrt(pi^2)*det(grad_cov2)))*exp(-1*t(x-mu2)%*%solve(grad_cov2)%*%(x-mu2)))
  }
}


newcol = vector()
for (i in 1:25){
  newcol[i] = 1
}
for (i in 26:65){
  newcol[i] = 2
}
for (i in 66:100){
  newcol[i] = 3
}
X_grad = cbind(X_grad, newcol)

X_grad = as.data.frame(X_grad)
colnames(X_grad) = c("x", "y", "ClassLabel")


TheClass = factor(X_grad$ClassLabel[1:100])

X = as.matrix(X_grad)
grad_quad_classifier = qda(X[1:100,1:2], TheClass)

tst_grid <- expand.grid(x = Xgx, y= Xgy)
grad_quad_guess = predict(grad_quad_classifier,tst_grid)
grad_quad_predict_class = grad_quad_guess$class
grad_quad_proba_class = grad_quad_guess$posterior

grad_g1qda = vector()
grad_g2qda = vector()
grad_g3qda = vector()
for (i in 1:2500){
  grad_g1qda[i] = grad_quad_proba_class[i,1] - max(grad_quad_proba_class[i,2], grad_quad_proba_class[i,3])
  grad_g2qda[i] = grad_quad_proba_class[i,2] - max(grad_quad_proba_class[i,1], grad_quad_proba_class[i,3])
  grad_g3qda[i] = grad_quad_proba_class[i,3] - max(grad_quad_proba_class[i,1], grad_quad_proba_class[i,2])
}
grad_g1qda_m = matrix(as.matrix(grad_g1qda), nrow = 50, ncol = 50)
grad_g2qda_m = matrix(as.matrix(grad_g2qda), nrow = 50, ncol = 50)
grad_g3qda_m = matrix(as.matrix(grad_g3qda), nrow = 50, ncol = 50)

plot(grad_f1, col='red', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X',ylab = 'Y')
par(new = TRUE)
plot(grad_f2, col='blue', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X',ylab = 'Y')
par(new = TRUE)
plot(grad_f3, col='magenta', xlim = c(-2, 10), ylim = c(-8,10),xlab = 'X',ylab = 'Y')
par(new = TRUE)
contour(Xgx, Xgy, g1_grad, xlim = c(-2, 10), ylim = c(-8,10), levels = 0)
par(new = TRUE)
contour(Xgx, Xgy, g2_grad, xlim = c(-2, 10), ylim = c(-8,10),levels = 0)
par(new = TRUE)
contour(Xgx, Xgy, g3_grad,xlim = c(-2, 10), ylim = c(-8,10),levels = 0)
par(new=TRUE)
contour(Xgx, Xgy, grad_g1qda_m, xlim = c(-2, 10), ylim = c(-8,10), levels = 0)
par(new=TRUE)
contour(Xgx, Xgy, grad_g2qda_m, xlim = c(-2, 10), ylim = c(-8,10), levels = 0,)
par(new=TRUE)
contour(Xgx, Xgy, grad_g3qda_m, xlim = c(-2, 10), ylim = c(-8,10), levels = 0,)



