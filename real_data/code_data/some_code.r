Data_Ohio$X <- NULL
reward <- function(s){
  R <- rep(0, length(s))
  R[s<=80] <- -(80-s[s<=80])^2/30
  R[s>=140] <- -(s[s>=140]-140)^(1.35)/30
}
R <- reward(Data_Ohio$glucose)
A <- Data_Ohio$A
S <- cbind(Data_Ohio$glucose, Data_Ohio$meal, Data_Ohio$exercise)
library(grf)
R1 <- R[5:1100]
A1 <- A[4:1099]
S1 <- matrix(0, 1097, 15)
for (j in 1:1096){
  S1[j,] <- c(S[j:(j+3),], A[j:(j+2)])
}

R2 <- R[1100+5:1100]
A2 <- A[1100+4:1099]
S2 <- matrix(0, 1097, 15)
for (j in 1:1096){
  S2[j,] <- c(S[1100+j:(j+3),], A[1100+j:(j+2)])
}

R3 <- R[2200+5:1100]
A3 <- A[2200+4:1099]
S3 <- matrix(0, 1097, 15)
for (j in 1:1096){
  S3[j,] <- c(S[2200+j:(j+3),], A[2200+j:(j+2)])
}

FQI <- function(R, A, S, gamma, maxiter=50){
  n <- length(A)
  Q0 <- regression_forest(cbind(A,S[1:n,]), R, tune.parameters = "all")
  for (i in 1:maxiter){
    Y <- matrix(0, n, 5)
    for (j in 0:4){
      Y[,j+1] <- predict(Q0, newdata = cbind(j,S[2:(n+1),]))$predictions
    }
    Y <- apply(Y, 1, max)
    Q1 <- regression_forest(cbind(A,S[1:n,]), R+gamma*Y, tune.parameters = "all")
    err <- predict(Q0, newdata = cbind(A,S[1:n,]))$predictions - 
      predict(Q1, newdata = cbind(A,S[1:n,]))$predictions
    if (mean(abs(err)) < 1e-6){
      print(i)
      return(Q1)
      break
    }
    Q0 <- Q1
  }
  print(maxiter)
  return(Q1)
}

set.seed(12345)
ind <- sample(1:L, 1000)
plot(1:1000, Q1$predictions[ind], col="red", ylim=c(-80, -30), xlab="", ylab = "optimal Q-function", pch=19)
points(1:1000, Q2$predictions[ind], col="blue", ylim=c(-80, -30), pch=19)
points(1:1000, Q3$predictions[ind], col="green", ylim=c(-80, -30), pch=19)
legend("bottom", legend=c("Patient 1", "Patient 2", "Patient 3"), pch=c(19,19,19), col=c("red", "blue", "green"))


