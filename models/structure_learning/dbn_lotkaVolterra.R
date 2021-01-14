library(dbnR)
library(caret)


inferGC_dbn_K1 <- function(df, method="psoho") {
  GCstruct <- NA
  p <- ncol(df)
  if (method == "psoho") {
    net <- learn_dbn_struc(df, 2, method="psoho")
    GCstruct <- amat(net)[1:p, (p + 1):(2 * p)] 
  } else {
    net <- learn_dbn_struc(df, 2, method="dmmhc")
    GCstruct <- amat(net)[(p + 1):(2 * p), 1:p]
  }
  return(t(GCstruct))
}

acc_score <- function(A1, A2) {
  a1 <- A1[row(A1)!=col(A1)]
  a2 <- A2[row(A2)!=col(A2)]
  return(sum(a1 == a2) / (length(a2)))
}

balacc_score <- function(A1, A2) {
  cm <- confusionMatrix(data=as.factor(as.vector(A1[row(A1)!=col(A1)])), reference=as.factor(as.vector(A2[row(A2)!=col(A2)])))
  return((unname(cm$byClass["Sensitivity"]) + unname(cm$byClass["Specificity"])) / 2)
}


method <- "dmmhc"

accs <- numeric(5)
balaccs <- numeric(5)

df <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_data_r_0.csv", 
               header = FALSE, sep = " ")
A <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_struct_r_0.csv", 
              header = FALSE, sep = " ")
A <- as.matrix(A)
A_hat <- inferGC_dbn_K1(df, method = method)
A_hat
(accs[1] <- acc_score(A_hat, A)) 
(balaccs[1] <- balacc_score(A_hat, A))

df <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_data_r_1.csv", 
               header = FALSE, sep = " ")
A <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_struct_r_1.csv", 
              header = FALSE, sep = " ")
A <- as.matrix(A)
A_hat <- inferGC_dbn_K1(df, method = method)
A_hat
(accs[2] <- acc_score(A_hat, A))
(balaccs[2] <- balacc_score(A_hat, A))

df <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_data_r_2.csv", 
               header = FALSE, sep = " ")
A <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_struct_r_2.csv", 
              header = FALSE, sep = " ")
A <- as.matrix(A)
A_hat <- inferGC_dbn_K1(df, method = method)
A_hat
(accs[3] <- acc_score(A_hat, A)) 
(balaccs[3] <- balacc_score(A_hat, A))

df <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_data_r_3.csv", 
               header = FALSE, sep = " ")
A <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_struct_r_3.csv", 
              header = FALSE, sep = " ")
A <- as.matrix(A)
A_hat <- inferGC_dbn_K1(df, method = method)
A_hat
(accs[4] <- acc_score(A_hat, A)) 
(balaccs[4] <- balacc_score(A_hat, A))

df <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_data_r_4.csv", 
               header = FALSE, sep = " ")
A <- read.csv("/mnt/data_disk/Projects/SENGC/datasets/experiment_data/lotka-volterra/lotka-volterra_struct_r_4.csv", 
              header = FALSE, sep = " ")
A <- as.matrix(A)
A_hat <- inferGC_dbn_K1(df, method = method)
A_hat
(accs[5] <- acc_score(A_hat, A)) 
(balaccs[5] <- balacc_score(A_hat, A))

mean(accs)
sd(accs)
mean(balaccs)
sd(balaccs)