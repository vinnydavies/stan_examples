######### Setup #########

# Following is initially setup. Should just need to install rstan after that
# I only managed to install once I updated R

# need to also install Rtools. Follow the link which comes up when you try to install things

# remove.packages("rstan")
# if (file.exists(".RData")) file.remove(".RData")
# 
# install.packages("rstan",repos = "https://cloud.r-project.org/", dependencies = TRUE)
# 
# pkgbuild::has_build_tools(debug = TRUE)
# needs to return true or C++ compiler isnt set up properly
# https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')

#### end ####

######### Linear Model ###########


##### set up model

model <- "
  
  data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
  }
  
  parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
  }
  
  model {
    y ~ normal(alpha + beta * x, sigma);
  }
  
"
##### generate data

par(mfrow=c(1,1))
x <- rnorm(40, 10, 5)
noise <- rnorm(40,0,1)
y <- x*0.5 + noise
dat <- list( x= x, y = y, N = length(x))
plot(x, y, pch=19, cex=2)

##### sample from model

fit <- stan(model_code = model, model_name = "example",
            data = dat, iter = 2012, chains = 3, sample_file = 'norm.csv',
            verbose = TRUE)
print(fit)

###### extract samples
e <- extract(fit, permuted = TRUE) # return a list of arrays
hist(e$beta)

###### Diagnostics

traceplot(fit)
pairs(fit)

#### end ####

######### Linear Model with Priors ########


##### set up model

model <- "

  data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
  }
  
  parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
  }
  
  model {
    y ~ normal(alpha + beta * x, sigma);
    alpha ~ cauchy(0,10);
    beta ~ cauchy(0.2,5);
  }
  
"
##### generate data

par(mfrow=c(1,1))
x <- rnorm(40, 10, 5)
noise <- rnorm(40,0,1)
y <- x*0.5 + noise
dat <- list( x= x, y = y, N = length(x))
plot(x, y, pch=19, cex=2)

##### sample from model

fit <- stan(model_code = model, model_name = "example",
            data = dat, iter = 2012, chains = 3, sample_file = 'norm.csv',
            verbose = TRUE)
print(fit)

###### extract samples
e <- extract(fit, permuted = TRUE) # return a list of arrays
hist(e$beta)

###### Diagnostics

traceplot(fit)
pairs(fit)

#### end ####

######### 8 Schools Example #######


### specify the model

schools_model <- "

  data {
    int<lower=0> J;          // number of schools 
    real y[J];               // estimated treatment effects
    real<lower=0> sigma[J];  // s.e. of effect estimates 
  }
  parameters {
    real mu; 
    real<lower=0> tau;
    vector[J] eta;
  }
  transformed parameters {
    vector[J] theta;
    theta = mu + tau * eta;
  }
  model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
  }
    
"

###### data 

schools_data <- list(
  J = 8,
  y = c(28,  8, -3,  7, -1,  1, 18, 12),
  sigma = c(15, 10, 16, 11,  9, 11, 10, 18)
)

##### sample from model

fit <- stan(model_code = schools_model, model_name = "example",
            data = schools_data, iter = 2012, chains = 3, sample_file = 'norm.csv',
            verbose = TRUE)
print(fit)

###### Diagnostics

traceplot(fit, pars=c('mu', 'tau'), nrow=2)
plot(fit,pars=c('theta'))

#### end ####

######### Gaussian Process Example #######

N = 100
x <- sort(runif(N, 0, 10))
beta <- c(0,0.1,0.2,-0.02, 0.00001)
sigma <- 0.1
y_true <- beta[1] + beta[2] * x + beta[3] * x^2 + beta[4] * x^3 + beta[5] * x^4
y <- y_true + rnorm(100,0,sigma)
plot(x,y)
lines(x,y_true)
gp_data = list(N=N, y=y, x=x)

##### Convert Data to lists

N_predict <- N
x_predict <- x
y_predict <- y_true

stan_rdump(c("N", "x", "y",
             "N_predict", "x_predict", "y_predict",
             "sample_idx"), file="gp.data.R")

data <- read_rdump('gp.data.R')

##### Model

gp_model <- "
  functions {
    vector gp_pred_rng(real[] x2,
                       vector y1, real[] x1,
                       real alpha, real rho, real sigma, real delta) {
      int N1 = rows(y1);
      int N2 = size(x2);
      vector[N2] f2;
      {
        matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho)
                           + diag_matrix(rep_vector(square(sigma), N1));
        matrix[N1, N1] L_K = cholesky_decompose(K);
  
        vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
        vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
        matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho);
        vector[N2] f2_mu = (k_x1_x2' * K_div_y1);
        matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
        matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, alpha, rho) - v_pred' * v_pred
                                + diag_matrix(rep_vector(delta, N2));
        f2 = multi_normal_rng(f2_mu, cov_f2);
      }
      return f2;
    }
  }
  
  data {
    int<lower=1> N;
    real x[N];
    vector[N] y;
  
    int<lower=1> N_predict;
    real x_predict[N_predict];
  }
  
  parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
    real<lower=0> sigma;
  }
  
  model {
    matrix[N, N] cov =   cov_exp_quad(x, alpha, rho)
                       + diag_matrix(rep_vector(square(sigma), N));
    matrix[N, N] L_cov = cholesky_decompose(cov);
  
    rho ~ normal(0, 20.0 / 3);
    alpha ~ normal(0, 2);
    sigma ~ normal(0, 1);
    y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
  }
  
  generated quantities {
    vector[N_predict] f_predict = gp_pred_rng(x_predict, y, x, alpha, rho, sigma, 1e-10);
    vector[N_predict] y_predict;
    for (n in 1:N_predict)
      y_predict[n] = normal_rng(f_predict[n], sigma);
  }
"

fit <- stan(model_code=gp_model, data=data, seed=5838298)
print(fit)

###### Plot Results

# plot leaves a lot to be desired, sorry!
f_out <- extract(fit)$f_predict

plot(x, y_true, type="l", lwd=2, xlab="x", ylab="y",
     xlim=c(0, 10), ylim=c(0, 4))
for(i in 1:ncol(f_out)){
  lines(x, f_out[i,], type='l', col='red')
}
lines(x, y_true, lwd=4)
points(x, y, col="white", pch=16, cex=1.5)
points(x, y, col="black", pch=16, cex=1.3)


#### end ####