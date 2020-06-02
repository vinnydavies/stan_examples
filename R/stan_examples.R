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

##### Simulate some data

N = 100
x <- sort(runif(N, 0, 10))
beta <- c(0,0.1,0.2,-0.02, 0.00001)
sigma <- 0.1
y_true <- beta[1] + beta[2] * x + beta[3] * x^2 + beta[4] * x^3 + beta[5] * x^4
y <- y_true + rnorm(100,0,sigma)
plot(x,y)
lines(x,y_true)
gp_data = list(N=N, y=y, x=x)

##### Model

gp_model <- "
 data {
  int<lower=1> N;
  real x[N];
  vector[N] y;
}
transformed data {
  vector[N] mu = rep_vector(0, N);
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
}
model {
  matrix[N, N] L_K;
  matrix[N, N] K = cov_exp_quad(x, alpha, rho);
  real sq_sigma = square(sigma);

  // diagonal elements
  for (n in 1:N)
    K[n, n] = K[n, n] + sq_sigma;

  L_K = cholesky_decompose(K);

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ std_normal();

  y ~ multi_normal_cholesky(mu, L_K);
}
"

##### sample from model

fit <- stan(model_code = gp_model, model_name = "example",
            data = gp_data, iter = 2012, chains = 3, sample_file = 'norm.csv',
            verbose = TRUE)
print(fit)

f_total <- extract(fit)$f[1,]
y_total <- extract(fit)$y[1,]

true_realization <- data.frame(f_total, x_total)

##### Optimisation

opt_fit <- optimizing(gp_model, data=gp_data, seed=5838298, hessian=FALSE)

###### Draw Sample from the Posterior of Gaussian Process



#### end ####