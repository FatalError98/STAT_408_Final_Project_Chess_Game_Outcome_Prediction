library(tidyverse)

# setting a seed to have the same result when run again.
set.seed(19950820)

# Reading the chess game dataset.
chess_game <- read_csv('Data/chess_game.csv')
head(chess_game)
# Making sure all variable are in the correct data type.
chess_game <- chess_game %>%
  mutate(
    game_type = as.factor(game_type),
    opening = as.factor(opening))

# shoing some summary for the data
summary(chess_game$elo_diff)
table(chess_game$game_type)

# Building a logitical model binomial family.
model <- glm(
  whiteWin ~ game_type + elo_diff + pawn_count_diff + knight_count_diff + bishop_count_diff + rook_count_diff + queen_count_diff + captured_score_diff + captured_diff + check_diff + opening, # added opening in end just to show the rest of estimate in summary without the need to increase the max print.
  data = chess_game,
  family = binomial()
)

summary(model)

## Checking for multicolinearity & Influential Observation

library(regclass)
# VIF(model)  ERROR: there are aliased coefficients in the model
new_model <- glm(
  whiteWin ~ . - captured_diff,
  data = chess_game,
  family = binomial()
)
VIF(new_model)

## Checking for Influential Observation
rstud <- rstudent(model)
n <- nrow(chess_game)
k <- 12
alpha <- 0.0001
crit <- qt(1-alpha / 2, n-1-k)
sort(which(abs(rstud) > crit), T)

## Choosing the best model using Elastic Net K-fold cross validation

library(glmnet)

X <- model.matrix(whiteWin ~ ., data = chess_game)[,-1]
y <- chess_game$whiteWin

K <- 5
foldid <- sample(rep(1:K, length.out = nrow(chess_game)))

alpha <- seq(0,1,0.05)
alpha_cv <- NULL

for (a in alpha) {
  lam_a <- cv.glmnet(x = X, y = y, family = "binomial", foldid = foldid, alpha = a)
  lambda_min <- lam_a$lambda.min
  alpha_cv <- c(alpha_cv,lam_a$cvm[lam_a$lambda == lambda_min])
}

best_alpha <- alpha[which.min(alpha_cv)]
best_alpha

bestModel <- glm(
  whiteWin ~ game_type + opening + elo_diff + queen_count_diff + bishop_count_diff + captured_score_diff + captured_diff + check_diff , 
  data = chess_game,
  family = binomial()
)

summary(bestModel)

# Just to know the base for the interpretation 
levels(chess_game$opening)[1]
levels(chess_game$game_type)[1]

newData <- data.frame(
  game_type = factor("Classical", levels = levels(chess_game$game_type)),
  elo_diff = 8,
  pawn_count_diff = 1,
  knight_count_diff = 0,
  bishop_count_diff = 0,
  rook_count_diff = -2,
  queen_count_diff = 1,
  captured_score_diff = 0,
  captured_diff = 0,
  check_diff = 1,
  opening = factor("Italian Game", levels = levels(chess_game$opening))
)


newX <- model.matrix(~ ., data = newData)[, -1]


pred_new <- predict(
  ex,
  newx = newX,
  s = "lambda.min",
  type = "response"
)

pred_new