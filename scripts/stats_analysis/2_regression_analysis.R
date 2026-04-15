##### Imports #####
library(fastDummies)
library(tidyverse)
library(glmnet)

##### A) Preprocessing #####

## Characterization DF
df_characterization <- read.csv(
  "0_characterization.csv",
  sep = ";", dec = ".", header = TRUE,
  na = c("", "na", "NA")
)

df_characterization <- df_characterization %>%
  mutate(
    iaa_metric = case_match(iaa_metric, "nr" ~ NA, "cohen" ~ "COHEN", .default = iaa_metric),
    theoretical_framework = case_match(theoretical_framework, "nr" ~ NA, .default = theoretical_framework)
  )

## Results DF
df_results <- read.csv(
  "2_test_results.csv",
  sep = ";", dec = ".", header = TRUE,
  na = c("", "na", "NA")
)

colnames(df_results)[4] <- "year"

## Errors DF
df <- read.csv(
  file      = "3_combined_errors_full.csv",
  sep       = ";",
  dec       = ".",
  header    = TRUE,
  na = c("","na","NA"),
  stringsAsFactors = TRUE
)
df$is_cgec <- as.factor(df$is_cgec)
df$article_id <- as.character(df$article_id)
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df$headline <- as.character(df$headline)
df$content <- as.character(df$content)
df$paragraph_text <- as.character(df$paragraph_text)
df$sentence_text <- as.character(df$sentence_text)
df <- df %>%
  mutate(
    predicted_label = as.character(predicted_label),
    true_label      = as.character(true_label),
    
    error_type = case_when(
      predicted_label == true_label ~ "correct",
      
      true_label %in% c("argumentative", "premise", "claim") &
        predicted_label %in% c("none", "non-argumentative") ~ "FN",
      
      true_label %in% c("non-argumentative", "none") &
        predicted_label %in% c("argumentative", "premise", "claim") ~ "FP",
      
      true_label %in% c("argumentative", "premise", "claim") &
        predicted_label %in% c("argumentative", "premise", "claim") ~ "LE",
      
      TRUE ~ "other"
    )
  )
df$predicted_label <- as.factor(df$predicted_label)
df$true_label <- as.factor(df$true_label)
df$error_type <- as.factor(df$error_type)
df_errors_complete <- df

head(df_errors_complete)
summary(df_errors_complete)
str(df_errors_complete)
##### Helpers #####

minmax <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (diff(rng) == 0) return(rep(0, length(x)))
  (x - rng[1]) / diff(rng)
}

characteristic_vars_continuous <- c(
  "total_instances", "total_tokens", "unique_tokens",
  "avg_instance_length", "median_instance_length", "vocabulary_overlap",
  "domain_similarity_tfidf", "size_ratio_log",
  "binary_argumentative_pct", "binary_balance_ratio",
  "component_premise_pct", "component_claim_pct",
  "binary_nonargumentative_pct",
  "linguistic_distance_beto", "domain_distance_beto",
  "linguistic_distance_roberta", "domain_distance_roberta",
  "linguistic_distance_xlmr", "domain_distance_xlmr"
)

characteristic_vars_ordinal <- c(
  "topical_scope", "domain_specificity",
  "annotation_scheme_compatibility",
  "granularity_match_with_target",
  "dialectal_distance"
)

characteristic_vars_categorical <- c(
  "register", "text_genre", "primary_domain",
  "granularity_level", "task_framing", "theoretical_framework"
)

build_lasso_matrix <- function(df) {
  
  ## Scale continuous
  df <- df %>%
    mutate(across(all_of(characteristic_vars_continuous),
                  minmax, .names = "mm_{.col}"))
  
  ## Dummy encode categorical
  df <- dummy_cols(
    df,
    select_columns = characteristic_vars_categorical,
    remove_first_dummy = TRUE,
    remove_selected_columns = TRUE
  )
  
  ## Predictor columns
  mm_cols <- paste0("mm_", characteristic_vars_continuous)
  
  #exclude_vars <- c("recall_macro", "accuracy", "precision_macro")
  exclude_vars <- c(
    "prob_premise",
    "prob_claim",
    "prob_argumentative",
    "confidence",
    "prediction",
    "gold_label",
    "error_type",
    "article_id",
    "task",
    "model"
  )
  
  dummy_cols_x <- names(df)[
    !(names(df) %in% c(
      "dataset", "year", "f1_macro"
    )) &
      !(names(df) %in% characteristic_vars_continuous) &
      !(names(df) %in% characteristic_vars_ordinal) &
      !(names(df) %in% exclude_vars)
  ]
  
  predictors <- c(mm_cols, characteristic_vars_ordinal, dummy_cols_x)
  
  list(df = df, predictor_cols = predictors)
}

##### B) Lasso — F1-score #####

## 1. Merge
df_reg <- merge(df_results, df_characterization,
                by = c("dataset", "year"))
## 3. Build matrix
lasso_b <- build_lasso_matrix(df_reg)

df_b <- lasso_b$df %>%
  select(f1_macro, all_of(lasso_b$predictor_cols))

## 4. Remove rows with NA in target only
df_b <- df_b[!is.na(df_b$f1_macro), ]

## 5. Replace remaining NA predictors with 0 (SAFE FOR LASSO)
df_b[is.na(df_b)] <- 0

## 6. Build X and y
X_b <- as.matrix(df_b[, lasso_b$predictor_cols])
y_b <- df_b$f1_macro

## DEBUG CHECK
cat("Rows X:", nrow(X_b), "\n")
cat("Length y:", length(y_b), "\n")

##### Run LASSO #####

set.seed(42)
lasso_cv_b <- cv.glmnet(
  x = X_b,
  y = y_b,
  alpha = 0.5,
  nfolds = min(10, nrow(X_b)),   # SAFE for small N
  standardize = TRUE
)

plot(lasso_cv_b)
title("Lasso CV (F1-score)", line = 3)

##### Extract coefficients #####
##### D) Lasso — Error type (multinomial) #####

## Merge with dataset characteristics
colnames(df_errors_complete)[9] <- "article_year"
colnames(df_errors_complete)[4] <- "year"
df_errors_complete$dataset <- as.factor(df_errors_complete$dataset)
df_characterization$dataset
df_errors_complete$year <- as.factor(df_errors_complete$year)
df_characterization$year

df_errors_type <- merge(
  df_errors_complete,
  df_characterization,
  by = c("dataset", "year")
)

## Build matrix
lasso_d <- build_lasso_matrix(df_errors_type)

df_d <- lasso_d$df %>%
  select(error_type, all_of(lasso_d$predictor_cols))

## Remove missing target
df_d <- df_d[!is.na(df_d$error_type), ]

## Replace NA predictors with 0
df_d[is.na(df_d)] <- 0

## X and y
X_d <- as.matrix(df_d[, lasso_d$predictor_cols])
y_d <- df_d$error_type

## Debug
cat("Rows X:", nrow(X_d), "\n")
cat("Length y:", length(y_d), "\n")
cat("Class distribution:\n")
print(table(y_d))

## Run multinomial LASSO
set.seed(42)
lasso_cv_d <- cv.glmnet(
  x = X_d,
  y = y_d,
  family = "multinomial",
  alpha = 0.5,               # Elastic Net (recommended)
  nfolds = min(10, nrow(X_d)),
  standardize = TRUE,
  type.measure = "deviance"
)

plot(lasso_cv_d)
title("Lasso CV (Error type)", line = 3)

cat("\n=== Lasso — Error type ===\n")
cat("lambda.min:", lasso_cv_d$lambda.min, "\n")
cat("lambda.1se:", lasso_cv_d$lambda.1se, "\n")

## Extract coefficients
extract_nonzero_multi <- function(cv_model, s) {
  cf <- coef(cv_model, s = s)
  
  lapply(names(cf), function(class) {
    df <- as.data.frame(as.matrix(cf[[class]]))
    df$predictor <- rownames(df)
    colnames(df)[1] <- "coefficient"
    
    df %>%
      filter(predictor != "(Intercept)", coefficient != 0) %>%
      arrange(desc(abs(coefficient))) %>%
      mutate(class = class)
  }) %>%
    bind_rows()
}

cat("\n--- Non-zero at lambda.1se ---\n")
print(extract_nonzero_multi(lasso_cv_d, "lambda.1se"))


extract_nonzero <- function(cv_model, s) {
  cf <- coef(cv_model, s = s)
  df <- as.data.frame(as.matrix(cf))
  df$predictor <- rownames(df)
  colnames(df)[1] <- "coefficient"
  
  df %>%
    filter(predictor != "(Intercept)", coefficient != 0) %>%
    arrange(desc(abs(coefficient)))
}
