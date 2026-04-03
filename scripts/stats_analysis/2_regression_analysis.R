# TODO: Add some plots

##### Imports ##### 
library(fastDummies)
library(tidyverse)
library(glmnet)
library(lme4)
library(lmerTest)
library(broom)
library(broom.mixed)
library(car)
library(buildmer)
library(performance)
library(mclogit)
library(nnet)
library(glmmTMB)
# library(effects)   # for plotting marginal effects

##### A) Preprocessing ##### 
### 01. CHARACTERIZATION DF
df <- read.csv(
  file      = "0_characterization.csv",
  sep       = ";",
  dec       = ".",
  header    = TRUE,
  na = c("","na","NA"),
  stringsAsFactors = TRUE
)
df <- df %>%
  mutate(
    iaa_metric = case_match(iaa_metric,
                            "nr"    ~ "NR",
                            "cohen" ~ "COHEN",
                            .default = iaa_metric
    ),
    iaa_value = case_match(iaa_value,
                            "nr"    ~ "NR",
                            .default = iaa_value
    ),
    theoretical_framework = case_match(theoretical_framework,
                           "nr"    ~ "NR",
                           .default = theoretical_framework
  )
)
df$iaa_metric <- as.factor(df$iaa_metric)
df$iaa_value <- as.factor(df$iaa_value)
df$theoretical_framework <- as.factor(df$theoretical_framework)
df_characterization <- df

head(df_characterization)
summary(df_characterization)
str(df_characterization)

## 02. TEST RESULTS DF
df <- read.csv(
  file      = "2_test_results.csv",
  sep       = ";",
  dec       = ".",
  header    = TRUE,
  na = c("","na","NA"),
  stringsAsFactors = TRUE
)
df$is_cgec <- as.factor(df$is_cgec)
df_results <- df

head(df_results)
summary(df_results)
str(df_results)

## 03. (FULL) TEST RESULTS DF
df <- read.csv(
  file      = "2_test_results_full.csv",
  sep       = ";",
  dec       = ".",
  header    = TRUE,
  na = c("","na","NA"),
  stringsAsFactors = TRUE
)
df$is_cgec <- as.factor(df$is_cgec)
df_results_full <- df

head(df_results_full)
summary(df_results_full)
str(df_results_full)

## 03. ANNOTATED ERRORS DF
df <- read.csv(
  file      = "1_errors_annotated.csv",
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
df$textual_evidence <- as.character(df$textual_evidence)
df_errors_annotated <- df

head(df_errors_annotated)
summary(df_errors_annotated)
str(df_errors_annotated)

## 04. ERRORS COMPLETE DF
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

# Unifies column names
colnames(df_errors_complete)[9] <- "article_year"
colnames(df_errors_complete)[4] <- "year"
colnames(df_errors_annotated)[9] <- "article_year"
colnames(df_errors_annotated)[4] <- "year"
colnames(df_results)[4] <- "year"
colnames(df_results_full)[4] <- "year"

##### B) Linear Regression (F1-Score) ##### 
##### B.1 Build experiment-level dataframe #####
df_reg <- merge(df_results, df_characterization, by = c("dataset", "year"))
df_reg$dataset <- paste(df_reg$dataset, df_reg$year, sep = "_")
df_reg$dataset <- as.factor(df_reg$dataset)

length(colnames(df_reg))
df_reg <- df_reg %>%
  select(
    # --- Outcome
    f1_macro, 
    accuracy,
    # --- Model and Task controls ---
    model,
    is_cgec,
    task,
    # --- Hyperparameter controls ---
    learning_rate,
    num_train_epochs,
    per_device_train_batch_size,
    warmup_ratio,
    weight_decay,
    num_train_epochs,
    warmup_ratio,
    # --- Dataset characteristics ---
    # Continuous
    total_instances,
    total_tokens,
    unique_tokens,
    avg_instance_length,
    median_instance_length,
    vocabulary_overlap,
    size_ratio_log,
    binary_argumentative_pct,
    binary_balance_ratio,
    component_premise_pct,
    component_claim_pct,
    binary_nonargumentative_pct,
    vocabulary_overlap,
    domain_similarity_tfidf,
    linguistic_distance_beto,
    domain_distance_beto,
    linguistic_distance_roberta,
    domain_distance_roberta,
    linguistic_distance_xlmr,
    domain_distance_xlmr,
    size_ratio_log,
    # Ordinal (already integer-encoded per manual)
    topical_scope,
    domain_specificity,
    annotation_scheme_compatibility,
    granularity_match_with_target,
    dialectal_distance,
    # Categorical (will be one-hot encoded below)
    register,
    text_genre,
    primary_domain,
    granularity_level,
    task_framing,
    theoretical_framework,
    # Grouping
    dataset
  )
length(colnames(df_reg))

##### B.2 Normalization #####
df_reg <- df_reg %>%
  mutate(
    c_f1_macro = f1_macro - mean(f1_macro, na.rm = TRUE),
    c_accuracy = accuracy - mean(accuracy, na.rm = TRUE)
    )
# --- Z-score standardization for continuous numeric predictors ---
# (makes regression coefficients directly comparable in magnitude)
continuous_vars <- c(
  "total_instances", "total_tokens", "unique_tokens", "avg_instance_length",
  "median_instance_length", "vocabulary_overlap", "size_ratio_log", 
  "binary_argumentative_pct", "binary_balance_ratio", "component_premise_pct",
  "component_claim_pct", "binary_nonargumentative_pct", "vocabulary_overlap",
  "domain_similarity_tfidf", "linguistic_distance_beto", "domain_distance_beto",
  "linguistic_distance_roberta", "domain_distance_roberta", "linguistic_distance_xlmr",
  "domain_distance_xlmr", "size_ratio_log", "learning_rate", "num_train_epochs",
  "warmup_ratio", "weight_decay", "warmup_ratio"
)
df_reg <- df_reg %>%
  mutate(across(all_of(continuous_vars),
                ~ scale(.x, center = TRUE, scale = TRUE)[,1],
                .names = "z_{.col}"))
# --- Ordinal variables: keep as integer (natural order preserved) ---
# topical_scope, domain_specificity, annotation_scheme_compatibility,
# granularity_match, dialectal_distance — already numeric, no transformation needed
# --- One-hot encoding for nominal categoricals ---
# (drop first level to avoid dummy trap)
df_reg <- dummy_cols(
  df_reg,
  select_columns = c("register", "text_genre", "primary_domain",
                     "granularity_level", "task_framing",
                     "theoretical_framework", "model", "task"),
  remove_first_dummy = TRUE,
  remove_selected_columns = FALSE  # keep originals for reference
)
# --- batch_size and num_epochs: treat as categorical dummies ---
# (discrete values, no natural continuous interpretation)
df_reg$per_device_train_batch_size  <- as.factor(df_reg$per_device_train_batch_size)
df_reg$num_train_epochs  <- as.factor(df_reg$num_train_epochs)
df_reg <- dummy_cols(
  df_reg,
  select_columns = c("per_device_train_batch_size", "num_train_epochs"),
  remove_first_dummy = TRUE,
  remove_selected_columns = FALSE
)

head(df_reg)
summary(df_reg)
str(df_reg)

##### B.3 Hypothesis testing #####
# The base model accounts for all experiment-level variance before testing
# any dataset characteristic. It includes:
#   - task and is_cgec as fixed experiment-level controls
#   - three continuous hyperparameters (z-score standardized) to partial out
#     variance due to Bayesian hyperparameter search results
#   - two discrete hyperparameters as fixed categorical controls
#   - (1 | dataset) to capture residual between-dataset grouping structure
#     not yet explained by characteristics (added per hypothesis below)
#   - (1 | model) to capture architecture-level clustering
base_lm_ml <- lmer(c_f1_macro ~
                     task +
                     is_cgec +
                     z_learning_rate +
                     z_weight_decay +
                     z_warmup_ratio +
                     per_device_train_batch_size +
                     num_train_epochs +
                     (1 | dataset) +
                     (1 | model),
                   data = df_reg,
                   REML = TRUE)
r2(base_lm_ml)
model_performance(base_lm_ml)
icc(base_lm_ml)
# The full model includes our hypothesis and compares it to the base model
h_full <- lmer(update(formula(base_lm_ml), . ~ . +
                        z_domain_similarity_tfidf + # H1
                        z_linguistic_distance_beto), # H2
               data = df_reg, REML = FALSE)
r2(h_full)
anova(base_lm_ml, h_full)
summary(h_full)

##### B.4 Lasso regression #####
# Important design decisions:
#   - glmnet requires a numeric matrix: all predictors must be numeric,
#     factors must be one-hot encoded, and all variables must be on the
#     same scale (min-max normalization used here, as glmnet is
#     scale-sensitive and min-max preserves interpretability as
#     [0,1]-bounded coefficients)
#   - The base model controls (task, is_cgec, hyperparameters) are included
#     as unpenalized predictors (penalty.factor = 0), so the Lasso only
#     penalizes the dataset characteristics — consistent with B.3
#   - Lambda is selected via 10-fold cross-validation (cv.glmnet)
#     with two criteria reported: lambda.min (lowest CV error) and
#     lambda.1se (most regularized model within 1 SE of minimum,
#     preferred for parsimony in small samples)

# Min-max normalization helper (applied only to dataset characteristics)
minmax <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (diff(rng) == 0) return(rep(0, length(x)))  # constant variable -> zero
  (x - rng[1]) / diff(rng)
}

# Dataset characteristics to include in Lasso
# (all characteristics from characterization matrix, excluding identifiers)
characteristic_vars_continuous <- c(
  "total_instances",
  "total_tokens",
  "unique_tokens",
  "avg_instance_length",
  "median_instance_length",
  "vocabulary_overlap",
  "domain_similarity_tfidf",
  "size_ratio_log",
  "binary_argumentative_pct",
  "binary_balance_ratio",
  "component_premise_pct",
  "component_claim_pct",
  "binary_nonargumentative_pct",
  "domain_similarity_tfidf",
  "linguistic_distance_beto",
  "domain_distance_beto",
  "linguistic_distance_roberta",
  "domain_distance_roberta",
  "linguistic_distance_xlmr",
  "domain_distance_xlmr"
)

characteristic_vars_ordinal <- c(
  "topical_scope",
  "domain_specificity",
  "annotation_scheme_compatibility",
  "granularity_match_with_target",
  "dialectal_distance"
)

characteristic_vars_categorical <- c(
  "register",
  "text_genre",
  "primary_domain",
  "granularity_level",
  "task_framing",
  "theoretical_framework"
)

# Base model controls (unpenalized)
control_vars_continuous <- c(
  "z_learning_rate",
  "z_weight_decay",
  "z_warmup_ratio"
)

control_vars_categorical <- c(
  "task",
  "is_cgec",
  "per_device_train_batch_size",
  "num_train_epochs"
)

# Apply min-max to continuous characteristics
df_lasso <- df_reg %>%
  mutate(across(all_of(characteristic_vars_continuous),
                minmax,
                .names = "mm_{.col}"))

# One-hot encode categorical characteristics
df_lasso <- dummy_cols(
  df_lasso,
  select_columns = characteristic_vars_categorical,
  remove_first_dummy  = TRUE,   # avoid dummy trap
  remove_selected_columns = TRUE
)

# One-hot encode categorical controls
df_lasso <- dummy_cols(
  df_lasso,
  select_columns = control_vars_categorical,
  remove_first_dummy  = TRUE,
  remove_selected_columns = TRUE
)

# Collect all predictor column names after encoding
mm_characteristic_cols <- paste0("mm_", characteristic_vars_continuous)
ordinal_cols            <- characteristic_vars_ordinal
dummy_characteristic_cols <- grep(
  paste(characteristic_vars_categorical, collapse = "|"),
  names(df_lasso), value = TRUE
)
dummy_control_cols <- grep(
  paste(control_vars_categorical, collapse = "|"),
  names(df_lasso), value = TRUE
)

all_characteristic_cols <- c(
  mm_characteristic_cols,
  ordinal_cols,
  dummy_characteristic_cols
)

all_control_cols <- c(
  control_vars_continuous,
  dummy_control_cols
)

all_predictor_cols <- c(all_control_cols, all_characteristic_cols)

# Build predictor matrix and outcome vector
# Complete cases only (glmnet does not handle NAs)
df_lasso_complete <- df_lasso %>%
  select(c_f1_macro, all_of(all_predictor_cols)) %>%
  drop_na()

X <- as.matrix(df_lasso_complete[, all_predictor_cols])
y <- df_lasso_complete$c_f1_macro

# Penalty factor:
#   0 = unpenalized (base model controls, always retained)
#   1 = penalized (dataset characteristics, subject to shrinkage)
penalty_factors <- c(
  rep(0, length(all_control_cols)),          # controls: never shrunk
  rep(1, length(all_characteristic_cols))    # characteristics: Lasso penalized
)

cat("\nPredictor matrix dimensions:", dim(X), "\n")
cat("Unpenalized predictors (controls):", length(all_control_cols), "\n")
cat("Penalized predictors (characteristics):", length(all_characteristic_cols), "\n")

# --- Step 2: Cross-validated Lasso ---

set.seed(42)  # for reproducibility of CV folds

lasso_cv <- cv.glmnet(
  x             = X,
  y             = y,
  alpha         = 0.5,              # alpha = 0.5 -> Elastic Net (L1 + L2 penalty mix)
  penalty.factor = penalty_factors,
  nfolds        = 10,             # 10-fold CV
  standardize   = FALSE           # already scaled; don't standardize again
)

# Plot CV error curve
plot(lasso_cv)
title("Lasso CV: MSE vs log(lambda)", line = 3)

cat("\n--- Optimal lambda values ---\n")
cat("lambda.min (lowest CV error):", lasso_cv$lambda.min, "\n")
cat("lambda.1se (most regularized within 1 SE):", lasso_cv$lambda.1se, "\n")

# --- Step 3: Extract non-zero coefficients ---
# At lambda.min: less regularized, more variables retained
coef_min <- coef(lasso_cv, s = "lambda.min")
coef_min_df <- as.data.frame(as.matrix(coef_min))
coef_min_df$predictor <- rownames(coef_min_df)
colnames(coef_min_df)[1] <- "coefficient"
coef_min_nonzero <- coef_min_df %>%
  filter(predictor != "(Intercept)", coefficient != 0) %>%
  arrange(desc(abs(coefficient)))

cat("\n--- Non-zero coefficients at lambda.min ---\n")
print(coef_min_nonzero)

# At lambda.1se: more regularized, parsimonious — preferred for small N
coef_1se <- coef(lasso_cv, s = "lambda.1se")
coef_1se_df <- as.data.frame(as.matrix(coef_1se))
coef_1se_df$predictor <- rownames(coef_1se_df)
colnames(coef_1se_df)[1] <- "coefficient"
coef_1se_nonzero <- coef_1se_df %>%
  filter(predictor != "(Intercept)", coefficient != 0) %>%
  arrange(desc(abs(coefficient)))

cat("\n--- Non-zero coefficients at lambda.1se (preferred) ---\n")
print(coef_1se_nonzero)

##### C) Logistic Regression (Errors) #####
##### C.1 Error type distribution #####
summary(df_errors_complete$error_type)
df_errors_complete$error_type <- relevel(
  df_errors_complete$error_type,
  ref = "FP"
)
df_errors_reg <- merge(
  df_errors_complete,
  df_characterization,
  by = c("dataset", "year")
)
df_errors_reg <- merge(
  df_errors_reg,
  df_results_full,
  by = c("dataset", "year", "task", "is_cgec", "model")
)
continuous_vars <- c(
  "total_instances", "total_tokens", "unique_tokens", "avg_instance_length",
  "median_instance_length", "vocabulary_overlap", "size_ratio_log", 
  "binary_argumentative_pct", "binary_balance_ratio", "component_premise_pct",
  "component_claim_pct", "binary_nonargumentative_pct", "vocabulary_overlap",
  "domain_similarity_tfidf", "linguistic_distance_beto", "domain_distance_beto",
  "linguistic_distance_roberta", "domain_distance_roberta", "linguistic_distance_xlmr",
  "domain_distance_xlmr", "size_ratio_log", "learning_rate", "num_train_epochs",
  "warmup_ratio", "weight_decay", "warmup_ratio"
)
df_errors_reg <- df_errors_reg %>%
  mutate(across(all_of(continuous_vars),   # reuse continuous_vars from B
                ~ scale(.x, center = TRUE, scale = TRUE)[,1],
                .names = "z_{.col}"))
df_errors_reg <- df_errors_reg %>%
  mutate(
    err_FP = as.integer(error_type == "FP"),
    err_FN = as.integer(error_type == "FN"),
    err_LE = as.integer(error_type == "LE")
  )
model_fp <- glmmTMB(
  err_FP ~
    task +
    is_cgec +
    z_learning_rate +
    z_weight_decay +
    z_warmup_ratio +
    per_device_train_batch_size +
    num_train_epochs +
    (1 | dataset) +
    (1 | model),
  data = df_errors_reg,
  family = binomial()
)
summary(model_fp)
r2(model_fp)

model_fn <- update(model_fp, formula = err_FN ~ .)
summary(model_fn)
r2(model_fn)

model_ler <- update(model_fp, formula = err_LE ~ .)
summary(model_ler)
r2(model_ler)

exp(coef(summary(model_fp))$cond[, "Estimate"])
exp(coef(summary(model_fn))$cond[, "Estimate"])
exp(coef(summary(model_ler))$cond[, "Estimate"])

tidy(model_fp, effects = "fixed", exponentiate = TRUE)
tidy(model_fn, effects = "fixed", exponentiate = TRUE)
tidy(model_ler, effects = "fixed", exponentiate = TRUE)

##### C.2 Error cause distribution #####
summary(df_errors_annotated$error_type)
df_errors_annotated$error_type <- relevel(
  df_errors_annotated$error_type,
  ref = "FP"
)
df_errors_annotated$is_cgec <- ifelse(df_errors_annotated$is_cgec == "yes", 1, 0)
df_errors_reg <- merge(
  df_errors_annotated,
  df_characterization,
  by = c("dataset", "year")
)
df_errors_reg <- merge(
  df_errors_reg,
  df_results_full,
  by = c("dataset", "year", "task", "is_cgec", "model")
)
continuous_vars <- c(
  "total_instances", "total_tokens", "unique_tokens", "avg_instance_length",
  "median_instance_length", "vocabulary_overlap", "size_ratio_log", 
  "binary_argumentative_pct", "binary_balance_ratio", "component_premise_pct",
  "component_claim_pct", "binary_nonargumentative_pct", "vocabulary_overlap",
  "domain_similarity_tfidf", "linguistic_distance_beto", "domain_distance_beto",
  "linguistic_distance_roberta", "domain_distance_roberta", "linguistic_distance_xlmr",
  "domain_distance_xlmr", "size_ratio_log", "learning_rate", "num_train_epochs",
  "warmup_ratio", "weight_decay", "warmup_ratio"
)
df_errors_reg <- df_errors_reg %>%
  mutate(across(all_of(continuous_vars),   # reuse continuous_vars from B
                ~ scale(.x, center = TRUE, scale = TRUE)[,1],
                .names = "z_{.col}"))
df_errors_reg <- df_errors_reg %>%
  mutate(
    err_FP = as.integer(error_type == "FP"),
    err_FN = as.integer(error_type == "FN"),
    err_LE = as.integer(error_type == "LE")
  )
df_causes_reg <- df_errors_reg %>%
  filter(!is.na(primary_cause)) %>%
  mutate(
    cause_group = case_when(
      grepl("^2\\.A", primary_cause) ~ "A",
      grepl("^2\\.B", primary_cause) ~ "B",
      grepl("^2\\.C", primary_cause) ~ "C",
      TRUE ~ NA_character_
    ),
    cause_group = factor(cause_group)
  )
df_causes_reg$cause_group <- relevel(df_causes_reg$cause_group, ref = "A")
cat("\nCause group distribution:\n")
print(table(df_causes_reg$cause_group))
df_causes_reg <- df_causes_reg %>%
  mutate(
    cause_A = as.integer(cause_group == "A"),
    cause_B = as.integer(cause_group == "B"),
    cause_C = as.integer(cause_group == "C")
  )

# Base models
base_cause_A <- glmmTMB(
  cause_A ~
    task +
    is_cgec +
    z_learning_rate +
    z_weight_decay +
    z_warmup_ratio +
    per_device_train_batch_size +
    num_train_epochs +
    (1 | dataset), #+ (1 | model)
  data = df_causes_reg,
  family = binomial()
)
base_cause_B <- update(base_cause_A, formula = cause_B ~ .)
base_cause_C <- update(base_cause_A, formula = cause_C ~ .)
summary(base_cause_A)
r2(base_cause_A)
summary(base_cause_B)
r2(base_cause_B)
summary(base_cause_C)
r2(base_cause_C)

# Full models
h_cause_B_full <- glmmTMB(
  cause_B ~
    task +
    is_cgec +
    error_type +
    z_domain_similarity_tfidf +
    z_linguistic_distance_beto +
    annotation_scheme_compatibility +
    dialectal_distance +
    z_total_instances +
    (1 | dataset),
  data = df_causes_reg,
  family = binomial()
)
h_cause_C_full <- update(h_cause_B_full, formula = cause_C ~ .)
h_cause_A_full <- update(h_cause_B_full, formula = cause_A ~ .)

summary(h_cause_A_full)
r2(h_cause_A_full)
summary(h_cause_B_full)
r2(h_cause_B_full)
summary(h_cause_C_full)
r2(h_cause_C_full)

anova(base_cause_A, h_cause_A_full)
anova(base_cause_B, h_cause_B_full)
anova(base_cause_C, h_cause_C_full)
exp(coef(summary(h_cause_B_full))$cond[, "Estimate"])
exp(coef(summary(h_cause_C_full))$cond[, "Estimate"])

chisq.test(table(df_causes_reg$error_type, df_causes_reg$cause_group))
##### C.3 Lasso regression #####
# Min-max normalization helper (applied only to dataset characteristics)
minmax <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (diff(rng) == 0) return(rep(0, length(x)))  # constant variable -> zero
  (x - rng[1]) / diff(rng)
}

# Dataset characteristics to include in Lasso
# (all characteristics from characterization matrix, excluding identifiers)
characteristic_vars_continuous <- c(
  "total_instances",
  "total_tokens",
  "unique_tokens",
  "avg_instance_length",
  "median_instance_length",
  "vocabulary_overlap",
  "domain_similarity_tfidf",
  "size_ratio_log",
  "binary_argumentative_pct",
  "binary_balance_ratio",
  "component_premise_pct",
  "component_claim_pct",
  "binary_nonargumentative_pct",
  "domain_similarity_tfidf",
  "linguistic_distance_beto",
  "domain_distance_beto",
  "linguistic_distance_roberta",
  "domain_distance_roberta",
  "linguistic_distance_xlmr",
  "domain_distance_xlmr"
)

characteristic_vars_ordinal <- c(
  "topical_scope",
  "domain_specificity",
  "annotation_scheme_compatibility",
  "granularity_match_with_target",
  "dialectal_distance"
)

characteristic_vars_categorical <- c(
  "register",
  "text_genre",
  "primary_domain",
  "granularity_level",
  "task_framing",
  "theoretical_framework"
)

# Base model controls (unpenalized)
control_vars_continuous <- c(
  "z_learning_rate",
  "z_weight_decay",
  "z_warmup_ratio"
)

control_vars_categorical <- c(
  "task",
  "is_cgec",
  "per_device_train_batch_size",
  "num_train_epochs"
)

# Apply min-max to continuous characteristics
df_lasso <- df_errors_reg %>%
  mutate(across(all_of(characteristic_vars_continuous),
                minmax,
                .names = "mm_{.col}"))

# One-hot encode categorical characteristics
df_lasso <- dummy_cols(
  df_lasso,
  select_columns = characteristic_vars_categorical,
  remove_first_dummy  = TRUE,   # avoid dummy trap
  remove_selected_columns = TRUE
)

# One-hot encode categorical controls
df_lasso <- dummy_cols(
  df_lasso,
  select_columns = control_vars_categorical,
  remove_first_dummy  = TRUE,
  remove_selected_columns = TRUE
)

# Collect all predictor column names after encoding
mm_characteristic_cols <- paste0("mm_", characteristic_vars_continuous)
ordinal_cols            <- characteristic_vars_ordinal
dummy_characteristic_cols <- grep(
  paste(characteristic_vars_categorical, collapse = "|"),
  names(df_lasso), value = TRUE
)
dummy_control_cols <- grep(
  paste(control_vars_categorical, collapse = "|"),
  names(df_lasso), value = TRUE
)

all_characteristic_cols <- c(
  mm_characteristic_cols,
  ordinal_cols,
  dummy_characteristic_cols
)

all_control_cols <- c(
  control_vars_continuous,
  dummy_control_cols
)
all_predictor_cols <- c(all_control_cols, all_characteristic_cols)
df_lasso <- df_lasso %>%
  filter(!is.na(primary_cause)) %>%
  mutate(
    cause_group = case_when(
      grepl("^2\\.A", primary_cause) ~ "A",
      grepl("^2\\.B", primary_cause) ~ "B",
      grepl("^2\\.C", primary_cause) ~ "C",
      TRUE ~ NA_character_
    ),
    cause_group = factor(cause_group)
  )
df_lasso$cause_group <- relevel(df_lasso$cause_group, ref = "A")
cat("\nCause group distribution:\n")
print(table(df_lasso$cause_group))
df_lasso <- df_lasso %>%
  mutate(
    cause_A = as.integer(cause_group == "A"),
    cause_B = as.integer(cause_group == "B"),
    cause_C = as.integer(cause_group == "C")
  )
# Build predictor matrix and outcome vector
# Complete cases only (glmnet does not handle NAs)
X <- as.matrix(df_lasso[, all_predictor_cols])
y <- df_lasso$error_type
# For C.2
y <- df_lasso$cause_group

# Penalty factor:
#   0 = unpenalized (base model controls, always retained)
#   1 = penalized (dataset characteristics, subject to shrinkage)
penalty_factors <- c(
  rep(0, length(all_control_cols)),          # controls: never shrunk
  rep(1, length(all_characteristic_cols))    # characteristics: Lasso penalized
)

cat("\nPredictor matrix dimensions:", dim(X), "\n")
cat("Unpenalized predictors (controls):", length(all_control_cols), "\n")
cat("Penalized predictors (characteristics):", length(all_characteristic_cols), "\n")

# --- Step 2: Cross-validated Lasso ---

set.seed(42)  # for reproducibility of CV folds

lasso_cv <- cv.glmnet(
  x              = X,
  y              = y,
  family         = "multinomial",   # 🔥 key change
  alpha          = 0.5,
  penalty.factor = penalty_factors,
  nfolds         = 10,
  standardize    = FALSE,
  type.measure   = "class"          # classification error
)

coef_min <- coef(lasso_cv, s = "lambda.min")
coef_min$FP
coef_min$FN
coef_min$LE

# Plot CV error curve
plot(lasso_cv)
title("Lasso CV: MSE vs log(lambda)", line = 3)

cat("\n--- Optimal lambda values ---\n")
cat("lambda.min (lowest CV error):", lasso_cv$lambda.min, "\n")
cat("lambda.1se (most regularized within 1 SE):", lasso_cv$lambda.1se, "\n")

# --- Step 3: Extract non-zero coefficients ---
# At lambda.min: less regularized, more variables retained
coef_min <- coef(lasso_cv, s = "lambda.min")
coef_min_df <- as.data.frame(as.matrix(coef_min))
coef_min_df$predictor <- rownames(coef_min_df)
colnames(coef_min_df)[1] <- "coefficient"
coef_min_nonzero <- coef_min_df %>%
  filter(predictor != "(Intercept)", coefficient != 0) %>%
  arrange(desc(abs(coefficient)))

cat("\n--- Non-zero coefficients at lambda.min ---\n")
print(coef_min_nonzero)

# At lambda.1se: more regularized, parsimonious — preferred for small N
coef_1se <- coef(lasso_cv, s = "lambda.1se")
coef_1se_df <- as.data.frame(as.matrix(coef_1se))
coef_1se_df$predictor <- rownames(coef_1se_df)
colnames(coef_1se_df)[1] <- "coefficient"
coef_1se_nonzero <- coef_1se_df %>%
  filter(predictor != "(Intercept)", coefficient != 0) %>%
  arrange(desc(abs(coefficient)))

cat("\n--- Non-zero coefficients at lambda.1se (preferred) ---\n")
print(coef_1se_nonzero)

