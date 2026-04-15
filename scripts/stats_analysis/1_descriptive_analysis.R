##### Imports ##### 
library(fastDummies)
library(tidyverse)
library(ggplot2)
library(visreg)
library(glmnet)
library(dplyr)
library(tidyr)
library(purrr)
library(broom)
library(gridExtra)
library(gt)

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
df_results_full$model
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

##### B) Descriptive Analysis ##### 
##### B.1 Tables \ Performance ##### 
df_results_full$dataset <- paste(df_results_full$dataset, df_results_full$year, sep="_")
df_results_full$dataset <- as.factor(df_results_full$dataset)

# By model
perf_by_model <- df_results_full %>%
  group_by(model) %>%
  summarise(
    mean_accuracy  = round(mean(accuracy,        na.rm = TRUE), 3),
    sd_accuracy    = round(sd(accuracy,          na.rm = TRUE), 3),
    mean_f1        = round(mean(f1_macro,        na.rm = TRUE), 3),
    sd_f1          = round(sd(f1_macro,          na.rm = TRUE), 3),
    mean_precision = round(mean(precision_macro, na.rm = TRUE), 3),
    mean_recall    = round(mean(recall_macro,    na.rm = TRUE), 3),
    n_runs         = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_f1))
perf_by_model
# write.csv(perf_by_model, "perf_by_model.csv")

# By dataset

unique(df_results_full$dataset)
perf_by_dataset <- df_results_full %>%
  filter(dataset %in% c("diversity_", "full_", "similarity_", "topk_")) %>%  # <- add your combined datasets here
  group_by(dataset) %>%
  summarise(
    mean_accuracy  = round(mean(accuracy,        na.rm = TRUE), 3),
    sd_accuracy    = round(sd(accuracy,          na.rm = TRUE), 3),
    mean_f1        = round(mean(f1_macro,        na.rm = TRUE), 3),
    sd_f1          = round(sd(f1_macro,          na.rm = TRUE), 3),
    mean_precision = round(mean(precision_macro, na.rm = TRUE), 3),
    mean_recall    = round(mean(recall_macro,    na.rm = TRUE), 3),
    n_runs         = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_f1))
perf_by_dataset
# write.csv(perf_by_dataset, "perf_by_dataset.csv")

# By task
perf_by_task <- df_results_full %>%
  group_by(task) %>%
  summarise(
    mean_accuracy  = round(mean(accuracy,        na.rm = TRUE), 3),
    sd_accuracy    = round(sd(accuracy,          na.rm = TRUE), 3),
    mean_f1        = round(mean(f1_macro,        na.rm = TRUE), 3),
    sd_f1          = round(sd(f1_macro,          na.rm = TRUE), 3),
    mean_precision = round(mean(precision_macro, na.rm = TRUE), 3),
    mean_recall    = round(mean(recall_macro,    na.rm = TRUE), 3),
    n_runs         = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_f1))
perf_by_task
# write.csv(perf_by_task, "perf_by_task.csv")

# By model x dataset x task
perf_full <- df_results_full %>%
  group_by(model, dataset, task) %>%
  summarise(
    mean_accuracy = round(mean(accuracy,  na.rm = TRUE), 3),
    mean_f1       = round(mean(f1_macro,  na.rm = TRUE), 3),
    mean_precision = round(mean(precision_macro, na.rm = TRUE), 3),
    mean_recall    = round(mean(recall_macro,    na.rm = TRUE), 3),
    n_runs         = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_f1))
perf_full
# write.csv(perf_full, "perf_full.csv")

##### B.2 Tables \ Errors ##### 
df_errors_complete$dataset <- paste(df_errors_complete$dataset, df_errors_complete$year, sep="_")
df_errors_complete$dataset <- as.factor(df_errors_complete$dataset)

df_errors_annotated$dataset <- paste(df_errors_annotated$dataset, df_errors_annotated$year, sep="_")
df_errors_annotated$dataset <- as.factor(df_errors_annotated$dataset)

# By model
errors_by_model <- df_errors_complete %>%
  group_by(model) %>%
  summarise(
    n_errors = n(),
    n_FN     = sum(error_type == "FN"),
    n_FP     = sum(error_type == "FP"),
    n_LE     = sum(error_type == "LE"),
    pct_FN   = round(n_FN / n_errors * 100, 2),
    pct_FP   = round(n_FP / n_errors * 100, 2),
    pct_LE   = round(n_LE / n_errors * 100, 2),
    .groups  = "drop"
  ) %>%
  arrange(desc(n_errors))
errors_by_model
# write.csv(errors_by_model, "errors_by_model.csv")

errors_by_model_annotated <- df_errors_annotated %>%
  group_by(model) %>%
  summarise(
    n_errors = n(),
    n_2.A.1     = sum(primary_cause == "2.A.1", na.rm = TRUE),
    n_2.A.2     = sum(primary_cause == "2.A.2", na.rm = TRUE),
    n_2.A.3     = sum(primary_cause == "2.A.3", na.rm = TRUE),
    n_2.A.4     = sum(primary_cause == "2.A.4", na.rm = TRUE),
    n_2.A.5     = sum(primary_cause == "2.A.5", na.rm = TRUE),
    n_2.B.1     = sum(primary_cause == "2.B.1", na.rm = TRUE),
    n_2.B.2     = sum(primary_cause == "2.B.2", na.rm = TRUE),
    n_2.C.1     = sum(primary_cause == "2.C.1", na.rm = TRUE),
    n_2.C.2     = sum(primary_cause == "2.C.2", na.rm = TRUE),
    n_2.C.3     = sum(primary_cause == "2.C.3", na.rm = TRUE),
    pct_2.A.1    = round(n_2.A.1 / n_errors * 100, 2),
    pct_2.A.2   = round(n_2.A.2 / n_errors * 100, 2),
    pct_2.A.3   = round(n_2.A.3 / n_errors * 100, 2),
    pct_2.A.4    = round(n_2.A.4 / n_errors * 100, 2),
    pct_2.A.5   = round(n_2.A.5 / n_errors * 100, 2),
    pct_2.B.1   = round(n_2.B.1 / n_errors * 100, 2),
    pct_2.B.2    = round(n_2.B.2 / n_errors * 100, 2),
    pct_2.C.1   = round(n_2.C.1 / n_errors * 100, 2),
    pct_2.C.2   = round(n_2.C.2 / n_errors * 100, 2),
    pct_2.C.3   = round(n_2.C.3 / n_errors * 100, 2),
    .groups  = "drop"
  ) %>%
  arrange(desc(n_errors))
errors_by_model_annotated
# write.csv(errors_by_model, "errors_by_model2.csv")

# By dataset
errors_by_dataset <- df_errors_complete %>%
  group_by(dataset) %>%
  summarise(
    n_errors = n(),
    n_FN     = sum(error_type == "FN"),
    n_FP     = sum(error_type == "FP"),
    n_LE     = sum(error_type == "LE"),
    pct_FN   = round(n_FN / n_errors * 100, 2),
    pct_FP   = round(n_FP / n_errors * 100, 2),
    pct_LE   = round(n_LE / n_errors * 100, 2),
    .groups  = "drop"
  ) %>%
  arrange(desc(n_errors))
errors_by_dataset
# write.csv(errors_by_dataset, "errors_by_dataset.csv")

errors_by_dataset_annotated <- df_errors_annotated %>%
  group_by(dataset) %>%
  summarise(
    n_errors = n(),
    n_2.A.1     = sum(primary_cause == "2.A.1", na.rm = TRUE),
    n_2.A.2     = sum(primary_cause == "2.A.2", na.rm = TRUE),
    n_2.A.3     = sum(primary_cause == "2.A.3", na.rm = TRUE),
    n_2.A.4     = sum(primary_cause == "2.A.4", na.rm = TRUE),
    n_2.A.5     = sum(primary_cause == "2.A.5", na.rm = TRUE),
    n_2.B.1     = sum(primary_cause == "2.B.1", na.rm = TRUE),
    n_2.B.2     = sum(primary_cause == "2.B.2", na.rm = TRUE),
    n_2.C.1     = sum(primary_cause == "2.C.1", na.rm = TRUE),
    n_2.C.2     = sum(primary_cause == "2.C.2", na.rm = TRUE),
    n_2.C.3     = sum(primary_cause == "2.C.3", na.rm = TRUE),
    n_NA      = sum(is.na(primary_cause)),
    pct_2.A.1    = round(n_2.A.1 / n_errors * 100, 2),
    pct_2.A.2   = round(n_2.A.2 / n_errors * 100, 2),
    pct_2.A.3   = round(n_2.A.3 / n_errors * 100, 2),
    pct_2.A.4    = round(n_2.A.4 / n_errors * 100, 2),
    pct_2.A.5   = round(n_2.A.5 / n_errors * 100, 2),
    pct_2.B.1   = round(n_2.B.1 / n_errors * 100, 2),
    pct_2.B.2    = round(n_2.B.2 / n_errors * 100, 2),
    pct_2.C.1   = round(n_2.C.1 / n_errors * 100, 2),
    pct_2.C.2   = round(n_2.C.2 / n_errors * 100, 2),
    pct_2.C.3   = round(n_2.C.3 / n_errors * 100, 2),
    .groups  = "drop"
  ) %>%
  arrange(desc(n_errors))
errors_by_dataset_annotated
# write.csv(errors_by_dataset, "errors_by_dataset2.csv")

# By task
errors_by_task <- df_errors_complete %>%
  group_by(task) %>%
  summarise(
    n_errors = n(),
    n_FN     = sum(error_type == "FN"),
    n_FP     = sum(error_type == "FP"),
    n_LE     = sum(error_type == "LE"),
    pct_FN   = round(n_FN / n_errors * 100, 2),
    pct_FP   = round(n_FP / n_errors * 100, 2),
    pct_LE   = round(n_LE / n_errors * 100, 2),
    .groups  = "drop"
  ) %>%
  arrange(desc(n_errors))
errors_by_task
# write.csv(errors_by_task, "errors_by_task.csv")

errors_by_task_annotated  <- df_errors_annotated %>%
  group_by(task) %>%
  summarise(
    n_errors = n(),
    n_2.A.1     = sum(primary_cause == "2.A.1", na.rm = TRUE),
    n_2.A.2     = sum(primary_cause == "2.A.2", na.rm = TRUE),
    n_2.A.3     = sum(primary_cause == "2.A.3", na.rm = TRUE),
    n_2.A.4     = sum(primary_cause == "2.A.4", na.rm = TRUE),
    n_2.A.5     = sum(primary_cause == "2.A.5", na.rm = TRUE),
    n_2.B.1     = sum(primary_cause == "2.B.1", na.rm = TRUE),
    n_2.B.2     = sum(primary_cause == "2.B.2", na.rm = TRUE),
    n_2.C.1     = sum(primary_cause == "2.C.1", na.rm = TRUE),
    n_2.C.2     = sum(primary_cause == "2.C.2", na.rm = TRUE),
    n_2.C.3     = sum(primary_cause == "2.C.3", na.rm = TRUE),
    n_NA      = sum(is.na(primary_cause)),
    pct_2.A.1    = round(n_2.A.1 / n_errors * 100, 2),
    pct_2.A.2   = round(n_2.A.2 / n_errors * 100, 2),
    pct_2.A.3   = round(n_2.A.3 / n_errors * 100, 2),
    pct_2.A.4    = round(n_2.A.4 / n_errors * 100, 2),
    pct_2.A.5   = round(n_2.A.5 / n_errors * 100, 2),
    pct_2.B.1   = round(n_2.B.1 / n_errors * 100, 2),
    pct_2.B.2    = round(n_2.B.2 / n_errors * 100, 2),
    pct_2.C.1   = round(n_2.C.1 / n_errors * 100, 2),
    pct_2.C.2   = round(n_2.C.2 / n_errors * 100, 2),
    pct_2.C.3   = round(n_2.C.3 / n_errors * 100, 2),
    .groups  = "drop"
  ) %>%
  arrange(desc(n_errors))
errors_by_task_annotated
# write.csv(errors_by_task, "errors_by_task2.csv")

# By model x dataset x task
error_full <- df_errors_complete %>%
  group_by(model, dataset, task) %>%
  summarise(
    n_errors = n(),
    n_FN     = sum(error_type == "FN"),
    n_FP     = sum(error_type == "FP"),
    n_LE     = sum(error_type == "LE"),
    .groups  = "drop"
  ) %>%
  arrange(desc(n_errors))
error_full

error_full_annotated <- df_errors_annotated %>%
  group_by(model, dataset, task) %>%
  summarise(
    n_errors = n(),
    n_2.A.1     = sum(primary_cause == "2.A.1", na.rm = TRUE),
    n_2.A.2     = sum(primary_cause == "2.A.2", na.rm = TRUE),
    n_2.A.3     = sum(primary_cause == "2.A.3", na.rm = TRUE),
    n_2.A.4     = sum(primary_cause == "2.A.4", na.rm = TRUE),
    n_2.A.5     = sum(primary_cause == "2.A.5", na.rm = TRUE),
    n_2.B.1     = sum(primary_cause == "2.B.1", na.rm = TRUE),
    n_2.B.2     = sum(primary_cause == "2.B.2", na.rm = TRUE),
    n_2.C.1     = sum(primary_cause == "2.C.1", na.rm = TRUE),
    n_2.C.2     = sum(primary_cause == "2.C.2", na.rm = TRUE),
    n_2.C.3     = sum(primary_cause == "2.C.3", na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  arrange(desc(n_errors))
error_full_annotated
