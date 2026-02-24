# Loading libraries
library(tidyverse)
library(skimr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(VIM)      
library(mice)   
library(scales)
library(caret)
library(nnet)
library(pROC)
library(xgboost)
library(randomForest)
library(glmnet)
library(forcats)

# Read dataset
mortgage <- read_csv('Mortgage.csv')

# Basic inspection
glimpse(mortgage)
summary(mortgage)
skim(mortgage)

# View column names
colnames(mortgage)

# Select a few key variables for exploration
key_vars <- c("FICO_orig_time", "LTV_time", "interest_rate_time",
              "balance_time", "uer_time", "hpi_time")
mortgage_subset <- mortgage %>% select(all_of(key_vars))

# Summary statistics for key variables
mortgage_subset %>% 
  summarise(across(everything(), list(
    mean   = ~mean(., na.rm = TRUE),
    median = ~median(., na.rm = TRUE),
    sd     = ~sd(., na.rm = TRUE),
    min    = ~min(., na.rm = TRUE),
    max    = ~max(., na.rm = TRUE)
  )))

# Add readable labels for target (panel level)
mortgage <- mortgage %>%
  mutate(
    status_lbl = factor(
      status_time,
      levels = c(0, 1, 2),
      labels = c("Current", "Default", "Payoff")
    )
  )

# Outcome Class Balance
mortgage %>%
  count(status_lbl) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = status_lbl, y = pct)) +
  geom_col(width = 0.6, fill = "steelblue") +
  geom_text(aes(label = scales::percent(pct, accuracy = 0.1)),
            vjust = -0.25, size = 3.8) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Outcome Class Balance (Panel-Level)",
       x = NULL, y = "Share of Observations") +
  theme_minimal()

# FICO Distribution
ggplot(mortgage, aes(x = FICO_orig_time)) +
  geom_histogram(binwidth = 25, color = "white", fill = "grey40") +
  labs(title = "Distribution of FICO Scores at Origination",
       x = "FICO Score", y = "Count") +
  theme_minimal()

# LTV Distribution
ggplot(mortgage %>% filter(!is.na(LTV_time)),
       aes(x = LTV_time)) +
  geom_histogram(binwidth = 5, color = "white", fill = "grey40") +
  labs(title = "Distribution of Loan-to-Value (LTV) Ratio",
       x = "LTV (%)", y = "Count") +
  theme_minimal()

# Interest Rate Distribution
ggplot(mortgage, aes(x = interest_rate_time)) +
  geom_histogram(binwidth = 0.5, color = "white", fill = "grey40") +
  labs(title = "Distribution of Interest Rates",
       x = "Interest Rate (%)", y = "Count") +
  theme_minimal()

# Outstanding Balance Distribution
ggplot(mortgage, aes(x = balance_time)) +
  geom_histogram(binwidth = 25000, color = "white", fill = "grey40") +
  scale_x_continuous(labels = scales::label_number(scale_cut = scales::cut_si(" "))) +
  labs(title = "Distribution of Outstanding Balances",
       x = "Balance ($)", y = "Count") +
  theme_minimal()

# Portfolio Mix (Property Type & Investor Share)
mortgage %>%
  transmute(
    Condo = REtype_CO_orig_time,
    PUD   = REtype_PU_orig_time,
    SF    = REtype_SF_orig_time,
    Investor = investor_orig_time
  ) %>%
  summarise(across(everything(), ~mean(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "flag", values_to = "share") %>%
  ggplot(aes(x = flag, y = share)) +
  geom_col(width = 0.6, fill = "grey40") +
  geom_text(aes(label = percent(share, accuracy = 0.1)), vjust = -0.25) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Portfolio Mix: Property Type & Investor Share (Panel-Level)",
       x = NULL, y = "Share of Observations") +
  theme_minimal()

# Missingness checks
cat("\n Missingness in dataset \n")
print(sapply(mortgage, function(x) sum(is.na(x))))

# Missingness compact table (mice)
md.pattern(mortgage, rotate.names = TRUE)

# Work on a copy to preserve original
mort_logic <- mortgage

# LOGICAL CONSISTENCY CHECKS ON DATASET
# Helper: numeric mode (fallback to median if ties/empty)
num_mode <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return(NA_real_)
  tab <- sort(table(x), decreasing = TRUE)
  top <- tab[tab == max(tab)]
  vals <- as.numeric(names(top))
  if (length(vals) == 1) vals else median(vals)
}

# Force per-id consistency in Interest_Rate_orig_time
ir_by_id <- mort_logic %>%
  dplyr::group_by(id) %>%
  dplyr::summarise(ir_id = num_mode(Interest_Rate_orig_time), .groups = "drop")

mort_logic <- mort_logic %>%
  dplyr::left_join(ir_by_id, by = "id") %>%
  dplyr::mutate(
    Interest_Rate_orig_time = dplyr::if_else(!is.na(ir_id), ir_id, Interest_Rate_orig_time)
  ) %>%
  dplyr::select(-ir_id)

# Zeros in certain columns not logical
mort_logic <- mort_logic %>%
  mutate(
    balance_orig_time = if_else(balance_orig_time == 0, NA_real_, balance_orig_time),
    balance_orig_time = if_else((orig_time >= 0) & (balance_time > 0) & balance_orig_time == 0,
                                NA_real_, balance_orig_time)
  )

# Identify out of range cases and convert to NA for imputation 
mort_logic <- mort_logic %>%
  mutate(
    LTV_time             = if_else(LTV_time < 0 | LTV_time > 300, NA_real_, LTV_time),
    interest_rate_time   = if_else(interest_rate_time < 0 | interest_rate_time > 40, NA_real_, interest_rate_time),
    FICO_orig_time       = if_else(FICO_orig_time < 300 | FICO_orig_time > 850, NA_real_, FICO_orig_time),
    hpi_time             = if_else(hpi_time < 0, NA_real_, hpi_time),
    balance_time         = if_else(balance_time < 0, NA_real_, balance_time),
    balance_orig_time    = if_else(balance_orig_time < 0, NA_real_, balance_orig_time)
  )

#  Missingness AFTER logic checks
cat("\n Missingness AFTER logic checks \n")
print(sapply(mort_logic, function(x) sum(is.na(x))))
md.pattern(mort_logic, rotate.names = TRUE)

# KNN IMPUTATION (VIM::kNN) on the logic-NA dataset
# -> Grouped roughly by orig_time bins to reduce time leakage

# Build a coarse origination-time bin
mort_logic <- mort_logic %>%
  mutate(
    orig_time_bin = cut(
      orig_time,
      breaks = pretty(orig_time, n = 10),
      include.lowest = TRUE
    )
  )

# Exclude target variables & helper columns from KNN
to_exclude <- c("status_lbl", "status_time", "default_time", "payoff_time", "orig_time_bin")
to_exclude <- intersect(to_exclude, names(mort_logic))
knn_vars   <- setdiff(names(mort_logic), to_exclude)

set.seed(123)

# Start from a copy and overwrite only knn_vars with imputed values
mort_imp <- mort_logic

bins <- unique(mort_logic$orig_time_bin)
bins <- bins[!is.na(bins)]

for (b in bins) {
  idx <- which(mort_logic$orig_time_bin == b)
  sub <- mort_logic[idx, knn_vars, drop = FALSE]
  sub_imp <- VIM::kNN(sub, k = 5, imp_var = FALSE)
  mort_imp[idx, knn_vars] <- sub_imp[, knn_vars, drop = FALSE]
}

# Build final cleaned PANEL dataset (remove orig_time_bin)
mortgage_clean_full <- mort_imp %>%
  select(-orig_time_bin)

# We keep only the last available time-point per id (based on 'time')
# If some ids have all time = NA, they will be dropped.
mortgage_clean <- mortgage_clean_full %>%
  arrange(id, time) %>%
  group_by(id) %>%
  filter(!is.na(time)) %>%
  slice_tail(n = 1) %>%
  ungroup()

# From this point onward, ALL analysis uses mortgage_clean
# (one record per customer, at latest observation).
# Predictor vs Outcome Visuals
# Ensure outcome labels exist on the cleaned LAST-OBS data
mortgage_clean <- mortgage_clean %>%
  mutate(
    status_lbl = factor(
      status_time,
      levels = c(0, 1, 2),
      labels = c("Current", "Default", "Payoff")
    )
  )

# Violin + box: FICO vs Outcome
ggplot(mortgage_clean, aes(x = status_lbl, y = FICO_orig_time)) +
  geom_violin(fill = "grey80", color = "grey30", trim = TRUE) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.75) +
  labs(title = "FICO at Origination by Outcome ",
       x = NULL, y = "FICO (origination)") +
  theme_minimal()

# Violin + box: LTV vs Outcome
ggplot(mortgage_clean %>% filter(!is.na(LTV_time)),
       aes(x = status_lbl, y = LTV_time)) +
  geom_violin(fill = "grey80", color = "grey30", trim = TRUE) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.75) +
  labs(title = "LTV at Observation by Outcome ",
       x = NULL, y = "LTV (%)") +
  theme_minimal()

# Violin + box: Interest Rate vs Outcome
ggplot(mortgage_clean %>% filter(!is.na(interest_rate_time)),
       aes(x = status_lbl, y = interest_rate_time)) +
  geom_violin(fill = "grey80", color = "grey30", trim = TRUE) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.75) +
  labs(title = "Interest Rate at Observation by Outcome ",
       x = NULL, y = "Interest Rate (%)") +
  theme_minimal()

# Violin + box (log scale): Balance vs Outcome
ggplot(mortgage_clean %>% filter(!is.na(balance_time), balance_time > 0),
       aes(x = status_lbl, y = balance_time)) +
  geom_violin(fill = "grey80", color = "grey30", trim = TRUE) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.75) +
  scale_y_log10(labels = label_number(scale_cut = cut_si(" "))) +
  labs(title = "Outstanding Balance by Outcome (log scale, Last Obs per Customer)",
       x = NULL, y = "Balance ($, log10)") +
  theme_minimal()

# Class-wise density: LTV
ggplot(mortgage_clean %>% filter(!is.na(LTV_time)),
       aes(x = LTV_time, fill = status_lbl)) +
  geom_density(alpha = 0.3) +
  labs(title = "LTV Distributions by Outcome Class ",
       x = "LTV (%)", y = "Density", fill = "Outcome") +
  theme_minimal()

# Class-wise density: Interest Rate
ggplot(mortgage_clean %>% filter(!is.na(interest_rate_time)),
       aes(x = interest_rate_time, fill = status_lbl)) +
  geom_density(alpha = 0.3) +
  labs(title = "Interest Rate Distributions by Outcome Class ",
       x = "Interest Rate (%)", y = "Density", fill = "Outcome") +
  theme_minimal()

# PROPORTION CURVES (binned): Default/Payoff rates vs LTV
ltv_bin_rates <- mortgage_clean %>%
  filter(!is.na(LTV_time)) %>%
  mutate(ltv_bin = floor(LTV_time / 5) * 5) %>%        # 5-pt bins: 0,5,10,...
  group_by(ltv_bin) %>%
  summarise(
    n          = n(),
    default_rt = mean(status_time == 1, na.rm = TRUE),
    payoff_rt  = mean(status_time == 2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(default_rt, payoff_rt),
               names_to = "which", values_to = "rate") %>%
  mutate(which = recode(which, default_rt = "Default", payoff_rt = "Payoff"))

ggplot(ltv_bin_rates, aes(x = ltv_bin, y = rate, color = which)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  scale_x_continuous(breaks = scales::pretty_breaks()) +
  labs(title = "Outcome Rates by LTV (binned, Last Obs per Customer)",
       x = "LTV bin (%, width = 5)",
       y = "Rate", color = "Class") +
  theme_minimal()

# PROPORTION CURVES (binned): Default/Payoff vs FICO
fico_bin_rates <- mortgage_clean %>%
  filter(!is.na(FICO_orig_time)) %>%
  mutate(fico_bin = floor(FICO_orig_time / 20) * 20) %>%  # 20-pt bins
  group_by(fico_bin) %>%
  summarise(
    n          = n(),
    default_rt = mean(status_time == 1, na.rm = TRUE),
    payoff_rt  = mean(status_time == 2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(default_rt, payoff_rt),
               names_to = "which", values_to = "rate") %>%
  mutate(which = recode(which, default_rt = "Default", payoff_rt = "Payoff"))

ggplot(fico_bin_rates, aes(x = fico_bin, y = rate, color = which)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  scale_x_continuous(breaks = seq(300, 850, 50)) +
  labs(title = "Outcome Rates by FICO (binned, Last Obs per Customer)",
       x = "FICO bin (width = 20)",
       y = "Rate", color = "Class") +
  theme_minimal()

# PROPORTION CURVES (binned): Default/Payoff vs Interest Rate
ir_bin_rates <- mortgage_clean %>%
  filter(!is.na(interest_rate_time)) %>%
  mutate(ir_bin = floor(interest_rate_time / 0.5) * 0.5) %>%  # 0.5% bins
  group_by(ir_bin) %>%
  summarise(
    n          = n(),
    default_rt = mean(status_time == 1, na.rm = TRUE),
    payoff_rt  = mean(status_time == 2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(default_rt, payoff_rt),
               names_to = "which", values_to = "rate") %>%
  mutate(which = recode(which, default_rt = "Default", payoff_rt = "Payoff"))

ggplot(ir_bin_rates, aes(x = ir_bin, y = rate, color = which)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(title = "Outcome Rates by Interest Rate (binned, Last Obs per Customer)",
       x = "Interest Rate bin (%, width = 0.5)",
       y = "Rate", color = "Class") +
  theme_minimal()

# PROPORTION CURVES (binned): Default/Payoff vs Unemployment
uer_bin_rates <- mortgage_clean %>%
  filter(!is.na(uer_time)) %>%
  mutate(uer_bin = floor(uer_time / 0.5) * 0.5) %>%   # 0.5-pt bins
  group_by(uer_bin) %>%
  summarise(
    n          = n(),
    default_rt = mean(status_time == 1, na.rm = TRUE),
    payoff_rt  = mean(status_time == 2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(default_rt, payoff_rt),
               names_to = "which", values_to = "rate") %>%
  mutate(which = recode(which, default_rt = "Default", payoff_rt = "Payoff"))

ggplot(uer_bin_rates, aes(x = uer_bin, y = rate, color = which)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(title = "Outcome Rates by Unemployment (binned, Last Obs per Customer)",
       x = "Unemployment Rate bin (width = 0.5)",
       y = "Rate", color = "Class") +
  theme_minimal()

# PROPORTION CURVES (binned): Default/Payoff vs HPI
hpi_bin_rates <- mortgage_clean %>%
  filter(!is.na(hpi_time)) %>%
  mutate(hpi_bin = floor(hpi_time / 25) * 25) %>%       # 25-pt bins
  group_by(hpi_bin) %>%
  summarise(
    n          = n(),
    default_rt = mean(status_time == 1, na.rm = TRUE),
    payoff_rt  = mean(status_time == 2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(default_rt, payoff_rt),
               names_to = "which", values_to = "rate") %>%
  mutate(which = recode(which, default_rt = "Default", payoff_rt = "Payoff"))

ggplot(hpi_bin_rates, aes(x = hpi_bin, y = rate, color = which)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(title = "Outcome Rates by HPI Level (binned, Last Obs per Customer)",
       x = "HPI bin (width = 25)",
       y = "Rate", color = "Class") +
  theme_minimal()

# Stacked proportion bars: Outcome composition across FICO buckets
mortgage_clean %>%
  filter(!is.na(FICO_orig_time)) %>%
  mutate(fico_bucket = cut_width(FICO_orig_time, width = 50, boundary = 300)) %>%
  count(fico_bucket, status_lbl, name = "n") %>%
  group_by(fico_bucket) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(x = fico_bucket, y = pct, fill = status_lbl)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Outcome Mix by FICO Bucket ",
       x = "FICO Bucket (width = 50)", y = "Share within Bucket", fill = "Outcome") +
  theme_minimal() +
  coord_flip()

# Global feature tests: VIF (numeric), ANOVA (numeric~status), Chi-squared + Cramer's V (categorical~status)
# Ensure status_lbl exists on the cleaned data
if (!"status_lbl" %in% names(mortgage_clean) && "status_time" %in% names(mortgage_clean)) {
  mortgage_clean <- mortgage_clean %>%
    mutate(status_lbl = factor(status_time, levels = c(0,1,2), labels = c("Current","Default","Payoff")))
}

# Helper functions
is_binary_numeric <- function(x) is.numeric(x) && dplyr::n_distinct(stats::na.omit(x)) == 2

# function to calculate VIF table
calc_vif_table <- function(df_num) {
  # Remove columns with zero variance or all NA
  good <- names(df_num)[sapply(df_num, function(v) {v <- v[!is.na(v)]; length(v) > 1 && stats::var(v) > 0})]
  df_num <- df_num[, good, drop = FALSE]
  if (ncol(df_num) < 2) {
    return(tibble(variable = good, R2 = NA_real_, VIF = NA_real_))
  }
  map_dfr(names(df_num), function(v) {
    others <- setdiff(names(df_num), v)
    fml <- stats::as.formula(paste(v, "~", paste(others, collapse = " + ")))
    fit <- try(stats::lm(fml, data = df_num), silent = TRUE)
    if (inherits(fit, "try-error")) {
      tibble(variable = v, R2 = NA_real_, VIF = NA_real_)
    } else {
      r2 <- summary(fit)$r.squared
      tibble(variable = v, R2 = r2, VIF = ifelse(r2 < 1, 1/(1 - r2), NA_real_))
    }
  }) %>% arrange(desc(VIF))
}

# function for chi square test 
safe_chisq_cv <- function(x, y) {
  tab <- table(x, y, useNA = "no")
  res <- try(suppressWarnings(chisq.test(tab)), silent = TRUE)
  if (inherits(res, "try-error")) {
    return(tibble(statistic = NA_real_, df = NA_real_, p.value = NA_real_, cramers_v = NA_real_))
  }
  n <- sum(tab)
  k <- min(nrow(tab), ncol(tab))
  v <- if (n > 0 && (k - 1) > 0) sqrt(as.numeric(res$statistic) / (n * (k - 1))) else NA_real_
  tibble(statistic = as.numeric(res$statistic),
         df        = as.numeric(res$parameter),
         p.value   = as.numeric(res$p.value),
         cramers_v = v)
}

# Build typed feature lists (auto)
exclude_cols <- c("id", "status_time", "status_lbl")
exclude_cols <- unique(c(exclude_cols, grep("_imp$", names(mortgage_clean), value = TRUE)))

all_cols <- setdiff(names(mortgage_clean), exclude_cols)

cat_vars_auto <- all_cols[
  sapply(mortgage_clean[all_cols], function(col)
    is.character(col) || is.factor(col) || is.logical(col) || is_binary_numeric(col)
  )
]

num_vars_auto <- setdiff(all_cols, cat_vars_auto)
num_vars_auto <- num_vars_auto[
  sapply(mortgage_clean[num_vars_auto], function(x) is.numeric(x) && dplyr::n_distinct(stats::na.omit(x)) > 2)
]

cat("\n--- Auto-detected variables  ---\n")
cat("Numeric predictors (>", length(num_vars_auto), "): ", paste(num_vars_auto, collapse = ", "), "\n")
cat("Categorical predictors (>", length(cat_vars_auto), "): ", paste(cat_vars_auto, collapse = ", "), "\n")

df_test <- mortgage_clean %>%
  mutate(across(all_of(cat_vars_auto), ~ fct_explicit_na(as.factor(.), na_level = "Missing"))) %>%
  filter(!is.na(status_lbl))

# VIF among numeric predictors (multicollinearity)
vif_input <- df_test %>%
  select(all_of(num_vars_auto)) %>%
  tidyr::drop_na()

vif_table <- calc_vif_table(vif_input)
cat("\n VIF (numeric predictors, Last Obs per Customer)  \n")
print(vif_table)
cat("Guideline: VIF > 5 = moderate collinearity; VIF > 10 = high collinearity\n")
high_vif_vars <- vif_table %>% filter(!is.na(VIF) & VIF > 10) %>% pull(variable)

# One-way ANOVA: numeric predictor ~ status_lbl
anova_num_table <- map_dfr(num_vars_auto, function(v) {
  fml <- as.formula(paste(v, "~ status_lbl"))
  fit <- try(aov(fml, data = df_test), silent = TRUE)
  if (inherits(fit, "try-error")) return(tibble(variable = v, df = NA_real_, statistic = NA_real_, p.value = NA_real_))
  broom::tidy(fit) %>%
    filter(term == "status_lbl") %>%
    transmute(variable = v, df = df, statistic = statistic, p.value = p.value)
}) %>% arrange(p.value)

cat("\n One-way ANOVA (numeric ~ status_lbl, Last Obs per Customer)  \n")
print(anova_num_table)

# Chi-squared + Cramer's V: categorical predictor vs status_lbl
chi_cat_table <- map_dfr(cat_vars_auto, function(v) {
  out <- safe_chisq_cv(df_test[[v]], df_test$status_lbl)
  mutate(out, variable = v, .before = 1)
}) %>% arrange(p.value)

cat("\n Chi-squared (categorical ~ status_lbl) with Cramér's V   \n")
print(chi_cat_table)
cat("Cramér's V: ~0.1 small, ~0.3 medium, ~0.5 large\n")


# Feature Engineering 
mortgage_fe <- mortgage_clean %>%
  mutate(
    loan_age   = time - orig_time,
    rate_delta = interest_rate_time - Interest_Rate_orig_time,
    hpi_change = hpi_time - hpi_orig_time
  )

cat("\n New Features Summary   \n")
summary(mortgage_fe[, c("loan_age", "rate_delta", "hpi_change")])

# Drop variables for collinearity / redundancy
drop_for_collinearity <- c(
  "balance_orig_time", # drop only orig balance, keep balance_time
  "LTV_orig_time",
  "hpi_orig_time"
)

drop_for_collinearity <- intersect(drop_for_collinearity, names(mortgage_fe))

mortgage_fe_red <- mortgage_fe %>%
  select(-all_of(drop_for_collinearity))

cat("\nDropped for collinearity / redundancy:\n")
print(drop_for_collinearity)

# Final predictor list (cleaned, de-collinear, no REtype, no hpi_time, no time)
core_predictors <- c(
  "FICO_orig_time",
  "LTV_time",
  "interest_rate_time",
  "rate_delta",
  "loan_age",
  "orig_time",
  "mat_time",
  "hpi_change",
  "uer_time",
  "gdp_time",
  "investor_orig_time",
  "balance_time"
)

candidate_predictors <- intersect(core_predictors, names(mortgage_fe_red))

cat("\nFinal candidate predictors (present in data):\n")
print(candidate_predictors)

# Build the final modeling dataframe 
id_cols <- intersect(c("id"), names(mortgage_fe_red))

modeling_df <- mortgage_fe_red %>%
  select(
    any_of(id_cols),
    status_time,
    status_lbl,
    all_of(candidate_predictors)
  ) %>%
  filter(!is.na(status_lbl))

cat("\n Modeling dataset structure (pre-modeling, Last Obs per Customer)  \n")
glimpse(modeling_df)

# Correlation matrix / heatmap for numeric predictors
num_vars_model <- names(modeling_df)[
  sapply(modeling_df, is.numeric)
]

num_vars_model <- setdiff(num_vars_model, c("status_time"))

corr_input <- modeling_df %>%
  select(all_of(num_vars_model)) %>%
  tidyr::drop_na()

corr_mat <- cor(corr_input, use = "pairwise.complete.obs")

cat("\n Correlation matrix (numeric predictors in final set, Last Obs per Customer)  \n")
print(round(corr_mat, 3))

corr_long <- as.data.frame(as.table(corr_mat)) %>%
  dplyr::rename(
    var1 = Var1,
    var2 = Var2,
    corr = Freq
  )

ggplot(corr_long, aes(x = var1, y = var2, fill = corr)) +
  geom_tile() +
  scale_fill_gradient2(
    limits = c(-1, 1),
    midpoint = 0,
    oob = scales::squish
  ) +
  labs(
    title = "Correlation Heatmap: Final Numeric Predictors ",
    x = NULL,
    y = NULL,
    fill = "Corr"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 7),
    axis.text.y = element_text(size = 7)
  )

# VIF re-check on the FINAL numeric set
vif_input_final <- modeling_df %>%
  select(all_of(num_vars_model)) %>%
  tidyr::drop_na()

vif_table_final <- calc_vif_table(vif_input_final)

cat("\n VIF on FINAL numeric predictors (pre-modeling, Last Obs per Customer)  \n")
print(vif_table_final)

high_vif_final <- vif_table_final %>%
  dplyr::filter(!is.na(VIF) & VIF > 10) %>%
  dplyr::pull(variable)

cat("\nVariables still with VIF > 10:\n")
print(high_vif_final)

# MODELING
# Random 70/30 split + CLASS-BALANCED TRAIN
set.seed(123)

# TRAIN / TEST SPLIT RANDOM 70/30 (STRATIFIED BY OUTCOME)

train_index <- createDataPartition(
  y    = modeling_df$status_lbl,
  p    = 0.7,
  list = FALSE
)

train_df_raw <- modeling_df[train_index, ]
test_df      <- modeling_df[-train_index, ]

cat("Train rows (UNBALANCED):", nrow(train_df_raw), "\n")
cat("Test rows:", nrow(test_df), "\n")

cat("\nClass distribution in TRAIN (unbalanced, as-is):\n")
print(table(train_df_raw$status_lbl))

cat("\nClass distribution in TEST:\n")
print(table(test_df$status_lbl))

# CLASS BALANCING UPSAMPLE MINORITY CLASSES IN TRAIN ONLY
predictor_cols_train <- setdiff(names(train_df_raw), c("status_lbl", "status_time"))

set.seed(123)
train_df_bal <- upSample(
  x     = train_df_raw[, predictor_cols_train, drop = FALSE],
  y     = train_df_raw$status_lbl,
  yname = "status_lbl"
)

cat("\nTrain rows AFTER balancing:", nrow(train_df_bal), "\n")
cat("Balanced class distribution in TRAIN:\n")
print(table(train_df_bal$status_lbl))

# function for Default recall (for all models)
default_recall <- function(pred, truth) {
  truth_bin <- factor(ifelse(truth == "Default", "Default", "Other"),
                      levels = c("Other", "Default"))
  pred_bin  <- factor(ifelse(pred  == "Default", "Default", "Other"),
                      levels = c("Other", "Default"))
  sensitivity(pred_bin, truth_bin, positive = "Default")
}

# MULTINOMIAL LOGISTIC REGRESSION
logit_predictors <- intersect(core_predictors, names(train_df_bal))

cat("\nLogistic predictor set used:\n")
print(logit_predictors)

logit_formula <- as.formula(
  paste("status_lbl ~", paste(logit_predictors, collapse = " + "))
)

cat("\n Training Multinomial Logistic Regression:  \n")
logit_fit <- multinom(logit_formula, data = train_df_bal, trace = FALSE)

logit_summary <- summary(logit_fit)
print(logit_summary)

# p-values for coefficients (for interpretation only)
z_vals <- logit_summary$coefficients / logit_summary$standard.errors
p_vals <- 2 * (1 - pnorm(abs(z_vals)))

cat("\n Logistic Regression p-values  \n")
print(round(p_vals, 4))

# Test-set predictions + metrics
logit_pred <- predict(logit_fit, newdata = test_df, type = "class")
logit_cm   <- confusionMatrix(logit_pred, test_df$status_lbl)

cat("\n Confusion Matrix: Multinomial Logistic   \n")
print(logit_cm)

logit_default_rec <- default_recall(logit_pred, test_df$status_lbl)
cat("\nDefault recall (Logistic):", round(logit_default_rec, 4), "\n")
cat("Accuracy (Logistic):", round(logit_cm$overall["Accuracy"], 4), "\n")

# MULTINOMIAL LOGISTIC REGRESSION WITH LASSO (glmnet)

cat("\n TRAINING MULTINOMIAL LOGISTIC REGRESSION (LASSO, BALANCED TRAIN, Last Obs per Customer)  \n")

y_train <- as.integer(train_df_bal$status_lbl) - 1
y_test  <- as.integer(test_df$status_lbl) - 1

x_train <- model.matrix(~ . - 1,
                        data = train_df_bal[, candidate_predictors, drop = FALSE])
x_test  <- model.matrix(~ . - 1,
                        data = test_df[, candidate_predictors, drop = FALSE])

set.seed(123)

cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "multinomial",
  alpha = 1,
  type.measure = "class",
  nfolds = 5,
  parallel = FALSE
)

cat("\nBest lambda (lambda.min):", cv_lasso$lambda.min, "\n")
cat("1-SE lambda (lambda.1se):", cv_lasso$lambda.1se, "\n")

lasso_fit <- glmnet(
  x = x_train,
  y = y_train,
  family = "multinomial",
  alpha = 1,
  lambda = cv_lasso$lambda.min
)

cat("\n LASSO model trained  \n")

cat("\n Non-zero Coefficients (LASSO Feature Selection)  \n")
print(lasso_fit$beta)

prob_pred <- predict(lasso_fit, newx = x_test, type = "response")[,,1]

lasso_pred_class <- apply(prob_pred, 1, function(row) which.max(row) - 1)
lasso_pred_lbl <- factor(
  lasso_pred_class,
  levels = 0:2,
  labels = levels(modeling_df$status_lbl)
)

lasso_cm <- confusionMatrix(lasso_pred_lbl, test_df$status_lbl)

cat("\n CONFUSION MATRIX: LASSO Logistic Regression   \n") 
print(lasso_cm)

lasso_default_recall <- default_recall(lasso_pred_lbl, test_df$status_lbl)

cat("\nDefault Recall (LASSO):", round(lasso_default_recall, 4), "\n")
cat("Accuracy (LASSO):", round(lasso_cm$overall["Accuracy"], 4), "\n")

# RANDOM FOREST MODEL
cat("\n Training Random Forest (BALANCED TRAIN, FULL predictors, Last Obs per Customer)  \n")

rf_train_df <- train_df_bal
max_rows_rf <- 120000
if (nrow(rf_train_df) > max_rows_rf) {
  frac_rf <- max_rows_rf / nrow(rf_train_df)
  rf_train_df <- rf_train_df %>%
    group_by(status_lbl) %>%
    sample_frac(size = min(1, frac_rf), replace = FALSE) %>%
    ungroup()
}

cat("RF training subset rows:", nrow(rf_train_df), "\n")
print(table(rf_train_df$status_lbl))

rf_formula <- as.formula(
  paste("status_lbl ~", paste(candidate_predictors, collapse = " + "))
)

rf_fit <- randomForest(
  rf_formula,
  data       = rf_train_df,
  ntree      = 200,
  mtry       = floor(sqrt(length(candidate_predictors))),
  importance = TRUE
)

cat("\nRandom Forest trained. Summary:\n")
print(rf_fit)

rf_pred <- predict(rf_fit, newdata = test_df, type = "class")
rf_cm   <- confusionMatrix(rf_pred, test_df$status_lbl)

cat("\n RF Confusion Matrix   \n")
print(rf_cm)

rf_default_recall <- default_recall(rf_pred, test_df$status_lbl)
cat("\nDefault recall (RF):", round(rf_default_recall, 4), "\n")
cat("Accuracy (RF):", round(rf_cm$overall["Accuracy"], 4), "\n")

rf_imp <- importance(rf_fit)
rf_imp_df <- data.frame(
  variable = rownames(rf_imp),
  MeanDecreaseGini = rf_imp[, "MeanDecreaseGini"]
) %>%
  arrange(desc(MeanDecreaseGini))

cat("\n RF Variable Importance (MeanDecreaseGini)  \n")
print(rf_imp_df)

# RANDOM FOREST VARIABLE IMPORTANCE PLOT
rf_imp_plot <- rf_imp_df %>%
  slice_max(MeanDecreaseGini, n = 10) %>%
  mutate(variable = reorder(variable, MeanDecreaseGini))

ggplot(rf_imp_plot, aes(x = MeanDecreaseGini, y = variable)) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Random Forest Variable Importance (MeanDecreaseGini)",
    x     = "Importance (MeanDecreaseGini)",
    y     = "Feature"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 8),
    plot.title  = element_text(face = "bold")
  )


# XGBOOST MODEL
cat("\n Training XGBoost (FULL predictors, BALANCED TRAIN, NO EXTRA CLASS WEIGHTS, Last Obs per Customer)  \n")

train_y <- as.integer(train_df_bal$status_lbl) - 1
test_y  <- as.integer(test_df$status_lbl) - 1

xgb_predictors <- candidate_predictors

train_x <- model.matrix(~ . - 1,
                        data = train_df_bal[, xgb_predictors, drop = FALSE])
test_x  <- model.matrix(~ . - 1,
                        data = test_df[, xgb_predictors, drop = FALSE])

dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest  <- xgb.DMatrix(data = test_x,  label = test_y)

xgb_params <- list(
  objective   = "multi:softprob",
  num_class   = length(levels(modeling_df$status_lbl)),
  eval_metric = "mlogloss",
  eta         = 0.1,
  max_depth   = 6
)

xgb_fit <- xgb.train(
  params    = xgb_params,
  data      = dtrain,
  nrounds   = 200,
  watchlist = list(train = dtrain),
  verbose   = 0
)

cat("\nXGBoost model trained.\n")

xgb_prob <- predict(xgb_fit, newdata = dtest)

n_class <- length(levels(modeling_df$status_lbl))
xgb_mat <- matrix(xgb_prob, ncol = n_class, byrow = TRUE)

xgb_class <- max.col(xgb_mat, ties.method = "first")

xgb_pred <- factor(
  xgb_class,
  levels = 1:n_class,
  labels = levels(modeling_df$status_lbl)
)

xgb_cm <- confusionMatrix(xgb_pred, test_df$status_lbl)
cat("\n XGBoost Confusion Matrix   \n")
print(xgb_cm)

xgb_default_recall <- default_recall(xgb_pred, test_df$status_lbl)
cat("\nDefault recall (XGBoost):", round(xgb_default_recall, 4), "\n")
cat("Accuracy (XGBoost):", round(xgb_cm$overall["Accuracy"], 4), "\n")

xgb_imp <- xgb.importance(model = xgb_fit)
cat("\n XGBoost Variable Importance  \n")
print(xgb_imp)


# ROC CURVES Default vs Non-Default (All Models, Test Set)
# Binary ground truth
truth_default <- ifelse(test_df$status_lbl == "Default", 1, 0)

# index of "Default" in factor levels
default_idx <- which(levels(modeling_df$status_lbl) == "Default")

# Multinomial Logistic Regression
logit_prob_mat <- predict(logit_fit, newdata = test_df, type = "probs")
logit_prob_default <- logit_prob_mat[, "Default"]

# LASSO
prob_array_lasso <- predict(lasso_fit, newx = x_test, type = "response")
prob_lasso_mat   <- prob_array_lasso[, , 1]
lasso_prob_default <- prob_lasso_mat[, default_idx]

# Random Forest
rf_prob_mat      <- predict(rf_fit, newdata = test_df, type = "prob")
rf_prob_default  <- rf_prob_mat[, "Default"]

# XGBoost
xgb_prob_all <- predict(xgb_fit, newdata = dtest)
n_class      <- length(levels(modeling_df$status_lbl))
xgb_mat2     <- matrix(xgb_prob_all, ncol = n_class, byrow = TRUE)
xgb_prob_default <- xgb_mat2[, default_idx]

# ROC objects & AUCs
roc_logit <- roc(truth_default, logit_prob_default, quiet = TRUE)
roc_rf    <- roc(truth_default, rf_prob_default,    quiet = TRUE)
roc_xgb   <- roc(truth_default, xgb_prob_default,   quiet = TRUE)
roc_lasso <- roc(truth_default, lasso_prob_default, quiet = TRUE)


auc_logit <- auc(roc_logit)
auc_rf    <- auc(roc_rf)
auc_xgb   <- auc(roc_xgb)
auc_lasso <- auc(roc_lasso)

cat("\n AUC (Default vs Non-Default, Test Set)  \n")
cat("Logistic:", round(auc_logit, 3), "\n")
cat("RF      :", round(auc_rf,    3), "\n")
cat("XGBoost :", round(auc_xgb,   3), "\n")
cat("LASSO   :", round(auc_lasso, 3), "\n")

# Combined ROC Plot
plot(
  roc_logit,
  col  = "blue",
  lwd  = 2,
  main = "ROC Curves (Default vs Non-Default) Test Set"
)

lines(roc_rf,    col = "red",    lwd = 2)
lines(roc_xgb,   col = "darkgreen", lwd = 2)
lines(roc_lasso, col = "purple", lwd = 2)

abline(a = 0, b = 1, lty = 2, col = "grey50")

legend(
  "bottomright",
  legend = c(
    paste0("Logistic (AUC = ", round(auc_logit, 3), ")"),
    paste0("Random Forest (AUC = ", round(auc_rf, 3), ")"),
    paste0("XGBoost (AUC = ", round(auc_xgb, 3), ")"),
    paste0("LASSO (AUC = ", round(auc_lasso, 3), ")")
  ),
  col = c("blue", "red", "darkgreen", "purple"),
  lwd = 2,
  cex = 0.8
)

# K-MEANS CLUSTERING UNSUPERVISED SEGMENTATION 
set.seed(123)

cluster_vars <- intersect(candidate_predictors, names(modeling_df))

cat("\nCluster input variables:\n")
print(cluster_vars)

cluster_input <- modeling_df %>%
  dplyr::select(all_of(cluster_vars)) %>%
  tidyr::drop_na()

cat("\nDimensions of clustering input (rows x vars): ",
    paste(dim(cluster_input), collapse = " x "), "\n")

cluster_scaled <- scale(cluster_input)

# Save scaling params
scale_center <- attr(cluster_scaled, "scaled:center")
scale_scale  <- attr(cluster_scaled, "scaled:scale")

k <- 3
km_fit <- kmeans(
  x        = cluster_scaled,
  centers  = k,
  nstart   = 25,
  iter.max = 100
)

cat("\n K-means clustering results (k = 3)  \n")
cat("Cluster sizes:\n")
print(km_fit$size)
cat("\nTotal within-cluster SSE:", km_fit$tot.withinss, "\n")
cat("Between-cluster SSE:", km_fit$betweenss, "\n")

complete_idx <- complete.cases(modeling_df[, cluster_vars])
modeling_df_km <- modeling_df[complete_idx, , drop = FALSE] %>%
  dplyr::mutate(cluster = factor(km_fit$cluster))

# Cluster Centers in ORIGINAL UNITS
centers_scaled   <- km_fit$centers
centers_unscaled <- sweep(centers_scaled, 2, scale_scale, "*")
centers_unscaled <- sweep(centers_unscaled, 2, scale_center, "+")

cluster_centers_df <- as.data.frame(centers_unscaled)
cluster_centers_df$cluster <- factor(1:k)
cluster_centers_df <- dplyr::relocate(cluster_centers_df, cluster)

cluster_centers_df_print <- cluster_centers_df %>%
  dplyr::mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("\n Cluster Feature Profiles (Means in Original Units)  \n")
print(cluster_centers_df_print)

# Empirical Cluster Profiles (Actual Mean of Members)
cluster_profile_empirical <- modeling_df_km %>%
  dplyr::group_by(cluster) %>%
  dplyr::summarise(
    dplyr::across(all_of(cluster_vars), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

cluster_profile_empirical_print <- cluster_profile_empirical %>%
  dplyr::mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("\n Empirical Cluster Profiles (Means from Actual Members)  \n")
print(cluster_profile_empirical_print)

# Cluster vs Loan Outcome (Current / Default / Payoff)
cat("\n Cluster vs Outcome Count Table  \n")
print(table(modeling_df_km$cluster, modeling_df_km$status_lbl))

cluster_outcome_summary <- modeling_df_km %>%
  dplyr::group_by(cluster) %>%
  dplyr::summarise(
    n            = dplyr::n(),
    current_rate = mean(status_lbl == "Current"),
    default_rate = mean(status_lbl == "Default"),
    payoff_rate  = mean(status_lbl == "Payoff"),
    .groups = "drop"
  )

cluster_outcome_summary_print <- cluster_outcome_summary %>%
  dplyr::mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("\n Cluster Outcome Rates (Current / Default / Payoff)  \n")
print(cluster_outcome_summary_print)

# Cluster Members (ID Lists)
if ("id" %in% names(modeling_df_km)) {
  cluster_members_list <- split(modeling_df_km$id, modeling_df_km$cluster)
  
  cat("\n Members per Cluster (Count Only)  \n")
  print(sapply(cluster_members_list, length))
}

# Cluster Profile Plot (Heatmap of Feature Means per Cluster)
cluster_profile_long <- cluster_centers_df %>%
  tidyr::pivot_longer(
    cols = -cluster,
    names_to = "feature",
    values_to = "mean_value"
  ) %>%
  dplyr::group_by(feature) %>%
  dplyr::mutate(
    mean_z = (mean_value - mean(mean_value, na.rm = TRUE)) /
      sd(mean_value,  na.rm = TRUE)
  ) %>%
  dplyr::ungroup()

ggplot(cluster_profile_long,
       aes(x = feature, y = cluster, fill = mean_z)) +
  geom_tile() +
  scale_fill_gradient2(
    name   = "Z-score",
    low    = "blue",
    mid    = "white",
    high   = "red",
    midpoint = 0
  ) +
  labs(
    title = "Cluster Feature Profiles (Standardized Means)",
    x     = "Feature",
    y     = "Cluster"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8),
    axis.text.y = element_text(size = 9)
  )

ggplot(cluster_profile_long,
       aes(x = feature, y = mean_z, group = cluster, color = cluster)) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    title = "Cluster Profiles (Standardized Feature Means)",
    x     = "Feature",
    y     = "Standardized Mean (Z-score)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8)
  )

