# FULLY AUTOMATED REGRESSION PIPELINE WITH EVALUATION
# ================== Package Auto-Installer ==================
required_packages <- c(
  "ggplot2",
  "GGally",
  "e1071",
  "caret",
  "Metrics"
)

installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

library(ggplot2)
library(GGally)
library(e1071)
library(caret)
library(Metrics)

# ================== A. Data Collection ======================
url <- "https://drive.google.com/uc?export=download&id=1M7Q-6x-JpmOaUPNAP5vGoM6wA3No_BOv"

safe_read <- function(url) {
  tryCatch({
    read.csv(url)
  }, error = function(e) {
    stop("Dataset load failed. Check internet or URL.")
  })
}
data <- safe_read(url)
cat("Rows:", nrow(data), "Columns:", ncol(data), "\n")

# ================== AUTO TARGET DETECTION ===================
detect_regression_target <- function(df) {
  numeric_cols <- names(df)[sapply(df, is.numeric)]
  candidates <- numeric_cols[sapply(df[numeric_cols], function(x) length(unique(x)) > 5)]
  if (length(candidates) == 0) stop("Continuous numeric target not found")
  return(candidates[1])
}
target_col <- detect_regression_target(data)
cat("Detected regression target:", target_col, "\n")

# ================== B. EDA =================================
numeric_cols <- names(data)[sapply(data, is.numeric)]
if (interactive()) {
  for (col in numeric_cols) {
    print(
      ggplot(data, aes_string(x = col)) +
        geom_histogram(bins = 30, fill = "skyblue") +
        theme_minimal() +
        labs(title = paste("Histogram of", col))
    )
  }
}
if (interactive()) ggpairs(data, columns = numeric_cols)
cor_matrix <- cor(data[, numeric_cols], use = "complete.obs")
print(cor_matrix)
missing_values <- sapply(data, function(x) sum(is.na(x)))
print(missing_values)
cat("EDA completed\n")

# ================== C. Preprocessing =======================
data_clean <- na.omit(data)
binary_numeric <- numeric_cols[sapply(data[numeric_cols], function(x) length(unique(x)) == 2)]
numeric_for_outlier <- setdiff(numeric_cols, binary_numeric)

# Outlier capping
for (col in numeric_for_outlier) {
  Q1 <- quantile(data_clean[[col]], 0.25)
  Q3 <- quantile(data_clean[[col]], 0.75)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  data_clean[[col]][data_clean[[col]] < lower] <- lower
  data_clean[[col]][data_clean[[col]] > upper] <- upper
}

# Encode categorical variables
categorical_cols <- names(data_clean)[sapply(data_clean, is.character) | sapply(data_clean, is.factor)]
for (col in categorical_cols) {
  if (col == target_col) next
  if (length(unique(data_clean[[col]])) == 2) {
    data_clean[[col]] <- as.numeric(factor(data_clean[[col]])) - 1
  } else {
    dummies <- model.matrix(~ . -1, data = data_clean[col])
    data_clean <- cbind(data_clean, dummies)
    data_clean[[col]] <- NULL
  }
}

# Skewness reduction
numeric_cols <- names(data_clean)[sapply(data_clean, is.numeric)]
binary_numeric <- numeric_cols[sapply(data_clean[numeric_cols], function(x) length(unique(x)) == 2)]
numeric_skew <- setdiff(numeric_cols, binary_numeric)
skew_vals <- sapply(data_clean[numeric_skew], skewness)
for (col in names(skew_vals[abs(skew_vals) > 1])) {
  data_clean[[col]] <- log1p(data_clean[[col]])
}

# Normalization
data_norm <- data_clean
for (col in numeric_skew) {
  data_norm[[col]] <- (data_clean[[col]] - min(data_clean[[col]])) /
    (max(data_clean[[col]]) - min(data_clean[[col]]))
}
cat("Preprocessing completed\n")

# ================== D. Feature Engineering ==================
if ("Age" %in% names(data_norm)) {
  data_norm$AgeGroup <- as.numeric(
    cut(data_norm$Age, breaks = c(0, 40, 60, 100), labels = c("Young", "Middle", "Old"))
  ) - 1
}
cat("Feature engineering completed\n")

# ================== Train–Test Split ========================
set.seed(123)
sample_index <- sample(1:nrow(data_norm), size = 0.7 * nrow(data_norm))
modeldata <- data_norm[sample_index, ]
testdata  <- data_norm[-sample_index, ]
cat("Train rows:", nrow(modeldata), "\n")
cat("Test rows :", nrow(testdata), "\n")

# ================== Regression Modeling =====================
formula <- as.formula(paste(target_col, "~ ."))
lm_model <- lm(formula, data = modeldata)
summary(lm_model)

pred_lm <- predict(lm_model, newdata = testdata)

# ================== Evaluation ==============================
rmse_val <- rmse(testdata[[target_col]], pred_lm)
mae_val  <- mae(testdata[[target_col]], pred_lm)
sst <- sum((testdata[[target_col]] - mean(testdata[[target_col]]))^2)
sse <- sum((testdata[[target_col]] - pred_lm)^2)
r_squared <- 1 - (sse / sst)

cat("\nRegression Performance Metrics\n")
cat("RMSE :", round(rmse_val, 3), "\n")
cat("MAE  :", round(mae_val, 3), "\n")
cat("R^2  :", round(r_squared, 3), "\n")

# Interpretation comments:
# RMSE shows the average error magnitude; lower is better.
# MAE shows the mean absolute difference; lower is better.
# R² shows variance explained by the model; closer to 1 is better.

# ================== Plot ==============================
if (interactive()) {
  ggplot(
    data.frame(Actual = testdata[[target_col]], Predicted = pred_lm),
    aes(x = Actual, y = Predicted)
  ) +
    geom_point(color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(title = "Actual vs Predicted")
}
