# FULLY AUTOMATED CLASSIFICATION PIPELINE
# ================== Package Auto-Installer ==================
required_packages <- c(
  "ggplot2",
  "GGally",
  "e1071",
  "caret",
  "rpart",
  "rpart.plot",
  "randomForest"
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
library(rpart)
library(rpart.plot)
library(randomForest)
# ============================================================


# ================== A. Data Collection ======================
url <- "https://drive.google.com/uc?export=download&id=1M7Q-6x-JpmOaUPNAP5vGoM6wA3No_BOv"

safe_read <- function(url) {
  tryCatch({
    read.csv(url)
  }, error = function(e) {
    stop("Dataset load failed. Check URL or internet.")
  })
}

data <- safe_read(url)

cat("Rows:", nrow(data), "Columns:", ncol(data), "\n")
# ============================================================


# ================== AUTO TARGET DETECTION ===================
detect_target <- function(df) {
  candidates <- names(df)[sapply(df, function(x) {
    (is.numeric(x) && length(unique(x)) == 2) ||
      (is.factor(x) && nlevels(x) == 2)
  })]
  
  if (length(candidates) == 0) {
    stop("Binary target column not found")
  }
  
  return(candidates[1])
}

target_col <- detect_target(data)
cat("Detected target column:", target_col, "\n")

data[[target_col]] <- as.factor(data[[target_col]])
# ============================================================


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

if (interactive()) {
  ggpairs(
    data,
    columns = numeric_cols,
    aes(color = data[[target_col]])
  )
}

cor_matrix <- cor(data[, numeric_cols], use = "complete.obs")
print(cor_matrix)

missing_values <- sapply(data, function(x) sum(is.na(x)))
print(missing_values)

cat("EDA completed\n")
# ============================================================


# ================== C. Data Preprocessing ===================
data_clean <- na.omit(data)

binary_numeric <- numeric_cols[
  sapply(data[numeric_cols], function(x) length(unique(x)) == 2)
]
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
categorical_cols <- names(data_clean)[
  sapply(data_clean, is.character) | sapply(data_clean, is.factor)
]

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
binary_numeric <- numeric_cols[
  sapply(data_clean[numeric_cols], function(x) length(unique(x)) == 2)
]
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
# ============================================================


# ================== D. Feature Engineering ==================
if ("Age" %in% names(data_norm)) {
  data_norm$AgeGroup <- as.numeric(
    cut(data_norm$Age,
        breaks = c(0, 40, 60, 100),
        labels = c("Young", "Middle", "Old"))
  ) - 1
}

num_features <- setdiff(
  names(data_norm)[sapply(data_norm, is.numeric)],
  target_col
)

data_norm$RiskScore <- rowSums(data_norm[, num_features], na.rm = TRUE)

cat("Feature engineering completed\n")
# ============================================================


# ================== Train–Test Split ========================
set.seed(123)

train_index <- createDataPartition(
  data_norm[[target_col]],
  p = 0.7,
  list = FALSE
)

modeldata <- data_norm[train_index, ]
testdata  <- data_norm[-train_index, ]

cat("Train rows:", nrow(modeldata), "\n")
cat("Test rows :", nrow(testdata), "\n")
# ============================================================


# ================== Modeling ================================
formula <- as.formula(paste(target_col, "~ ."))

# -------- Decision Tree --------
cat("\n--- Decision Tree ---\n")
model_dt <- rpart(formula, data = modeldata, method = "class")

if (interactive()) {
  rpart.plot(model_dt)
}

pred_dt <- predict(model_dt, testdata, type = "class")

conf_dt <- confusionMatrix(pred_dt, testdata[[target_col]])
print(conf_dt)
cat("Decision Tree Accuracy:", conf_dt$overall["Accuracy"], "\n")

# -------- Logistic Regression --------
cat("\n--- Logistic Regression ---\n")
model_log <- glm(formula, data = modeldata, family = binomial)

prob_log <- predict(model_log, testdata, type = "response")
pred_log <- as.factor(ifelse(prob_log > 0.5, 1, 0))

conf_log <- confusionMatrix(pred_log, testdata[[target_col]])
print(conf_log)
cat("Logistic Regression Accuracy:", conf_log$overall["Accuracy"], "\n")

# ================== Model Evaluation & Interpretation ==================

# Function to calculate F1-score
f1_score <- function(precision, recall) {
  if (precision + recall == 0) return(0)
  return(2 * precision * recall / (precision + recall))
}

# ----- Decision Tree Metrics -----
cat("\n=== Decision Tree Evaluation ===\n")

# Confusion Matrix already computed: conf_dt
dt_cm <- conf_dt$table
print(dt_cm)  # Show confusion matrix

# Calculate metrics
dt_accuracy  <- conf_dt$overall["Accuracy"]
dt_precision <- conf_dt$byClass["Precision"]
dt_recall    <- conf_dt$byClass["Recall"]
dt_f1        <- f1_score(dt_precision, dt_recall)

cat(sprintf("Accuracy : %.3f\n", dt_accuracy))
cat(sprintf("Precision: %.3f\n", dt_precision))
cat(sprintf("Recall   : %.3f\n", dt_recall))
cat(sprintf("F1-Score : %.3f\n", dt_f1))

# Plot confusion matrix
library(ggplot2)
cm_df <- as.data.frame(as.table(dt_cm))
ggplot(cm_df, aes(Prediction, Reference, fill=Freq)) +
  geom_tile() +
  geom_text(aes(label=Freq), color="white", size=5) +
  scale_fill_gradient(low="steelblue", high="red") +
  theme_minimal() +
  labs(title="Decision Tree Confusion Matrix")


# ----- Logistic Regression Metrics -----
cat("\n=== Logistic Regression Evaluation ===\n")
log_cm <- conf_log$table
print(log_cm)  # Show confusion matrix

# Calculate metrics
log_accuracy  <- conf_log$overall["Accuracy"]
log_precision <- conf_log$byClass["Precision"]
log_recall    <- conf_log$byClass["Recall"]
log_f1        <- f1_score(log_precision, log_recall)

cat(sprintf("Accuracy : %.3f\n", log_accuracy))
cat(sprintf("Precision: %.3f\n", log_precision))
cat(sprintf("Recall   : %.3f\n", log_recall))
cat(sprintf("F1-Score : %.3f\n", log_f1))

# Plot confusion matrix
cm_df_log <- as.data.frame(as.table(log_cm))
ggplot(cm_df_log, aes(Prediction, Reference, fill=Freq)) +
  geom_tile() +
  geom_text(aes(label=Freq), color="white", size=5) +
  scale_fill_gradient(low="steelblue", high="red") +
  theme_minimal() +
  labs(title="Logistic Regression Confusion Matrix")



