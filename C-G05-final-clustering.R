# Fully Automated Clustering Pipeline with Silhouette Evaluation

#-------------------- Libraries ---------------------------
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(GGally)) install.packages("GGally")
if(!require(e1071)) install.packages("e1071")
if(!require(caret)) install.packages("caret")
if(!require(cluster)) install.packages("cluster")  # for silhouette
if(!require(factoextra)) install.packages("factoextra") # for visualization

library(ggplot2)
library(GGally)
library(e1071)
library(caret)
library(cluster)
library(factoextra)

#-------------------- A. Data Collection ------------------
url <- "https://drive.google.com/uc?export=download&id=1M7Q-6x-JpmOaUPNAP5vGoM6wA3No_BOv"
data <- read.csv(url)
cat("Rows:", nrow(data), "Columns:", ncol(data), "\n")

#-------------------- B. Identify Column Types -------------
numeric_cols <- names(data)[sapply(data, is.numeric)]
categorical_cols <- names(data)[sapply(data, is.factor) | sapply(data, is.character)]

#-------------------- C. EDA -------------------------------
# Histograms
for (col in numeric_cols) {
  print(
    ggplot(data, aes_string(x = col)) +
      geom_histogram(bins = 30, fill = "skyblue", color = "black") +
      theme_minimal() +
      labs(title = paste("Histogram of", col))
  )
}

# Boxplots
for (col in numeric_cols) {
  print(
    ggplot(data, aes_string(y = col)) +
      geom_boxplot(fill = "tomato") +
      theme_minimal() +
      labs(title = paste("Boxplot of", col))
  )
}

# Scatterplot matrix
pairs(data[, numeric_cols], main = "Scatterplots of Numeric Features", pch = 19, col = "steelblue")

# Correlation
cor_matrix <- cor(data[, numeric_cols], use = "complete.obs")
print(cor_matrix)
ggcorr(data[, numeric_cols], label = TRUE)

# Missing values
missing_values <- sapply(data, function(x) sum(is.na(x)))
print(missing_values)
cat("Total missing values:", sum(missing_values), "\n")

#-------------------- D. Data Preprocessing ----------------
data_clean <- na.omit(data)

# Detect binary numeric columns
numeric_cols <- names(data_clean)[sapply(data_clean, is.numeric)]
binary_numeric <- numeric_cols[sapply(data_clean[numeric_cols], function(x) length(unique(x)) == 2)]
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

#-------------------- E. Automatic Target Removal ----------
possible_targets <- names(data_norm)[sapply(data_norm, function(x) length(unique(x)) <= 2)]
cluster_data <- data_norm
cluster_data[possible_targets] <- NULL
cluster_data <- cluster_data[, sapply(cluster_data, is.numeric)]
cluster_data <- cluster_data[, sapply(cluster_data, function(x) all(is.finite(x)))]
cluster_data <- cluster_data[, apply(cluster_data, 2, var) != 0]
cat("Final clustering features:", ncol(cluster_data), "\n")

#-------------------- F. Scaling ---------------------------
cluster_scaled <- scale(cluster_data)

#-------------------- G. Elbow Method ----------------------
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(cluster_scaled, centers = k, nstart = 25)$tot.withinss
})
plot(1:10, wss, type = "b", pch = 19,
     xlab = "Number of Clusters (k)",
     ylab = "Within-Cluster Sum of Squares",
     main = "Elbow Method")

#-------------------- H. K-Means Clustering ----------------
set.seed(123)
optimal_k <- 3
kmeans_model <- kmeans(cluster_scaled, centers = optimal_k, nstart = 25)
data_norm$Cluster <- as.factor(kmeans_model$cluster)
print(table(data_norm$Cluster))

#-------------------- I. Silhouette Score ------------------
sil <- silhouette(kmeans_model$cluster, dist(cluster_scaled))
avg_sil <- mean(sil[, 3])
cat("\nAverage Silhouette Width:", round(avg_sil, 3), "\n")
# Interpretation:
# Silhouette score close to 1 -> well-separated clusters
# Close to 0 -> overlapping clusters
# Negative -> misclassified points

# Silhouette plot
fviz_silhouette(sil)

#-------------------- J. PCA Visualization -----------------
pca_res <- prcomp(cluster_scaled)
pca_df <- data.frame(
  PC1 = pca_res$x[,1],
  PC2 = pca_res$x[,2],
  Cluster = data_norm$Cluster
)
ggplot(pca_df, aes(PC1, PC2, color = Cluster)) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "K-Means Clustering Visualization (PCA)")

#-------------------- K. Cluster Summary -------------------
cluster_summary <- aggregate(cluster_data, by = list(Cluster = data_norm$Cluster), mean)
print(cluster_summary)
