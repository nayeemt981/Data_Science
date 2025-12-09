#..........................Data Science Project Group-05(MID)......................

#A. Data Understanding 
install.packages("ggplot2")
install.packages("GGally")
install.packages("FSelectorRcpp")


#Download and loading dada
url <- "https://drive.google.com/uc?export=download&id=1M7Q-6x-JpmOaUPNAP5vGoM6wA3No_BOv"
data <- read.csv(url)
head(data,50)

library(ggplot2)

#Shape of the dataset
cat("Number of rows:", nrow(data))
cat("Number of colums:", ncol(data))
#Types
str(data)
summary(data)
# Numeric columns select
numeric_cols <- sapply(data, is.numeric)

# Basic descriptive statistics
summary_stats <- data.frame(
  Mean =  sapply(data[, numeric_cols], mean, na.rm = TRUE),
  Median = sapply(data[, numeric_cols], median, na.rm = TRUE),
  SD  = sapply(data[, numeric_cols], sd, na.rm = TRUE),
  Min  = sapply(data[, numeric_cols], min, na.rm = TRUE),
  Max  = sapply(data[, numeric_cols], max, na.rm = TRUE),
  Count  = sapply(data[, numeric_cols], function(x) sum(!is.na(x)))
)

summary_stats

get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

mode_values <- sapply(data[, numeric_cols], get_mode)
mode_values

#Identifying Colums
# Numerical columns
numerical_features <- names(data)[sapply(data, is.numeric)]

# Categorical columns
categorical_features <- names(data)[sapply(data, is.character) | sapply(data, is.factor)]

numerical_features
categorical_features

#B. Data Exploration & Visualization .............
numeric_cols <- names(data)[sapply(data, is.numeric)]
print(numeric_cols)

library(ggplot2)

# Histogram for age
numeric_cols <- names(data)[sapply(data, is.numeric)]

for (col in numeric_cols) {
  p <- ggplot(data, aes_string(x = col)) +
    geom_histogram(bins = 30, fill = "skyblue", color = "black") +
    labs(title = paste("Histogram of", col),
         x = col,
         y = "Frequency") +
    theme_minimal()
  print(p)
}


# Boxplot for Cholesterol

ggplot(data, aes(x = "", y = Cholesterol)) +
  geom_boxplot(fill = "tomato", outlier.shape = 16, outlier.color = "black", outlier.size = 2) +
  labs(
    title = "Boxplot of Cholesterol ",
    x = "", 
    y = "Cholesterol"
  )


for (col in categorical_features) {
  p <- ggplot(data, aes_string(x = col)) +
    geom_bar(fill = "purple") +
    labs(title = paste("Bar Chart of", col),
         x = col, y = "Count") +
    theme_minimal()
  print(p)
}


# Frecuency of all categorical features
for (col in categorical_features) {
  
  cat("Frequency for'", col, "' is: ")
  print(table(data[[col]]))
  cat("-------")
}


#B.2
#Correlation Heatmap
library("GGally")
ggcorr(data, label = TRUE)


# Numeric and categorical columns
numeric_cols <- names(data)[sapply(data, is.numeric)]

binary_numeric <- numeric_cols[sapply(data[numeric_cols], function(x) length(unique(x)) == 2)]
binary_numeric

#excluding binary columns
numeric_for_outlier <- setdiff(numeric_cols, binary_numeric)


categorical_cols <- names(data)[sapply(data, is.character) | sapply(data, is.factor)]


#Boxplots between categorical and numeric features 
for (cat_col in categorical_cols) {
  for (num_col in numeric_for_outlier) {   # <-- changed here
    p <- ggplot(data, aes_string(x = cat_col, y = num_col)) +
      geom_boxplot(fill = "skyblue", outlier.shape = 16, outlier.color = "red", outlier.size = 2) +
      labs(title = paste("Boxplot of", num_col, "by", cat_col), x = cat_col, y = num_col) +
      theme_minimal()
    print(p)
  }
}


# CORRELATION MATRIX and heatmap.....................................
numeric_cols <- names(data)[sapply(data, is.numeric)]
cor_matrix <- cor(data[, numeric_cols], use = "complete.obs")
print(cor_matrix)   # <-- NOW WILL SHOW OUTPUT
ggcorr(data, label = TRUE)

print(numeric_cols)
str(data[, numeric_cols])

pairs(data[, numeric_cols[ numeric_cols != "HeartDisease" ]],
      main = "Scatter Plots of Numeric Features",
      pch = 19, col = "skyblue")
# Pair Plot
ggpairs(data, columns = c("Age", "RestingBP", "Cholesterol", 
                          "MaxHR", "Oldpeak"), 
        aes(color = factor(HeartDisease)))

# SKEWNESS (ONLY for NON-BINARY Num Columns)
library(e1071)
skew_values <- sapply(data[, numeric_for_outlier], skewness, na.rm = TRUE)
print(skew_values)                      # <-- SHOW THIS

# Boxplots between categorical and numeric features
library(ggplot2)

# Loop through all categorical columns
for (cat_col in categorical_cols) {
  # Loop through numeric columns (excluding binary numeric features if desired)
  for (num_col in numeric_for_outlier) {
    p <- ggplot(data, aes_string(x = cat_col, y = num_col)) +
      geom_boxplot(fill = "skyblue", 
                   outlier.shape = 16, 
                   outlier.color = "red", 
                   outlier.size = 2) +
      labs(title = paste("Boxplot of", num_col, "by", cat_col),
           x = cat_col, y = num_col) +
      theme_minimal()
    print(p)
  }
}

# OUTLIER BOXPLOTS
library(ggplot2)
for (col in numeric_for_outlier) {
  p <- ggplot(data, aes_string(y = col)) +
    geom_boxplot(fill = "lightgreen",
                 outlier.shape = 16,
                 outlier.color = "red",
                 outlier.size = 2) +
    labs(title = paste("Boxplot of", col), y = col) +
    theme_minimal()
  print(p)
}

#C. Data Preprocessing 

#1.Detecting missing values
# Total missing values per column
missing_values <- sapply(data, function(x) sum(is.na(x)))
missing_values

#Total for data
total_missing <- sum(is.na(data))
cat("Number of missing values:",total_missing)

#Removing Missing values
cat("Number of rows:", nrow(data))
data_clean <- na.omit(data)
cat("Number of rows:", nrow(data_clean))


#2.  Handling Outlire

# OUTLIER COUNT
outlier_count <- sapply(data_clean[, numeric_for_outlier], function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  sum(x < (Q1 - 1.5*IQR) | x > (Q3 + 1.5*IQR), na.rm = TRUE)
})
print(outlier_count)

# CAPPING OUTLIERS
for (col in numeric_for_outlier) {
  Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  data[[col]][data[[col]] < lower_bound] <- lower_bound
  data[[col]][data[[col]] > upper_bound] <- upper_bound
}


#3. Data Conversion  
#Now considering "data_clean" as data


# Identify categorical columns
categorical_cols <- names(data_clean)[sapply(data_clean, is.character) | sapply(data_clean, is.factor)]

# Count unique values in each categorical column
unique_counts <- sapply(data_clean[categorical_cols], function(x) length(unique(x)))

# Binary categorical columns (2 unique values)
binary_cols <- names(unique_counts[unique_counts == 2])

# Multi-class categorical columns (more than 2 unique values)
multi_class_cols <- names(unique_counts[unique_counts > 2])

# Convert binary columns safely to 0/1
for (col in binary_cols) {
  # Make first unique value = 0, second = 1
  data_clean[[col]] <- ifelse(data_clean[[col]] == unique(data_clean[[col]])[1], 0, 1)
}

# One-hot encoding for multi-class columns
for (col in multi_class_cols) {
  dummies <- model.matrix(~.-1, data = data_clean[col])  # creating dummy variables
  data_clean <- cbind(data_clean, dummies)    # add dummies to dataset
  data_clean[[col]] <- NULL    # remove original column
}

# Verifying
str(data_clean)
all_numeric <- all(sapply(data_clean, is.numeric))
all_numeric  # TRUE if all features are now numeric


#4. Data Transformation 
# Identify numeric columns
numeric_cols <- names(data_clean)[sapply(data_clean, is.numeric)]

# Identify binary numeric columns
binary_numeric <- numeric_cols[sapply(data_clean[numeric_cols], function(x) length(unique(x)) == 2)]

# Numeric columns (exclude binary)
numeric_for_norm <- setdiff(numeric_cols, binary_numeric)

# Min-Max Normalization
data_norm <- data_clean
for (col in numeric_for_norm) {
  data_norm[[col]] <- (data_clean[[col]] - min(data_clean[[col]])) /
    (max(data_clean[[col]]) - min(data_clean[[col]]))
}

# Verifying ["data_norm" instead of "data_clean"]
str(data_norm)

#5. Feature Selection
#Targeted Feature= "HeartDisease"

#Variance Thresholding
apply(data_norm, 2, var)

#Feature Selection
variance_threshold <- 0.01 #not much spread

# numeric columns
numeric_cols <- names(data_norm)[sapply(data_norm, is.numeric)]

# calculating variance
feature_variances <- sapply(data_norm[numeric_cols], var)
print(feature_variances)

# high variance features 
selected_features <- names(feature_variances[feature_variances > variance_threshold])
selected_features

data_selected <- data_norm[, selected_features]

cat("Feature Selection applied using variance threshold of", variance_threshold, 
    ". Features with very low variance were removed. Retained features are:\n")
print(selected_features)

# Mutual Information / Information Gain 
library(FSelectorRcpp)

weights <- information_gain(HeartDisease ~ ., data = data_norm)

print(weights)


