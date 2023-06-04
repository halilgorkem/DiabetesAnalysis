
diabetes_data <- read.csv("diabetes.csv")

head(diabetes_data)

str(diabetes_data)

summary(diabetes_data)

set.seed(123)
train_indices <- sample(1:nrow(diabetes_data), 0.7 * nrow(diabetes_data))
train_data <- diabetes_data[train_indices, ]
test_data <- diabetes_data[-train_indices, ]

# Korelasyon matrisi
cor_matrix <- cor(train_data[, -9])
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7, tl.col = "black")

# Şeker hastası ve sağlıklı bireylerin dağılımı
ggplot(train_data, aes(x = Outcome)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(x = "Outcome", y = "Count", title = "Histogram - Outcome")

library(glmnet)

# Bağımlı değişkeni ve bağımsız değişkenleri ayırma
y_train <- train_data$Outcome
x_train <- train_data[, -9]

# Veriyi standartlaştırma
x_train <- scale(x_train)

# Lojistik regresyon modelini oluşturma
model_logit <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
library(e1071)

# Destek vektör makineleri modelini oluşturma
model_svm <- svm(Outcome ~ ., data = train_data, kernel = "linear", probability = TRUE)
# Test veri seti üzerinde tahmin yapma
x_test <- test_data[, -9]

# Lojistik regresyon ile tahmin yapma
x_test_scaled <- scale(x_test)
predicted_logit <- predict(model_logit, newx = x_test_scaled, type = "response")

# Destek vektör makineleri ile tahmin yapma
predicted_svm <- predict(model_svm, newdata = x_test, probability = TRUE, type = "response")

# predicted_logit uzunluğunu test_data$Outcome uzunluğuna ayarla
predicted_logit <- predicted_logit[1:length(test_data$Outcome)]

library(pROC)
roc_logit <- roc(test_data$Outcome, predicted_logit)
roc_svm <- roc(test_data$Outcome, predicted_svm)

plot(roc_logit, col = "blue", main = "ROC Curve Comparison")
lines(roc_svm, col = "red")
legend("bottomright", legend = c("Logistic Regression", "Support Vector Machines"), col = c("blue", "red"), lty = 1)

auc_logit <- auc(roc_logit)
auc_svm <- auc(roc_svm)

print(paste("Logistic Regression AUC:", auc_logit))
print(paste("Support Vector Machines AUC:", auc_svm))


