# Veri setini yükleme
diabetes_data <- read.csv("diabetes.csv")
# Veri setinin ilk birkaç gözlemine bakma
head(diabetes_data)

# Veri setinin yapısal özelliklerini kontrol etme
str(diabetes_data)

# Veri setinin özet istatistiklerini görüntüleme
summary(diabetes_data)
# Eğitim ve test veri setlerini oluşturma
set.seed(123) # Tekrarlanabilirlik için seed ayarı
train_indices <- sample(1:nrow(diabetes_data), 0.7 * nrow(diabetes_data))
train_data <- diabetes_data[train_indices, ]
test_data <- diabetes_data[-train_indices, ]
# Lojistik regresyon modelini oluşturma
library(glmnet)

# Bağımlı değişkeni ve bağımsız değişkenleri ayırma
y_train <- train_data$Outcome
x_train <- train_data[, -9] # Outcome sütununu çıkar

# Veriyi standartlaştırma
x_train <- scale(x_train)

# Lojistik regresyon modelini oluşturma
model_logit <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
# Destek vektör makineleri modelini oluşturma
library(e1071)

# Destek vektör makineleri modelini oluşturma
model_svm <- svm(Outcome ~ ., data = train_data, kernel = "linear", probability = TRUE)
# Test veri seti üzerinde tahmin yapma
x_test <- test_data[, -9] # Outcome sütununu çıkar

# Lojistik regresyon ile tahmin yapma
x_test_scaled <- scale(x_test)
predicted_logit <- predict(model_logit, newx = x_test_scaled, type = "response")

# Destek vektör makineleri ile tahmin yapma
predicted_svm <- predict(model_svm, newdata = x_test, probability = TRUE, type = "response")

# predicted_logit uzunluğunu test_data$Outcome uzunluğuna ayarla
predicted_logit <- predicted_logit[1:length(test_data$Outcome)]

# ROC eğrilerini çizme
library(pROC)
roc_logit <- roc(test_data$Outcome, predicted_logit)
roc_svm <- roc(test_data$Outcome, predicted_svm)

plot(roc_logit, col = "blue", main = "ROC Curve Comparison")
lines(roc_svm, col = "red")
legend("bottomright", legend = c("Logistic Regression", "Support Vector Machines"), col = c("blue", "red"), lty = 1)

# AUC değerlerini elde etmek için
auc_logit <- auc(roc_logit)
auc_svm <- auc(roc_svm)

# Modellerin karşılaştırmasını yapmak için AUC değerlerini yazdır
print(paste("Logistic Regression AUC:", auc_logit))
print(paste("Support Vector Machines AUC:", auc_svm))

