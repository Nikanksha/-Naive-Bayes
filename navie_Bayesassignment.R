# read the sms data into the sms data frame
#sms_raw <- read.csv('/Volumes/Data/Course Content/DS content/NB and Knn/NB/sms_spam.csv', stringsAsFactors = FALSE)
sms_raw <- sms_raw_NB
# examine the structure of the sms data
str(sms_raw)
# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)

# build a corpus using the text mining (tm) package
library(tm)
sms_corpus <- Corpus(VectorSource(sms_raw$text))

# clean up the corpus using tm_map()
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
inspect(corpus_clean)
# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(corpus_clean)
# creating training and test datasets
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test  <- sms_raw[4170:5559, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test  <- corpus_clean[4170:5559]
#From Srinivas ExcelR to Everyone:  06:02 PM
# check that the proportion of spam is similar
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

# word cloud visualization
library(wordcloud)

wordcloud(sms_corpus_train, min.freq = 30, random.order = FALSE)

# subset the training data into spam and ham groups
spam <- subset(sms_raw_train, type == "spam")
ham  <- subset(sms_raw_train, type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5),colors = 'blue')
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# indicator features for frequent words
sms_dict<-findFreqTerms(sms_dtm_train, 3)
#sms_dict <- Dictionary(findFreqTerms(sms_dtm_train, 5))
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))

# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2, convert_counts)
#From Srinivas ExcelR to Everyone:  06:02 PM
## Step 3: Training a model on the data ----
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)


## Step 4: Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
## Step 5: Improving model performance ----
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

 
Total Observations in Table:  1390 

 
             | actual 
   predicted |       ham |      spam | Row Total | 
-------------|-----------|-----------|-----------|
         ham |      1205 |        30 |      1235 | 
             |     0.998 |     0.164 |           | 
-------------|-----------|-----------|-----------|
        spam |         2 |       153 |       155 | 
             |     0.002 |     0.836 |           | 
-------------|-----------|-----------|-----------|
Column Total |      1207 |       183 |      1390 | 
             |     0.868 |     0.132 |           | 
-------------|-----------|-----------|-----------|
