#load libraries
library(randomForest)
library(ISLR)
library(gbm)
library(glmnet)
library(xgboost)
library(e1071)
library(tidyverse)
library(tm)
library(text2vec)
library(SnowballC)
library(vip)
library(caret)
library(class)

#load data files
train_x <- read_csv("ks_training_X.csv") %>%
  mutate(original_TR = 1)
train_y <- read_csv("ks_training_y.csv")
test_x <- read_csv("ks_test_X.csv") %>%
  mutate(original_TR = 0)

#rbind binds rows
total <- rbind(train_x, test_x)

#join the training y to the training x file
#also turn two of the target variables into factors
total <- total %>%
  left_join(train_y, by = "id")

load("38050-0003-Data.rda")

da38050.0003 <- 
  da38050.0003 %>%
  distinct(PID, .keep_all = TRUE)

total <- total %>%
  left_join(da38050.0003, by = c(id = "PID"))

summary(total)

accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

find_length <- function(ch){
  ifelse(is.na(ch) | (sum(is.na(as.numeric(strsplit(ch, ",")[[1]]))) > 0), 0, length(as.numeric(strsplit(ch, ",")[[1]])))
}

find_min <- function(ch){
  ifelse(is.na(ch) | (sum(is.na(as.numeric(strsplit(ch, ",")[[1]]))) > 0), NA, min(as.numeric(strsplit(ch, ",")[[1]])))
}

find_max <- function(ch){
  ifelse(is.na(ch) | (sum(is.na(as.numeric(strsplit(ch, ",")[[1]]))) > 0), NA, max(as.numeric(strsplit(ch, ",")[[1]])))
}

find_sd <- function(ch){
  ifelse(is.na(ch) | (sum(is.na(as.numeric(strsplit(ch, ",")[[1]]))) > 0), NA, sd(as.numeric(strsplit(ch, ",")[[1]])))
}

total$num_rewards <- as.numeric(lapply(total$reward_amounts, find_length))
total$min_reward <- as.numeric(lapply(total$reward_amounts, find_min))
total$max_reward <- as.numeric(lapply(total$reward_amounts, find_max))
total$sd_reward <- as.numeric(lapply(total$reward_amounts, find_sd))


# EXAMPLE PREDICTIONS FOR Y = BACKERS_COUNT

#create a simple model to predict backers_count and generate predictions in the test data

total_success <- total %>%
  select(success, goal, id, name, blurb, captions, tag_names, USER_LOCATION_COUNTRY, USER_LOCATION_STATE,
         region, category_parent, num_words, category_name, avg_wordlengths, deadline, launched_at, created_at,
         grade_level, minage_creator, location_slug, smiling_creator, sentence_counter, minage_project,
         ADV, NOUN, ADP, PRT, DET, PRON, VERB, NUM, CONJ, ADJ, male_creator, female_creator, numfaces_project,
         afinn_pos, afinn_neg, original_TR, contains_youtube, numfaces_creator, smiling_project,
         num_rewards, min_reward, max_reward, sd_reward, maxage_creator, avgsentencelength, reward_descriptions) %>%
  mutate(region = as.factor(region),
         category_parent = as.factor(category_parent)) %>%
  mutate(deadline = as.Date(deadline),
         created_at = as.Date(created_at),
         launched_at = as.Date(launched_at),
         time_gap = deadline - launched_at) %>%
  group_by(location_slug) %>%
  mutate(location_slug_freq = n()) %>%
  ungroup() %>%
  mutate(location_slug = ifelse(location_slug_freq < 1500, 'Other Location Slug', location_slug),
         location_slug = ifelse(is.na(location_slug), 'Other Location Slug', location_slug),
         location_slug = as.factor(location_slug)) %>%
  group_by(USER_LOCATION_COUNTRY) %>%
  mutate(location_country_freq = n()) %>%
  ungroup() %>%
  mutate(USER_LOCATION_COUNTRY = as.character(USER_LOCATION_COUNTRY),
         USER_LOCATION_COUNTRY = ifelse(location_country_freq < 10 | USER_LOCATION_COUNTRY == "  " | is.na(USER_LOCATION_COUNTRY), 'Other Country', USER_LOCATION_COUNTRY),
         USER_LOCATION_COUNTRY = as.factor(USER_LOCATION_COUNTRY)) %>%
  group_by(USER_LOCATION_STATE) %>%
  mutate(location_state_freq = n()) %>%
  ungroup() %>%
  mutate(USER_LOCATION_STATE = as.character(USER_LOCATION_STATE),
         USER_LOCATION_STATE = ifelse(location_state_freq < 100 | USER_LOCATION_STATE == " " | is.na(USER_LOCATION_STATE), 'Other State', USER_LOCATION_STATE),
         USER_LOCATION_STATE = as.factor(USER_LOCATION_STATE)) %>%
  group_by(category_name) %>%
  mutate(category_name_freq = n()) %>%
  ungroup() %>%
  mutate(category_name = ifelse(category_name_freq < 3000, 'Other Category', category_name),
         category_name = ifelse(is.na(category_name), 'Other Category', category_name),
         category_name = as.factor(category_name)) %>%
  mutate(afinn_overall = afinn_pos - afinn_neg,
         afinn_overall = ifelse(is.na(afinn_overall), 0, afinn_overall),
         success = ifelse(success == "YES", 1, 0),
         time_gap = as.numeric(time_gap),
         proj_duration = as.numeric(launched_at - created_at),
         day_of_week = as.factor(as.character(format(launched_at, format="%A"))),
         launched_at = as.numeric(launched_at),
         reward_descriptions = ifelse(is.na(reward_descriptions), "None", reward_descriptions),
         blurb = ifelse(is.na(blurb), "None", blurb),
         name = ifelse(is.na(name), "None", name), 
         captions = ifelse(is.na(captions), "None", captions),
         tag_names = ifelse(is.na(tag_names), "None", tag_names),
         extra_female_creators = female_creator - male_creator,
         contains_youtube = as.factor(contains_youtube)) %>%
  select(-c(category_name_freq, location_slug_freq))

summary(total_success)

formula <- success ~ USER_LOCATION_STATE+USER_LOCATION_COUNTRY+goal+region+category_parent+num_words+grade_level+minage_creator+avg_wordlengths+time_gap+location_slug+category_name+afinn_overall+contains_youtube+avgsentencelength+numfaces_creator+sentence_counter+ADV+NOUN+ADP+PRT+DET+VERB+CONJ+num_rewards+min_reward+sd_reward+max_reward+extra_female_creators+numfaces_project+minage_project+smiling_creator+proj_duration:goal+category_parent:goal+maxage_creator:goal+proj_duration:numfaces_creator+num_words:proj_duration+contains_youtube:goal+goal:max_reward+grade_level:maxage_creator+afinn_overall:category_parent+grade_level:category_parent+day_of_week

labeled_data <- total_success %>%
  filter(original_TR == 1)

use_data <- total_success %>%
  filter(original_TR == 0) 

original_TR_indices <- which(as.logical(total_success$original_TR))

# Text Featurization

cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(stopwords(kind="en")) %>% #remove stopwords
    stemDocument %>%
    word_tokenizer 
}

prep_fun = tolower
tok_fun = cleaning_tokenizer

# Reward Description Column

it_reward = itoken(total_success$reward_descriptions,
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun, 
                   ids = total_success$id, 
                   progressbar = FALSE)

stop_words_reward = c("None", "will", "copi", "plus", "receiv", "get", "one", "includ", "ship", "special", "also", "print", "will_receiv", "download", "edit", "x", "choic", "list", "us", "well", "can", "add", "two", "limit", "youll", "limit_edit")
vocab_reward = create_vocabulary(it_reward, ngram = c(1L, 2L), stopwords = stop_words_reward)

vocab_reward <- prune_vocabulary(vocab_reward, term_count_min = 20, doc_count_min = 10)

vectorizer_reward = vocab_vectorizer(vocab_reward)

dtm_reward = create_dtm(it_reward, vectorizer_reward)

# Name column

it_name = itoken(total_success$name,
                 preprocessor = prep_fun, 
                 tokenizer = tok_fun, 
                 ids = total_success$id, 
                 progressbar = FALSE)

stop_words_name = c("None", "new", "short", "help", "make", "one", "live", "collect", "get")

vocab_name = create_vocabulary(it_name, ngram = c(1L, 2L), stopwords = stop_words_name)

vocab_name <- prune_vocabulary(vocab_name, term_count_min = 20, doc_count_min = 10)

vectorizer_name = vocab_vectorizer(vocab_name)

dtm_name = create_dtm(it_name, vectorizer_name)

# Blurb column

it_blurb = itoken(total_success$blurb,
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = total_success$id, 
                  progressbar = FALSE)
stop_words_blurb <- c("None", "help", "new", "one", "short", "two", "like", "start", "build", "come", "look", "go", "made", "will", "make", "need", "creat", "us", "get", "can", "live", "want", "bring", "work", "take", "use", "play")
vocab_blurb = create_vocabulary(it_blurb, ngram = c(1L, 2L), stopwords = stop_words_blurb)

vocab_blurb <- prune_vocabulary(vocab_blurb, term_count_min = 20, doc_count_min = 10)

vectorizer_blurb = vocab_vectorizer(vocab_blurb)

dtm_blurb = create_dtm(it_blurb, vectorizer_blurb)

# Captions column

it_captions = itoken(total_success$captions,
                     preprocessor = prep_fun, 
                     tokenizer = tok_fun, 
                     ids = total_success$id, 
                     progressbar = FALSE)

stop_words_captions <- c("none", "close", "front", "white", "black", "red", "blue", "larg", "green", "yellow", "next")
vocab_captions = create_vocabulary(it_captions, ngram = c(1L, 2L))

vocab_captions <- prune_vocabulary(vocab_captions, term_count_min = 20, doc_count_min = 10)

vectorizer_captions = vocab_vectorizer(vocab_captions)

dtm_captions = create_dtm(it_captions, vectorizer_captions)

# Tag names column

it_tag_names = itoken(total_success$tag_names,
                      preprocessor = prep_fun, 
                      tokenizer = tok_fun, 
                      ids = total_success$id, 
                      progressbar = FALSE)

stop_words_tag_names <- c("none", "white", "fast", "black")
vocab_tag_names = create_vocabulary(it_tag_names, ngram = c(1L, 2L))

vocab_tag_names <- prune_vocabulary(vocab_tag_names, term_count_min = 20, doc_count_min = 10)

vectorizer_tag_names = vocab_vectorizer(vocab_tag_names)

dtm_tag_names = create_dtm(it_tag_names, vectorizer_tag_names)


labeled_data_matrix <- model.matrix(formula, model.frame(~ ., labeled_data, na.action=na.pass)) 
labeled_data_matrix <- cbind(labeled_data_matrix, dtm_reward[original_TR_indices,], dtm_name[original_TR_indices,], dtm_blurb[original_TR_indices,], dtm_captions[original_TR_indices,])

use_data_matrix <- model.matrix(formula,model.frame(~ ., use_data, na.action=na.pass))
use_data_matrix <- cbind(use_data_matrix, dtm_reward[-original_TR_indices,], dtm_name[-original_TR_indices,], dtm_blurb[-original_TR_indices,], dtm_captions[-original_TR_indices,])

set.seed(1)
training_indices <- sample(nrow(labeled_data_matrix), .7*nrow(labeled_data_matrix))
tr_x <- labeled_data_matrix[training_indices,]
va_x <- labeled_data_matrix[-training_indices,]

tr_y <- labeled_data[training_indices,]$success
va_y <- labeled_data[-training_indices,]$success



bst <- xgboost(data = tr_x, label = tr_y, max.depth = 4, eta = 0.2, nrounds = 900,  objective = "binary:logistic")

bst_pred <- predict(bst, va_x, type='response')
bst_classifications <- ifelse(bst_pred > 0.5, 1, 0)
bst_acc <- mean(ifelse(bst_classifications == va_y, 1, 0))

bst_acc

# Predict for use data
bst_pred_use <- predict(bst, use_data_matrix, type='response')
preds_class <- ifelse(bst_pred_use>.5,"YES","NO")

#output your predictions
write.table(preds_class, "success_group26.csv", row.names = FALSE)
