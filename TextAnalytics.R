# Ensure the pacman package is installed and loaded
if (!require(pacman, quietly = TRUE)) {
  install.packages("pacman", dependencies = TRUE)
  library(pacman)
}

# Use pacman to load and install necessary packages
p_load("tm", "stringr", "tidyverse", "tidytext", "wordcloud", "quanteda", "syuzhet", "topicmodels")

message("All required packages have been successfully loaded.")

# Set the working directory to the project folder
setwd('/Users/rithik/Documents/3rd semester/Big data/Project 3/Working File')
getwd()

# Create a corpus from the main text directory
tarzan_corpus = VCorpus(DirSource(".", ignore.case = TRUE, mode = "text"))
str(tarzan_corpus)
tarzan_corpus

# Extract the content of the first document in the corpus
main_text = content(tarzan_corpus[[1]])
main_text

# Identify the indices where chapters begin
chapter_indices = str_which(main_text, pattern = "^Chapter\\s")
chapter_indices = chapter_indices[1:28]
chapter_indices

# Split the text into chapters based on the identified indices
chapters = list()
for (i in 1:(length(chapter_indices) - 1)) {
  chapters[[i]] = main_text[chapter_indices[i]: (chapter_indices[i + 1] - 1)]
}

# Create a directory to save individual chapter files
dir.create("chapters")
for (i in seq_along(chapters)) {
  chapter_path = file.path("chapters/", paste0("Chapter_", i, ".txt"))
  writeLines(chapters[[i]], chapter_path)
}

# Create a new corpus from the chapter files
tarzan_apes_corpus = VCorpus(DirSource("/Users/rithik/Documents/3rd semester/Big data/Project 3/Working File/chapters", ignore.case = TRUE, mode = "text"))
str(tarzan_apes_corpus)
tarzan_apes_corpus

# Transform the corpus into a tidy format for easier analysis
tidy_chapters = tidy(tarzan_apes_corpus)
tidy_chapters

# Extract the first chapter after tidying
first_chapter_text = tidy_chapters$text[1]

# Extract the first chapter ID
first_chapter_id_number = gsub("[^0-9]", "", tidy_chapters$id[1])
first_chapter_id_number

# Find the 10 longest words in the first chapter
words = str_extract_all(first_chapter_text, "\\w+") %>% unlist()
sorted_words = words[order(nchar(words), decreasing = TRUE)]
longest_unique_words = sorted_words %>% unique() %>% head(10)
longest_unique_words

# Split the text into sentences
sentences <- str_split(first_chapter_text, "\\.\\s") %>% unlist()
# Sort the sentences by length in non-increasing order
sorted_sentences <- sentences[order(nchar(sentences), decreasing = TRUE)]
# Remove duplicates and retrieve the top 10 longest sentences
longest_sentences_top10 <- sorted_sentences %>% unique() %>% head(10)
longest_sentences_top10

# Define a function to find the 10 longest words and sentences for all XXVII chapters
find_longest_words_and_sentences <- function(chapter_text, chapter_index) {
  # Extract chapter number
  chapter_number <- gsub("[^0-9]", "", tidy_chapters$id[chapter_index])
  # Find the 10 longest words
  words <- str_extract_all(chapter_text, "\\w+") %>% unlist()
  sorted_words <- words[order(nchar(words), decreasing = TRUE)]
  top10_longest_words <- sorted_words %>% unique() %>% head(10)
  
  # Find the 10 longest sentences
  sentences <- str_split(chapter_text, "\\.\\s") %>% unlist()
  sorted_sentences <- sentences[order(nchar(sentences), decreasing = TRUE)]
  top10_longest_sentences <- sorted_sentences %>% unique() %>% head(10)
  
  # Return a tibble with results
  return(tibble(Chapter = chapter_number,
                ItemType = rep(c("Word", "Sentence"), each = 10),
                Item = c(top10_longest_words, top10_longest_sentences),
                Length = c(nchar(top10_longest_words), nchar(top10_longest_sentences))))
}

# Retrieve and process text from each chapter file to generate a results table
chapter_files <- list.files(path = "/Users/rithik/Documents/3rd semester/Big data/Project 3/Working File/chapters", pattern = "\\.txt$")
chapter_texts <- vector("character", length(chapter_files))

# Read the contents of each text file
for (i in seq_along(chapter_files)) {
  chapter_texts[i] <- readr::read_file(file.path("/Users/rithik/Documents/3rd semester/Big data/Project 3/Working File/chapters", chapter_files[i]))
}

results_table <- map_dfr(seq_along(chapter_texts), ~find_longest_words_and_sentences(chapter_texts[.], .))
print(results_table, n = 600)

for (chapter in 1:28){
  cat("Chapter", chapter, "\n")
  results_table %>%
    filter(Chapter == chapter)%>%
    print(n=20)
}

# Generate a results table directly from the tidy corpus
results_table <- map_dfr(seq_along(tidy_chapters$text), ~find_longest_words_and_sentences(tidy_chapters$text[.], .))
print(results_table, n = 600)

# Generate a Document Term Matrix of the chapters corpus
tarzan_dtm = DocumentTermMatrix(tarzan_apes_corpus)
tarzan_dtm
inspect(tarzan_dtm)
str(tarzan_dtm)

# Generate a Term Document Matrix of the entire corpus
tarzan_tdm = TermDocumentMatrix(tarzan_apes_corpus)
tarzan_tdm
inspect(tarzan_tdm)
str(tarzan_tdm)

# Create a data frame from the main text
tarzan_text_df = data.frame(MainText = main_text)
tarzan_text_df

# Remove numbers from the main text
text_no_numbers = removeNumbers(main_text)
text_no_numbers

# Remove punctuation from the text that already had numbers removed
text_no_numbers_or_punctuation = removePunctuation(text_no_numbers)
text_no_numbers_or_punctuation

# Function to remove numbers and punctuation from each document in a VCorpus
remove_numbers_and_punctuation <- function(text) {
  gsub("[^[:alpha:][:space:]]*", "", text)
}

# Apply the removal function to the VCorpus
clean_corpus = tm::tm_map(tarzan_apes_corpus, content_transformer(remove_numbers_and_punctuation))
str(clean_corpus)

content(clean_corpus[[1]])

inspect(clean_corpus)

# Convert text in the cleaned corpus to lowercase
clean_corpus_lowercase = tm::tm_map(clean_corpus, content_transformer(tolower))
content(clean_corpus_lowercase[[1]])

# Create a Term Document Matrix from the cleaned and lowercased corpus
tdm_clean_lower = TermDocumentMatrix(clean_corpus_lowercase)
tdm_clean_lower
inspect(tdm_clean_lower)
str(tdm_clean_lower)

# Create a Document Term Matrix from the cleaned and lowercased corpus
dtm_clean_lower = DocumentTermMatrix(clean_corpus_lowercase)
dtm_clean_lower
inspect(dtm_clean_lower)
str(dtm_clean_lower)

# Convert TDM to matrices
matrix_tdm = as.matrix(tdm_clean_lower)
matrix_tdm

# Convert DTM to matrices
matrix_dtm = as.matrix(dtm_clean_lower)
matrix_dtm

# Remove English stop words from the lowercased corpus
stop_words = tm::stopwords(kind = 'en')
stop_words
clean_corpus_no_stopwords = tm::tm_map(clean_corpus_lowercase, removeWords, stop_words)
str(clean_corpus_no_stopwords)

#Inspecting First Chapter after removing stop words
tm::inspect(clean_corpus_no_stopwords[[1]])

# Create a TDM from the corpus without stop words
tdm_no_stopwords = TermDocumentMatrix(clean_corpus_no_stopwords)
tdm_no_stopwords
inspect(tdm_no_stopwords)
str(tdm_no_stopwords)

# Create a DTM from the corpus without stop words
dtm_no_stopwords = DocumentTermMatrix(clean_corpus_no_stopwords)
dtm_no_stopwords
inspect(dtm_no_stopwords)
str(dtm_no_stopwords)

# Find the frequency of words in the no stop words DTM
word_frequencies = colSums(as.matrix(dtm_no_stopwords))
word_frequencies

# Find the frequency of words in the no stop words TDM
frequent_terms = tm::findFreqTerms(tdm_no_stopwords)
frequent_terms

# Find the frequency of words in the no stop words corpus chapter-1
text_frequencies = tm::termFreq(clean_corpus_no_stopwords[[1]])
text_frequencies

#Dendrogram
# Convert the first document of the Term Document Matrix without stop words to a data frame
document_frequency_df <- as.data.frame(tdm_no_stopwords[[1]])

# Calculate the Euclidean distance matrix for the document
document_distance <- dist(document_frequency_df)

# Perform hierarchical clustering using Ward's D2 method
document_clustering <- hclust(document_distance, method="ward.D2")
str(document_clustering)
plot(document_clustering)

# Dendrogram after removing words with high sparsity
# Convert the TDM to a matrix
tdm_matrix <- as.matrix(tdm_no_stopwords)

# Set sparsity threshold (e.g., 0.9 means words that appear in less than 99% of the documents are kept)
sparsity_threshold <- 0.9

# Calculate sparsity for each word
sparsity_values <- rowSums(tdm_matrix > 0) / ncol(tdm_matrix)

# Select words with sparsity below the threshold
low_sparsity_words <- which(sparsity_values < sparsity_threshold)

# Create a new TDM with the selected low sparsity words
filtered_tdm <- tdm_matrix[low_sparsity_words,]

# Convert the filtered TDM to a data frame
filtered_document_df <- as.data.frame(filtered_tdm)

# Calculate the distance matrix
filtered_document_distance <- dist(filtered_document_df)

# Perform hierarchical clustering using Ward's method on the filtered data
filtered_document_clustering <- hclust(filtered_document_distance, method = "ward.D2")
str(filtered_document_clustering)
plot(filtered_document_clustering)

# Word Cloud
word_frequencies <- word_frequencies
color_palette <- brewer.pal(9, "Spectral")  # Define a color palette
word_cloud <- wordcloud(names(word_frequencies), word_frequencies, colors = color_palette)

# Quanteda for text tokenization
# Extract content from the first document of the cleaned corpus
document_content <- clean_corpus[[1]]$content

# Show the first 10 elements of content
head(document_content, 10)

# Tokenize the content
TarzanApesToken <- quanteda::tokens(document_content)
TarzanApesToken

# Convert tokens to a document-feature matrix using Quanteda
document_feature_matrix = quanteda::dfm(TarzanApesToken)
str(document_feature_matrix)

# Calculate frequency of words using Quanteda
word_frequencies = quanteda::docfreq(document_feature_matrix)
word_frequencies

# Apply weighting to the document-feature matrix
weighted_dfm = quanteda::dfm_weight(document_feature_matrix)
weighted_dfm
str(weighted_dfm)

# Calculate Term Frequency-Inverse Document Frequency (TF-IDF)
tfidf_matrix = quanteda::dfm_tfidf(document_feature_matrix, scheme_tf = "count")
tfidf_matrix

# Using the syuzhet package to analyze sentiment
# Extract content from the first document of the corpus
chapter_content = tarzan_apes_corpus[[1]]$content
chapter_content

# Load text from a specific chapter file for sentiment analysis
chapter_text = get_text_as_string("/Users/rithik/Documents/3rd semester/Big data/Project 3/Working File/chapters/Chapter_1.txt")
chapter_text

# Split the text into sentences
sentences_from_chapter = get_sentences(chapter_text)
sentences_from_chapter

# Compute sentiment using the 'syuzhet' method
chapter_sentiment_syuzhet = get_sentiment(sentences_from_chapter, "syuzhet")
chapter_sentiment_syuzhet

# Retrieve the sentiment dictionary used by syuzhet
syuzhet_dictionary = get_sentiment_dictionary("syuzhet")
syuzhet_dictionary

# Calculate and summarize overall sentiment using syuzhet
sentiment_sum_syuzhet = sum(chapter_sentiment_syuzhet)
sentiment_sum_syuzhet
sentiment_mean_syuzhet = mean(chapter_sentiment_syuzhet)
sentiment_mean_syuzhet
summary(chapter_sentiment_syuzhet)
plot(chapter_sentiment_syuzhet, main = "Tarzan Apes Sentiment Trajectory using Syuzhet", xlab = "Narrative", ylab = "Emotional Valence")

# Retrieve and use the 'bing' sentiment dictionary
bing_dictionary = get_sentiment_dictionary("bing")
bing_dictionary

# Compute sentiment using the 'bing' method
chapter_sentiment_bing = get_sentiment(sentences_from_chapter, "bing")

# Calculate and summarize overall sentiment using Bing
sentiment_sum_bing = sum(chapter_sentiment_bing)
sentiment_sum_bing
sentiment_mean_bing = mean(chapter_sentiment_bing)
sentiment_mean_bing
summary(chapter_sentiment_bing)
plot(chapter_sentiment_bing, main = "Tarzan Apes Sentiment Trajectory using Bing", xlab = "Narrative", ylab = "Emotional Valence")

# Retrieve and use the 'nrc' sentiment dictionary of syuzhet
nrc_dictionary = get_sentiment_dictionary("nrc")
nrc_dictionary

# Compute sentiment using the 'nrc' method
chapter_sentiment_nrc = get_sentiment(sentences_from_chapter, "nrc")
chapter_sentiment_nrc

# Calculate and summarize overall sentiment using NRC
sentiment_sum_nrc = sum(chapter_sentiment_nrc)
sentiment_sum_nrc
sentiment_mean_nrc = mean(chapter_sentiment_nrc)
sentiment_mean_nrc 
summary(chapter_sentiment_nrc)
plot(chapter_sentiment_nrc, main = "Tarzan Apes Sentiment Trajectory using NRC of Syuzhet", xlab = "Narrative", ylab = "Emotional Valence")

# Compute NRC sentiment using tidytext for provided sentences
tarzan_apes_nrc_sentiment <- get_nrc_sentiment(sentences_from_chapter)
tarzan_apes_nrc_sentiment

# Define a function to compute sentiment using multiple sentiment analysis methods
compute_sentiment <- function(text) {
  # Extract sentences from the provided text
  sentences <- get_sentences(text)
  
  # Calculate sentiment using the Syuzhet method
  syuzhet_sentiments <- get_sentiment(sentences, "syuzhet")
  total_syuzhet_sentiment <- sum(syuzhet_sentiments)
  average_syuzhet_sentiment <- mean(syuzhet_sentiments)
  
  # Calculate sentiment using the Bing method
  bing_sentiments <- get_sentiment(sentences, "bing")
  total_bing_sentiment <- sum(bing_sentiments)
  average_bing_sentiment <- mean(bing_sentiments)
  
  # Calculate sentiment using the NRC method
  nrc_sentiments <- get_sentiment(sentences, "nrc")
  total_nrc_sentiment <- sum(nrc_sentiments)
  average_nrc_sentiment <- mean(nrc_sentiments)
  
  # Return a list of sentiment summaries for each method
  list(syuzhetSum = total_syuzhet_sentiment, syuzhetMean = average_syuzhet_sentiment,
       bingSum = total_bing_sentiment, bingMean = average_bing_sentiment,
       nrcSum = total_nrc_sentiment, nrcMean = average_nrc_sentiment)
}

# Retrieve the list of chapter files from the specified directory
chapter_files <- list.files(path = "/Users/rithik/Documents/3rd semester/Big data/Project 3/Working File/chapters/", pattern = "Chapter_.*\\.txt$", full.names = TRUE)

# Compute sentiment for each chapter using the compute_sentiment function
sentiment_results <- lapply(chapter_files, function(file) {
  chapter_text <- get_text_as_string(file)
  compute_sentiment(chapter_text)
})

# Prepare and display a summary table of sentiment analysis results
sentiment_summary_table <- tibble(
  Chapter = basename(chapter_files),
  Syuzhet_Total = sapply(sentiment_results, `[[`, "syuzhetSum"),
  Syuzhet_Average = sapply(sentiment_results, `[[`, "syuzhetMean"),
  Bing_Total = sapply(sentiment_results, `[[`, "bingSum"),
  Bing_Average = sapply(sentiment_results, `[[`, "bingMean"),
  NRC_Total = sapply(sentiment_results, `[[`, "nrcSum"),
  NRC_Average = sapply(sentiment_results, `[[`, "nrcMean")
)

# Print the sentiment summary table
print(sentiment_summary_table)

# Analyze sentiment distribution in 10 bins
tarzan_apes_sentiment_10_bins <- get_percentage_values(chapter_sentiment_syuzhet, bins=10)
tarzan_apes_sentiment_10_bins
structure(tarzan_apes_sentiment_10_bins)
plot(tarzan_apes_sentiment_10_bins, main = "Tarzan Apes Sentiment Analysis in 10 Bins", xlab = "Narrative", ylab = "Emotional Valence", col='red')

# Analyze sentiment distribution in 20 bins
tarzan_apes_sentiment_20_bins <- get_percentage_values(chapter_sentiment_syuzhet, bins=20)
tarzan_apes_sentiment_20_bins
structure(tarzan_apes_sentiment_20_bins)
plot(tarzan_apes_sentiment_20_bins, main = "Tarzan Apes Sentiment Analysis in 20 Bins", xlab = "Narrative", ylab = "Emotional Valence", col='red')

# Topic Modeling: Latent Dirichlet Allocation (LDA)
# Conduct LDA to identify latent topics within the corpus
lda_tarzan_apes <- topicmodels::LDA(dtm_no_stopwords, k = 10)
# Extract and print the topic-word distribution
topic_word_distribution_lda <- as.data.frame(lda_tarzan_apes@beta)
print(topic_word_distribution_lda)
# Extract and print the document-topic distribution
document_topic_distribution_lda <- as.data.frame(lda_tarzan_apes@gamma)
print(document_topic_distribution_lda)

# Topic Modeling: Correlated Topic Model (CTM)
# Conduct CTM to identify correlated topics within the corpus
ctm_tarzan_apes <- CTM(dtm_no_stopwords, k = 2)
# Extract and print the topic-word distribution
topic_word_distribution_ctm <- as.data.frame(ctm_tarzan_apes@beta)
print(topic_word_distribution_ctm)
# Extract and print the document-topic distribution
document_topic_distribution_ctm <- as.data.frame(ctm_tarzan_apes@gamma)
print(document_topic_distribution_ctm)

# Word Cloud Visualization
# Create a word cloud to visualize word frequencies
word_cloud_palette <- brewer.pal(11, "Spectral")
tarzan_apes_word_cloud <- wordcloud(names(word_frequencies), word_frequencies, colors = word_cloud_palette)

# Define colors for document groups in comparison and commonality clouds
comparison_colors <- brewer.pal(6, "Set1")
# Create a matrix from the Term Document Matrix for visualization
tarzan_apes_tdm_matrix <- as.matrix(tdm_no_stopwords)
tarzan_apes_tdm_matrix

# Comparison Cloud: Visualize word frequency differences across documents
comparison.cloud(tarzan_apes_tdm_matrix, colors = comparison_colors, scale = c(4, 0.5), random.order = FALSE, title.size = 0.9)

# Commonality Cloud: Visualize common words across multiple documents
commonality.cloud(tarzan_apes_tdm_matrix, colors = comparison_colors, scale = c(4, 0.5), random.order = FALSE)

# Text Mining Transformations
# Apply Term Frequency weighting to the Document Term Matrix
tarzan_apes_dtm_tf <- weightTf(dtm_no_stopwords)
inspect(tarzan_apes_dtm_tf)
# Apply TF-IDF weighting to the Document Term Matrix
tarzan_apes_dtm_tfidf <- weightTfIdf(dtm_no_stopwords)
inspect(tarzan_apes_dtm_tfidf)

# Word Stemming: Reduce words to their root form
tarzan_apes_stemmed_words <- tm_map(clean_corpus_no_stopwords, stemDocument)
inspect(tarzan_apes_stemmed_words)
lapply(tarzan_apes_stemmed_words, as.character)

