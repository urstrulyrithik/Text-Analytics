# Text Analytics and Networking Project

## Overview

This project involves extensive text mining and social network analysis using R. The objective was to analyze the book "Tarzan of the Apes" by identifying the most significant words and sentences, performing sentiment analysis, and applying topic modeling techniques to uncover latent topics. The project also included visualizations such as word clouds and sentiment trajectories to enhance understanding of the text's emotional flow.

The main steps taken in this project included:
1. Corpus Creation and Preprocessing
2. Text Analysis (e.g., identifying longest words and sentences)
3. Sentiment Analysis using multiple methods (Syuzhet, Bing, NRC)
4. Term Document and Document Term Matrix Creation
5. Topic Modeling using Latent Dirichlet Allocation (LDA) and Correlated Topic Model (CTM)
6. Visualization of Word Frequencies and Sentiments

## Technologies Used
- **Programming Language**: R
- **Libraries/Packages**: `tm`, `stringr`, `tidyverse`, `tidytext`, `wordcloud`, `quanteda`, `syuzhet`, `topicmodels`
- **Visualization**: Word clouds, dendrograms, sentiment plots

## Installation
To run the code in this project, make sure you have R installed along with the necessary packages. You can use the `pacman` package to easily load and install them:

```r
if (!require(pacman, quietly = TRUE)) {
  install.packages("pacman", dependencies = TRUE)
  library(pacman)
}

# Use pacman to load and install necessary packages
p_load("tm", "stringr", "tidyverse", "tidytext", "wordcloud", "quanteda", "syuzhet", "topicmodels")
```

## Project Structure
- **Corpus Creation and Preprocessing**: Created a corpus of text from the book and preprocessed it by removing numbers, punctuation, and stopwords, and converting it to lowercase.
- **Text Analysis**: Extracted key metrics such as the longest words and sentences from each chapter of the book.
- **Sentiment Analysis**: Utilized three different methods (Syuzhet, Bing, and NRC) to compute sentiment scores for the text and visualized the emotional trajectory.
- **Topic Modeling**: Performed LDA and CTM to uncover latent topics in the text.
- **Visualization**: Created word clouds to show the most common words and comparison clouds to visualize differences across documents.

## Results
- **Sentiment Analysis**: Provided detailed insights into the emotional flow of "Tarzan of the Apes" using sentiment trajectory plots.
- **Topic Modeling**: Identified the key topics discussed in the book using LDA and CTM, visualized with document-topic distributions.
- **Visualization**: Word clouds, commonality clouds, and comparison clouds provided clear insights into word frequencies and relationships.

## Running the Code
The project can be run as an R script, and all the outputs including visualizations, topic models, and sentiment plots are generated within the R environment. To create a reproducible environment:
1. Clone the repository.
2. Run the R script to perform text analytics and sentiment analysis.
3. The results will be displayed in the console and graphical windows.

```bash
git clone https://github.com/urstrulyrithik/Text-Analytics
cd Text-Analytics-Networking-Project
```

## Example Usage
- **Corpus Preprocessing**: Cleaned text for further analysis
- **Sentiment Analysis**: Plotting sentiment using Syuzhet, Bing, and NRC methods to compare overall emotional trends throughout the book.

## Visualizations
- **Word Cloud**: A word cloud was generated to visualize the most frequent words in the corpus, highlighting key terms from the book.
- **Dendrogram**: Clustered documents based on their word usage to understand the similarities in content.
- **Sentiment Plot**: A sentiment trajectory plot showcasing the rise and fall in emotional content throughout the book.

## Contributing
Feel free to contribute to the project by opening issues and pull requests. Any additional features, optimizations, or error fixes are always welcome!

## License
This project is licensed under the MIT License. Feel free to use it for educational or research purposes.

## Contact
For any questions, feel free to reach out at:
- **Email**: urstrulyrithik@gmail.com
- **LinkedIn**: [Rithik Reddy](https://www.linkedin.com/in/rithikreddypv)
- **GitHub**: [rithikreddy](https://github.com/urstrulyrithik)
