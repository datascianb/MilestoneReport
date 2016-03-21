title: "MilestoneReportFinal"
author: "Anusha Bhat"
date: "March 20, 2016"
output: html_document
---

## Introduction

This Milestone Report is the first step toward completing the Capstone Project. The report starts with understanding the data, analyzing a large corpus of text documents, discovering and structure of the data. It shows methods used to clean and analyze text data. 

### Goal:

- Perform exploratory analysis of the data
- Understand the distribution of words and relationship between the words in the corpora
- Understand frequencies of words and word pairs

```{r libreq, echo=FALSE, results="hide", message=FALSE, warning=FALSE}

library(knitr)
library(downloader)
library(caret)
library(tools)
library(R.utils)
library(RCurl)
library(installr)
library(tm)
library(quanteda)
library(NLP)
library(stylo)
library(qdap)
library(openNLP)
library(RWeka)
library(ggplot2)

```

## Reading and Sampling

As the data provided is pretty large, only a part of the each dataset (first 1000 lines for each blog, news and twitter data) is loaded and used as sample data. 

```{r loadsample, message=FALSE, warning=FALSE}

filenamesUS <- list.files(pattern = "/*.txt", full.names = TRUE)

conBlog <- file(filenamesUS[1], "rb")
conNews <- file(filenamesUS[2], "rb")
conTwit <- file(filenamesUS[3], "rb")
conProf <- file(filenamesUS[4], "rb")

usBlog <- readLines(conBlog,encoding="latin1", n=1000)
usNews <- readLines(conNews,encoding="latin1", n=1000) 
usTwit <- readLines(conTwit,encoding="latin1", n=1000) 
usProf <- readLines(conProf,encoding="latin1") 

close(conBlog)
close(conNews)
close(conTwit)
close(conProf)

```


## Priliminary Analysis

The maximum and minimum characters in any given sentence is analyzed as a part of priliminary analysis. 

```{r charsize, message=FALSE, warning=FALSE}

hWordCntBlog <- max(nchar(usBlog), na.rm = TRUE)
lWordCntBlog <- min(nchar(usBlog), na.rm = TRUE)
hWordCntBlog
lWordCntBlog

hWordCntNews <- max(nchar(usNews), na.rm = TRUE)
lWordCntNews <- min(nchar(usNews), na.rm = TRUE)
hWordCntNews
lWordCntNews

hWordCntTwit <- max(nchar(usTwit), na.rm = TRUE)
lWordCntTwit <- min(nchar(usTwit), na.rm = TRUE)
hWordCntTwit
lWordCntTwit

```

The three datasets are then merged and tokenized as is to understand the data as a whole.

```{r wholedata, message=FALSE, warning=FALSE}

allText <- paste(usBlog, usNews, usTwit)
byLineText <- sent_detect(allText, language = "en", model = NULL)

textToken1 <- tokenize(toLower(allText), removePunct = TRUE,
                       removeNumbers = TRUE, removeTwitter = TRUE, ngrams = 1)
textToken2 <- tokenize(toLower(allText), removePunct = TRUE,
                        removeNumbers = TRUE, removeTwitter = TRUE, ngrams = 2)
textToken3 <- tokenize(toLower(allText), removePunct = TRUE,
                        removeNumbers = TRUE, removeTwitter = TRUE, ngrams = 3)

```

### Priliminary Results  

The complete dataset is analyzed by tokenization of raw data. Plots are drawn to understand the data and frequencies of words are plotted. The words appearing in bold and large fonts are seen more frequently in the data. Frequencies of 1-gram, 2-grams and 3-grams are depicted.

```{r predMod, message=FALSE, warning=FALSE}

dfmToken1 <- dfm(textToken1)
dim(dfmToken1)
plot(dfmToken1, min.freq = 1000, random.order = FALSE)

dfmToken2 <- dfm(textToken2)
dim(dfmToken2)
plot(dfmToken2, min.freq = 500, random.order = FALSE)

dfmToken3 <- dfm(textToken3)
dim(dfmToken3)
plot(dfmToken3, min.freq = 100, random.order = FALSE)

```


## Analysis

The corpus is created to clean the data.The data is all converted to lower case and plain text, cleaned for numbers, white spaces, punctuation marks, and certain patterns, for e.g. accounts and websites.

```{r corpus, message=FALSE, warning=FALSE}

sourceText <- VectorSource(byLineText)
corpusText <- VCorpus(sourceText)
corpusText <- tm_map(corpusText, removeNumbers)
corpusText <- tm_map(corpusText, content_transformer(tolower))
corpusText <- tm_map(corpusText, PlainTextDocument)
corpusText <- tm_map(corpusText, removePunctuation)
corpusText <- tm_map(corpusText, stripWhitespace)
corpusText <- tm_map(corpusText, removeWords, stopwords('english'))
getRidOff <- content_transformer(function(x, pattern) gsub(pattern, "", x))
corpusText <- tm_map(corpusText, getRidOff, "@[[:alnum:]]*")
corpusText <- tm_map(corpusText, getRidOff, "http[[:alnum:]]*")


```

The corpus is tokenized against 1-gram, 2-grams and 3-grams to understand the frequencies. The tokenized corpora are converted to term document matrices.

```{r corpustok, message=FALSE, warning=FALSE}

oneGramTokenizer <- function(x) NGramTokenizer(x,
                                               Weka_control(min = 1, max = 1))

twoGramTokenizer <- function(x) NGramTokenizer(x,
                                               Weka_control(min = 2, max = 2))

threeGramTokenizer <- function(x) NGramTokenizer(x,
                                                 Weka_control(min = 3, max = 3))

onetdm <- TermDocumentMatrix(corpusText, control = list(tokenize = oneGramTokenizer))
twotdm <- TermDocumentMatrix(corpusText, control = list(tokenize = twoGramTokenizer))
threetdm <- TermDocumentMatrix(corpusText, control = list(tokenize = threeGramTokenizer))

```

## Results

The documents are then converted to data frames and top 20 frequencies are extracted. These are plotted to understand the distributions of top words and their associations.  

```{r results, message=FALSE, warning=FALSE}

oneDf <- as.data.frame(as.table(onetdm))
twoDf <- as.data.frame(as.table(twotdm))
threeDf <- as.data.frame(as.table(threetdm))

oneDfSort <- oneDf[order(oneDf$Freq, decreasing = TRUE), ]
twoDfSort <- twoDf[order(twoDf$Freq, decreasing = TRUE), ]
threeDfSort <- threeDf[order(threeDf$Freq, decreasing = TRUE), ]

oneForGraph <- oneDfSort[1:20, ]
twoForGraph <- twoDfSort[1:20, ]
threeForGraph <- threeDfSort[1:20, ]

ggplot(oneForGraph, aes(x=Terms,y=Freq)) + 
    geom_bar(stat="Identity", fill="Red")

ggplot(twoForGraph, aes(x=Terms,y=Freq)) + 
    geom_bar(stat="Identity", fill="Blue")

ggplot(threeForGraph, aes(x=Terms,y=Freq)) + 
    geom_bar(stat="Identity", fill="Green")


```
