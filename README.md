# intelunnati_Trailblazers

# Introduction
The lives of individuals have been impacted by significant technological flaws in recent years as a result of information sharing.
The era of social media has accelerated the spread of a lot of fake news from anti-social groups throughout the globe.
Applying the machine learning techniques we learned in class will help our group tackle this challenge by producing a result 
that makes it posssible for us to do so.
The spread of false information has caused instablity in both past and the present,
whihch has also resulted in the loss of human lives. In order to detect wether news is legitimate and bogus using machine learning concepts,
our organiztion aims to help ensure peace and sanity.

# Team Members
All of us are E.C.E undergrads at `NMIT`
- Akash Rai
- Harsh Ranjan
- Ankur

# Fake news detection
In this project, we have used various natural language processing techniques and machine learning algorithms
to clarify fake news article using sci-kit libraries from python. We have also used a pre trained model that is BERT,
Instead of only human-generated fake news classification we have also added another parameter in our Labels
feature and that is AI-generated fake news as nowadays many sources do use AI-generated text for creating Fake News.

# Dataset
![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/c030c6fe-9645-4c7d-84af-fed91aa4692b)

We had been given the ISOT Dataset as our base dataset and no restrictions were placed on adding more on top of it, so we tried to add more datasets due to the reason that our news dataset covers only a limited number of domains such as Politics and World-News focused on US & Middle-East governments. This couldn’t be effective for news from different domains, so we combined it with other resources, but it took different pre-processing techniques for each of them respectively before concatenating them and would be attempted later on while pursuing to perfectly complete our project as a whole and being deployed.

## Exploratory Data Analysis
The most important step in the ML development cycle is the data preprocessing or the EDA part as it prepares our dataset in such a way that the machine could take in the most important features for classification. We had begun building by using the most Basic ML algorithms such as Logistic Regression, Support Vector Classifiers, XGBoost, Passive Aggressive Classifiers, and Random Forests. We applied preprocessing techniques like Regex and lowering cases to remove unnecessary symbols and link present our data.

   * In the beginning, when we ran our models, we noticed that we had very high accuracies nearing 99%. This made us realize that a lot of our data was not pre-processed 
    fully since our model was obtaining near 100% accuracy while only being trained on the data. Then we went back to the EDA, cleaning our data and noticed some features 
    were very distinguishable between the different classes (avg. words per sentence, article lengths, news content). We normalized these features and were able to 
    successfully remove those factors.

Even after getting reasonably good accuracies, the model was not able to predict the real-world recent scenarios as fake or true, so we figured out that either our model was not strong enough to learn semantics or we were required to train or more data, thus we decide to move onto Deep Learning models as the Neural Networks lay a very strong learning foundation which learns a lot of features and is proven more powerful than basic ML models. The link for the complete EDA is below.

## EDA Deep Learning
There were multiple techniques we applied here for creating word Embeddings like &#8594;
* GloVe (Global Vectors for Word Representation):  Based on matrix factorization techniques on the word-context matrix.
* Word2Vec: Word2vec is not a single algorithm but a combination of two techniques – CBOW(Continuous bag of words) and Skip-gram model.
* One Hot Representation & Tokenizers: Used in NLP to encode categorical factors as binary vectors, such as words or part-of-speech identifiers.

***The best results were obtained by using GloVe embeddings which were about `98.11%` on the test set and validation accuracy of about `98.06%` using nearly the same model for 
  all  the different techniques.***

# EDA on Pre-Trained Model BERT (SOTA)
Even though the accuracy did seem very good but there was not enough semantics according to us for the model to learn the way of human interpretations, so we decided to move for one of the state of the art models in Natural Language which is BERT that is trained on Wikipedia’s dataset but didn’t use the large model having 340M  parameters due to the increase in computation times.
The transformer model BERT uses sub-word tokenization, here the BERT tokenizer splits the string into multiple substrings and the tokenization we’d done was using BertTokenizerFast for tokenizing the texts of the words on the ‘Title’ of the dataset rather than the ‘Text’ of the dataset. BertTokenizerFast is a tokenizer class that inherits from PreTrainedTokenizerFast which contains most of the main methods.

# Metric & Model Selection
After rigorous use of Machine Learning and Deep Learning models, the model we selected was a transformer model ‘bert-base-uncased’ having 12 encoders with 12 bidirectional self-attention heads totaling `110 million parameters`. By sacrificing a bit of accuracy, we saved a lot of computation time here as the ‘bert-large-uncased’ is a heavy model pre-trained on the Toronto BookCorpus `(800M words)` and English Wikipedia(2,500M words) with 24 encoders with 16 bidirectional self-attention heads totaling `340 million parameters`. BERT was trained previously trained on Wikipedia’s dataset and thus has a very comprehensive understanding of Human Language Semantics and thus is used extensively in the NLP domain, thus we gave it priority over other trained models. Another option could be the `RoBERTa model` which has the same base architecture as `BERT` but has different tokenization and other techniques explained forward.

# Model Evaluation
For our ML-based work, we evaluate our models on the accuracy score and also other measures included in the Confusion Matrix. The average accuracy we’d gained on binary classification was found to be `92.17%` and `AUC score of 0.8`
The model evaluation is done using our metrics such as accuracy and loss. We received the best accuracy by using Glove representations as `97.86%` on **training data** and `97.81%` on **testing data**. Also the ROC-AUC scores were 
By using **SOTA models** such as `BERT` got the best accuracy to be `89%` on BERT but it worked too well while using current news sources as compared to our DL models. Using Tf-idf embeddings with the BERT Architecture got us better accuracy of `93.59%` only on 1 epoch, thus have a very high chance of going even higher if trained on 5-10 epochs.
We also tried training on other transformer-based models such as DistilBERT & RoBERTa but were not able to wrap it up completely.



