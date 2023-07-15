
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

# **Work Flow**
![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/c0900d97-6b4d-471b-a46b-053ccd081159)

1. First of all we bild a simple ML model using `Random Forest` and we got an unreasonably High Accuracies.
2. Then we tried to add `Fake AI parameter` which comes under unsupervised learning and here we got a accuracy but not a good performance on real world news.
3. After this we tried `Deep Learning(LSTM)` under which a variety of EDA is done but still some issue was there while checking on real world news sets.
4. So we tried to switch to a `Pre-Trained BERT model` which showed a very good semantic understanding with good performance on real world news sets.
5. Now we tried `BERT optimized(Attention)` and we were trying to implement newer layers for better performance.
# Dataset
![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/c030c6fe-9645-4c7d-84af-fed91aa4692b)


We were provided with the ISOT Dataset as our base dataset and no restrictions were placed on adding more on top of it, so we tried to add more datasets due to the reason that our news dataset covers only a limited number of domains such as Politics and World-News focused on US & Middle-East governments. This couldn’t be effective while checking out for news from different domains, so we combined it with other resources, but it took different pre-processing techniques for each of them respectively before concatenating them and would be attempted later on while pursuing to perfectly complete our project as a whole and being deployed.

# In-Depth Exploratory Data Analysis

* About Dataset
  The distribution of news 
  across different domains is as follows:
  ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/cd8b6989-a228-4020-b313-0b4dbdf3bfef)
  <br></br>
  ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/fb1d2a2c-8349-482c-a0a3-601e820e1a62)

 * As we can see here that only Fake news has been spread throughout 6 different domains while the True file has news sub-divided only into 2 domains.
  The WordCloud is shown below and it describes the most repeated words in both files:

  * **True Data**
    
    ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/7e7bd945-e09b-41ff-9cb6-4d57a1aff9ff)


  * **False Data**
    
     ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/eadb78b1-e158-4177-8280-e1e337331eff)


    * Split Of Data between Real and Fake News Files
      
      ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/b2ab2b50-260c-49a8-8909-2adbe10643a1)


 **Our Machine Learning model is consisting of the classification of fake news between human-authored fake news, real news and AI-generated fake news. Further we’ve done two 
 way classification between them and the results are displayed as follows**

 ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/85e12a57-3d92-4614-a287-465d75b9e9be)

 **These represent the ROC-AUC curve and the Precision-Recall curves respectively and as we can see even though our Accuracy was quite good the results didn’t converge up to 
   the mark.**










## Exploratory Data Analysis
The most important step in the ML development cycle is the data preprocessing or the EDA part as it prepares our dataset in such a way that the machine could take in the most important features for classification. We had begun building by using the most Basic ML algorithms such as Logistic Regression, Support Vector Classifiers, XGBoost, Passive Aggressive Classifiers, and Random Forests. We applied preprocessing techniques like Regex and lowering cases to remove unnecessary symbols and link present our data.

   * In the beginning, when we ran our models, we noticed that we had very high accuracies nearing 99%. This made us realize that a lot of our data was not pre-processed 
    fully since our model was obtaining near 100% accuracy while only being trained on the data. Then we went back to the EDA, cleaning our data and noticed some features 
    were very distinguishable between the different classes (avg. words per sentence, article lengths, news content). We normalized these features and were able to 
    successfully remove those factors.

Even after getting reasonably good accuracies, the model was not able to predict the real-world recent scenarios as fake or true, so we figured out that either our model was not strong enough to learn semantics or we were required to train or more data, thus we decide to move onto Deep Learning models as the Neural Networks lay a very strong learning foundation which learns a lot of features and is proven more powerful than basic ML models. The link for the complete EDA is  <a href="https://github.com/akashrai2003/intelunnati_Trailblazers/blob/main/Docs/Extensive%20EDA%20with%20Visualizations.docx"> here</a>

## EDA Deep Learning
There were multiple techniques we applied here for creating word Embeddings like &#8594;
* GloVe (Global Vectors for Word Representation):  Based on matrix factorization techniques on the word-context matrix.
* Word2Vec: Word2vec is not a single algorithm but a combination of two techniques – CBOW(Continuous bag of words) and Skip-gram model.
* One Hot Representation & Tokenizers: Used in NLP to encode categorical factors as binary vectors, such as words or part-of-speech identifiers.

***The best results were obtained by using GloVe embeddings which were about `98.11%` on the test set and validation accuracy of about `98.06%` using nearly the same model for 
  al  the different techniques.***

# EDA on Pre-Trained Model BERT (SOTA)
Even though the accuracy did seem very good but there was not enough semantics according to us for the model to learn the way of human interpretations, so we decided to move for one of the state of the art models in Natural Language which is BERT that is trained on Wikipedia’s dataset but didn’t use the large model having 340M  parameters due to the increase in computation times.
The transformer model BERT uses sub-word tokenization, here the BERT tokenizer splits the string into multiple substrings and the tokenization we’d done was using BertTokenizerFast for tokenizing the texts of the words on the ‘Title’ of the dataset rather than the ‘Text’ of the dataset. BertTokenizerFast is a tokenizer class that inherits from PreTrainedTokenizerFast which contains most of the main methods.

# Metric & Model Selection
After rigorous use of Machine Learning and Deep Learning models, the model we selected was a transformer model ‘bert-base-uncased’ having 12 encoders with 12 bidirectional self-attention heads totaling `110 million parameters`. By sacrificing a bit of accuracy, we saved a lot of computation time here as the ‘bert-large-uncased’ is a heavy model pre-trained on the Toronto BookCorpus `(800M words)` and English Wikipedia(2,500M words) with 24 encoders with 16 bidirectional self-attention heads totaling `340 million parameters`. BERT was trained previously trained on Wikipedia’s dataset and thus has a very comprehensive understanding of Human Language Semantics and thus is used extensively in the NLP domain, thus we gave it priority over other trained models. Another option could be the `RoBERTa model` which has the same base architecture as `BERT` but has different tokenization and other techniques explained forward.
  * ***Metrics*** used here was the ***Confusion Matrix*** for BERT’s initial training as it is considered one of the best to be used in binary classification tasks. It takes into 
    consideration all four values in the confusion matrix and can be defined by the following &#8594;

     ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/a9f1a8cc-b44c-4143-9a02-f926f6466ca6)

      Other metrics that were chosen are the ROC-AUC curve and Confusion Matrix elements for evaluating our ML models and for Deep Learning models.
      All the encoding methods used in our project were introduced briefly inside the EDA section and a more in-depth version of the file could be seen <a href="https://github.com/akashrai2003/intelunnati_Trailblazers/blob/main/Docs/Extensive%20EDA%20with%20Visualizations.docx"> here</a>



# Model Evaluation
For our ML-based work, we evaluate our models on the accuracy score and also other measures included in the Confusion Matrix. The average accuracy we’d gained on binary classification was found to be `92.17%` and `AUC score of 0.8`
The model evaluation is done using our metrics such as accuracy and loss. We received the best accuracy by using Glove representations as `97.86%` on **training data** and `97.81%` on **testing data**. Also the ROC-AUC scores were 
By using **SOTA models** such as `BERT` got the best accuracy to be `89%` on BERT but it worked too well while using current news sources as compared to our DL models. Using Tf-idf embeddings with the BERT Architecture got us better accuracy of `93.59%` only on 1 epoch, thus have a very high chance of going even higher if trained on 5-10 epochs.
We also tried training on other transformer-based models such as `DistilBERT` & `RoBERTa` but were not able to wrap it up completely.

# Deep Learning Models
The best performance was obtained by training our LSTM model with the layers specified below and using Glove Embeddings and the evaluation metric have been plotted:





![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/314fd1ec-cf2a-4c28-afdb-bd775fb5c944)


* Accuracy v/s Epochs


![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/9d4d28fc-bc19-43dc-9c61-f66aa5a902fc)


* Loss v/s Epochs

![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/3e2780e9-0db0-4b2a-b2ae-6d718fb2ce96)


* ROC Curve


 ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/5ae44689-6380-414d-b0ce-2b584d732adb)


 
 * CONFUSION MATRIX
 
 
 ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/439f6273-1a1f-4b5d-83b6-438aa9d5cb0a)


   * Even though we had a very good accuracy on our deep learning models the only problem was that of generalization and to solve it we needed too much data in order to make 
     our machine learn human semantics of Natural Language and thus we came to the point of choosing transformer-based models and hence experimented on BERT, RoBERTa and 
     DistilBERT. And were successful only in verifying results from our model built upon the BERT Architecture due to the assigned timeframe. Stil,l other results have also 
     been displayed below:

# BERT
Our training accuracy was about 90% and the confusion matrix is represented here. Even though our accuracy was not as good as our deep learning model, it outperformed all our previous models in classifying real-world news titles as fake or true.
 
 * We just used the complete pre-trained model in this case and didn’t use any of the word embeddings except tokenizing the sentences using BertTokenizer.
   
   * CONFUSION MATRIX

     ![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/51daf2b0-c7e7-4e38-b1fa-27a5e1c1685e)


# BERT with TF-IDF Vectorizer:

 Now moving forward we apply one of the mostly used embedding techniques i.e Tf-idf Vectorizer and thus our results are highly improved still without having any additional layers on top of the BERT Architecture. This model couldn't be tested thoroughly on custom news data due to the matter of submissions closing in but it can surely outperform our previous BERT model due to the addition of word embeddings.

  * Even while training on a single epoch(due to time constraints) we were able to get a very high accuracy of 93.59% and could have easily increased if allowed to be 
    trained on 5-10 epochs.



![image](https://github.com/akashrai2003/intelunnati_Trailblazers/assets/134039081/f8d78af5-74ad-4a84-8836-975049c28563)  


 

 


# Future Aspects
This repository will keep on being updated from our side with the help of possible more solutions which couldn't be created due to the timeframe which will include experimenting with other models such as DistilBERT & RoBERTa etc.
Instead of going for a different dataset as this dataset doesn't have much info about the authenticity of the sources and we can actually create better models using the attention mechanism and give more weightage to other features rather than only focusing on the label and the context. Also, a method which would be able to support our decisions by displaying links related to the context done with the help of APIs & web scraping.
* Attention modelling requires features to be present which can be given more importance too for example, we can affirm by training which author has more chances of producing a real article
* Clickbait can also be a classification criterion with the inclusion of True and Fake
* Using a more complex network of layers we would be able to increase our generalization of our model and can highly expect a very good accuracy 




