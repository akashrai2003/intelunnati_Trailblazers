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
* The most important step in the ML development cycle is the data preprocessing or the EDA part as it prepares our dataset in such a way that the machine could take in the most important features for classification. We had begun building by using the most Basic ML algorithms such as Logistic Regression, Support Vector Classifiers, XGBoost, Passive Aggressive Classifiers, and Random Forests. We applied preprocessing techniques like Regex and lowering cases to remove unnecessary symbols and link present our data.
   * 	In the beginning, when we ran our models, we noticed that we had very high accuracies nearing 99%. This made us realize that a lot of our data was not pre-processed 
        fully since our model was obtaining near 100% accuracy while only being trained on the data. Then we went back to the EDA, cleaning our data and noticed some features 
        were very distinguishable between the different classes (avg. words per sentence, article lengths, news content). We normalized these features and were able to 
        successfully remove those factors.

