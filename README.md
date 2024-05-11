# SentAnalysis

This project includes two sub-projects. The first is a model to determine sentimental analysis on movie reivews. Then second is a model to determine sentimental analysis on amazon product reviews.
The datasets and models are too large to add to guthub so they are stored locally. However, the generation and training of all the models are included in the code. Running the programs would generate, train and save the models locally.
The datasets can be found on Kaggle but any sentimental analysis datasets can be used in this project. Movie Reivew Model: 91%. Amazon Product Review Model: 88%.

The model uses multiple classifiers to predict a label when given an input. The label with majority vote among the classifiers becomes the prediction. The classifiers used are Naive Bayes, Multinomial Naive Bayes, Bernoulli, 
Logistic Regression, SGD, SVC, Linear SVC, and NuSVC.
