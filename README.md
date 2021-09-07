# Japanese-Honorific-Classifier
As a foreign student in Japan, we are compelled to encounter various unfamiliar situation and are asked to speak appropriate 
Japanese in terms of the environment. It is a tough task for foreigners to differentiate between normal Japanese and Proper 
Japanese. There are Honorifics in Japanese, and it turns out to be confusing for foreigners whose mother language does not 
contain such forms.

This classifier aims to help foreigners in Japan understand the form of Japanese that they are using to cope with issues under different situation.
To be more specific, there is a numbers of proper forms in Japanese Keigo (敬
語), such as Kenjyougo (謙譲語)、Sonkeigo (尊敬語)、Bikago (美化語)、Teineigo (丁寧語). These are all honorifics, so 
called proper Japanese, but they are asked to be used only under certain conditions. 

This classifier is going to classify whether a sentence is informal, polite (丁寧語), or formal (尊敬語、謙讓語、美化語). As shown in the image, users can type in a sentence and press the "predict" botton to check the form of the sentence.  

![image](https://user-images.githubusercontent.com/71431125/132279071-cec6715a-7d23-4d05-bc56-239eeefeddcb.png)

## Data Collection
Since it is a classification task, I tried to find sentences related to its Keigo labels in the internet, but I couldn't find any. 
Therefore, I decided to manually collect 200 sentences from Japanese animation, news reports, novels, and textbooks, and label it with three classes: informal, polite, and formal. However, although it doesn't really affect the performance of the classifier, I do end up with some problems due to the shortage of data.

## Training
The 200 data is split with the ratio of 8:2; therefore, there are 160 sentences for training and 40 sentences for testing. 
### Nagisa Tokenizer and TFIDF Vectorizer
Japanese is not like English, that words can be tokenized easily. I applied Nagisa Tokenizer (https://github.com/taishi-i/nagisa), which I found pretty nice. It helps tokenize words and also deleting the stopwords to make things easier for me. 
### Machine Learning Models
It would be nice if I could build up a neural network to improve its performance, but with only 200 data, it wouldn't really work. Therefore, I decided to only implement several machine learning models, such as logistic regression, naive bayes, SGD classifier, and support vector machine (SVM). And the performance is as follows:
![image](https://user-images.githubusercontent.com/71431125/132280727-25204d33-4d9b-45a7-b882-02d76af5e40b.png)

