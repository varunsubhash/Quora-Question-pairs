# Quora-Question-pairs

<h1> 1. Business Problem </h1>

<h2> 1.1 Description </h2>

<p>Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.</p>
<p>
Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
</p>
<br>
> Credits: Kaggle 

  # Problem Statement<br>
- Identify which questions asked on Quora are duplicates of questions that have already been asked. <br>
- This could be useful to instantly provide answers to questions that have already been answered. <br>
- We are tasked with predicting whether a pair of questions are duplicates or not. <br>

<h2> 1.2 Source</h2>

- Source : https://www.kaggle.com/c/quora-question-pairs

<h2>1.3 Real world/Business Objectives and Constraints </h2>

1. The cost of a mis-classification can be very high.
2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
3. No strict latency concerns.
4. Interpretability is partially important.

<h1>2. Machine Learning Probelm </h1>

<h2> 2.1 Data </h2>

<h3> 2.1.1 Data Overview </h3>

<p> 
- Data will be in a file Train.csv <br>
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate <br>
- Size of Train.csv - 60MB <br>
- Number of rows in Train.csv = 404,290
</p>

<h3> 2.1.2 Example Data point </h3>

<pre>
"id","qid1","qid2","question1","question2","is_duplicate"
"0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
"11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
</pre>

<h2> 2.2 Mapping the real world problem to an ML problem </h2>

<h3> 2.2.1 Type of Machine Leaning Problem </h3>

<p> It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not. </p>

<h3> 2.2.2 Performance Metric </h3>

Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s): 
* log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
* Binary Confusion Matrix

<h2> 2.3 Train and Test Construction </h2>

<p>  </p>
<p> We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with. </p>

<h1>3. Exploratory Data Analysis </h1>

<h2> 3.1 Reading data and basic stats </h2>

- Number of data points: 404290<br>

- We get the below information regarding the data
 <class 'pandas.core.frame.DataFrame'>
RangeIndex: 404290 entries, 0 to 404289 <br>
Data columns (total 6 columns):<br>
 #   Column         Non-Null Count    Dtype <br>
---  ------         --------------    ----- 
- 0   id             404290 non-null  int64 <br>
- 1   qid1           404290 non-null  int64 <br>
- 2   qid2           404290 non-null  int64 <br>
- 3   question1      404289 non-null  object<br>
- 4   question2      404288 non-null  object<br>
- 5   is_duplicate   404290 non-null  int64 <br>
-dtypes: int64(4), object(2)<br>

- We are given a minimal number of data fields here, consisting of:<br>

  - id:  Looks like a simple rowID<br>
  - qid{1, 2}:  The unique ID of each question in the pair<br>
  - question{1, 2}:  The actual textual contents of the questions.<br>
  - is_duplicate:  The label that we are trying to predict - whether the two questions are duplicates of each other.<br>
  
 <h3> 3.2.1 Distribution of data points among output classes</h3>
- Number of duplicate(smilar) and non-duplicate(non similar) questions<br>
- Plotting the bar graph for duplicate and non duplicate question we get the below graph.

![](Capture1.PNG)

- Total number of question pairs for training:  404290<br>
- Question pairs are not Similar (is_duplicate = 0):63.08%<br>
- Question pairs are Similar (is_duplicate = 1): 36.92%<br>

<h3> 3.2.2 Number of unique questions </h3>

- We find the below information regarding questions.<br>

  - Total number of  Unique Questions are: 537933<br>

  - Number of unique questions that appear more than one time: 111780 (20.77953945937505%)<br>

  - Max number of times a single question is repeated: 157<br>

![](Capture2.PNG)

- We check if any duplicates are present.<br>
- A log histograms of question appearance count  are as shown below.<br>


![](Capture3.PNG)

- We will replace Nan values using ' '.<br>

- <h2>3.3 Basic Feature Extraction (before cleaning) </h2>


Let us now construct a few features like:<br>
 - ____freq_qid1____ = Frequency of qid1's
 - ____freq_qid2____ = Frequency of qid2's 
 - ____q1len____ = Length of q1
 - ____q2len____ = Length of q2
 - ____q1_n_words____ = Number of words in Question 1
 - ____q2_n_words____ = Number of words in Question 2
 - ____word_Common____ = (Number of common unique words in Question 1 and Question 2)
 - ____word_Total____ =(Total num of words in Question 1 + Total num of words in Question 2)
 - ____word_share____ = (word_common)/(word_Total)
 - ____freq_q1+freq_q2____ = sum total of frequency of qid1 and qid2 
 - ____freq_q1-freq_q2____ = absolute difference of frequency of qid1 and qid2 
 
 <h3> 3.3.1 Analysis of some of the extracted features </h3>
 
 - Here are some questions have only one single words.
 
 - Minimum length of the questions in question1 :  1<br>
 - Minimum length of the questions in question2 :  1<br>
 - Number of Questions with minimum length [question1] : 67<br>
 - Number of Questions with minimum length [question2] : 24<br>
 
 - Plotting violin plot of is duplicate vs wordshare and distribution plots of the same we get the below results.
 
 ![](Capture4.PNG)
 
 - The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity<br>
- The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)<br>

- Now doing the same with word common.We get the below plots.

![](Capture5.PNG)

- The distributions of the word_Common feature in similar and non-similar questions are highly overlapping.

<h2> 3.4 Preprocessing of Text </h2>

- Preprocessing:<br>

- Here we pre process the below data.
    - Removing html tags <br>
    - Removing Punctuations<br>
    - Performing stemming<br>
    - Removing Stopwords<br>
    - Expanding contractions etc.<br>
    
 <h2> 3.5 Advanced Feature Extraction (NLP and Fuzzy Features) </h2>
    
- Function to Compute and get the features : With 2 parameters of Question 1 and Question 2<br>

Definition:
- __Token__: You get a token by splitting sentence a space
- __Stop_Word__ : stop words as per NLTK.
- __Word__ : A token that is not a stop_word


Features:<br>
-  __cwc_min__ :  Ratio of common_word_count to min lenghth of word count of Q1 and Q2 <br>cwc_min = common_word_count / (min(len(q1_words), len(q2_words))<br>
-   __cwc_max_ :  Ratio of common_word_count to max lenghth of word count of Q1 and Q2 <br>cwc_max = common_word_count / (max(len(q1_words), len(q2_words))<br>

-  __csc_min_ :  Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 <br> csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))<br>

-   __csc_max_ :  Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2<br>csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))<br>

-   __ctc_min_ :  Ratio of common_token_count to min lenghth of token count of Q1 and Q2<br>ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))<br>

-   __ctc_max_ :  Ratio of common_token_count to max lenghth of token count of Q1 and Q2<br>ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))<br>
        
-  __last_word_eq_ :  Check if First word of both questions is equal or not<br>last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])<br>

-  __first_word_eq_ :  Check if First word of both questions is equal or not<br>first_word_eq = int(q1_tokens[0] == q2_tokens[0])<br>
        
- __abs_len_diff_ :  Abs. length difference<br>abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))<br>

- __mean_len_ :  Average Token Length of both Questions<br>mean_len = (len(q1_tokens) + len(q2_tokens))/2<br>

- __longest_substr_ratio_ :  Ratio of length longest common substring to min lenghth of token count of Q1 and Q2<br>longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))

<h3>3.5.1 Analysis of extracted features </h3>

<h4> 3.5.1.1 Plotting Word clouds</h4>

- Creating Word Cloud of Duplicates and Non-Duplicates Question pairs<br>
- We can observe the most frequent occuring words<br>

- We read the text files and removing the stop words<br>

- Word Clouds are generated from  duplicate pair question's text and we get the below.<br>

![](Capture6.PNG)

- Similarly word Clouds are generated from non duplicate pair question's text and we get the below.<br>

![](Capture7.PNG)

<h4> 3.5.1.2 Pair plot of features ['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'] </h4>

- We get the below plots

![](Capture8.PNG)

- Plotting the viloin plot for token_sort_ratio vs is_duplicate and plotting the distribution plot for token_sort_ratio we get the below plots.

 ![](Capture9.PNG)

- Now repeating for fuzz_ratio we get the below

 ![](Capture10.PNG)

- After using tsne and reducing to 3 dimensions plotting sns plot we get the below.

 ![](Capture11.PNG)

- Plotting the 3d plot we get the below.

 ![](Capture12.PNG)
 
 <h2> 3.6 Featurizing text data with tfidf weighted word-vectors </h2>
 
 - After we find TF-IDF scores, we convert each question to a weighted average of word2vec vectors by these scores.<br>
- here we use a pre-trained GLOVE model which comes free with "Spacy".<br>
- It is trained on Wikipedia and therefore, it is stronger in terms of word semantics. <br>

-  Questions 1 and 2 tfidf weighted word2vec are computed.

<h1>4. Machine Learning Models </h1>

<h2> 4.1 Reading data from file and storing into sql table </h2>

- Data is read and stored into sql table.

<h2> 4.2 Converting strings to numerics </h2>

-  After we read from sql table each entry was read it as a string ,we convert all the features into numaric before we apply any model.

 <h2> 4.3 Random train test split( 70:30) </h2>
 
 - The data is split in 70-30 ratio.
 
 <h2> 4.4 Building a random model (Finding worst-case log-loss) </h2>
 
 - This is the worst possible model.<br>
 - We get the Log loss on Test Data using Random Model 0.887242646958 and the below confusion matrix, precision and recall matrix.
 
 ![](Capture13.PNG)
 
 <h2> 4.4 Logistic Regression with hyperparameter tuning </h2>
 
 - For values of best alpha =  1 The train log loss is: 0.513842874233<br>
 
 - For values of best alpha =  1 The test log loss is: 0.520035530431<br>
 
 - Plot of alpha vs error measure and the confusion, precision and recall matrix are as shown below respectively.
 
  ![](Capture14.PNG)
  
  ![](Capture15.PNG)
  
 <h2> 4.5 Linear SVM with hyperparameter tuning </h2>
 
- For values of alpha =  1e-05 The log loss is: 0.657611721261<br>
- For values of alpha =  0.0001 The log loss is: 0.489669093534<br>
- For values of alpha =  0.001 The log loss is: 0.521829068562<br>
- For values of alpha =  0.01 The log loss is: 0.566295616914<br>
- For values of alpha =  0.1 The log loss is: 0.599957866217<br>
- For values of alpha =  1 The log loss is: 0.635059427016<br>
- For values of alpha =  10 The log loss is: 0.654159467907<br>

- For values of best alpha =  0.0001 The train log loss is: 0.478054677285<br>
- For values of best alpha =  0.0001 The test log loss is: 0.489669093534<br>

- Plot of alpha vs error measure and the confusion, precision and recall matrix are as shown below respectively.


  ![](Capture16.PNG)
  
  
  ![](Capture17.PNG)
  
  <h2> 4.6 XGBoost </h2>
  
  - The test log loss is: 0.357054433715.
  - Plot of confusion, precision and recall matrix are as shown below respectively.
  - We can see it is the best model.
  
  ![](Capture18.PNG)
  



