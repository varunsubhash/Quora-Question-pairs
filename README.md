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
