# Iris Classification Using Machine Learning

This project is based on Iris Flower Classification based on three Machine Learning methods, i.e., **Decision Tree** (a supervised learning regression and classification model), **K-Nearest Neighbor** (a supervised learning classification model), and **K-Means Clustering** (an unsupervised learning clustering model for continuous data).

## Problem Statement and Dataset

This is a classification problem wherein we classify irises into one of three classes - Iris Setosa, Iris Versicolour, and Iris Virginica - based on four attributes. These factors are sepal length, sepal width, petal length, and petal width. The data set used includes 50 instances of each of the three iris classes for a total of 150 instances. While one of the classes is linearly separable, the other two are not. The dataset used is: https://archive.ics.uci.edu/dataset/53/iris.

![Iris Classification](https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png)

## Installation Guide

Welcome to the Iris Classification project! This guide will walk you through the steps to set up and run the project on your local machine. Follow the instructions below to get started with classifying iris flowers using machine learning techniques.

**Prerequisites**

Before you begin, ensure you have the following installed on your machine:

1. Python (version 3.6 or later)
2. Jupyter Notebook
3. Git (optional, for cloning the repository)
   
**Step-by-Step Installation Guide**

Step 1: Download the Dataset - You have two options to get the necessary dataset for the project:
Option 1: Download from the Provided Link: https://archive.ics.uci.edu/dataset/53/iris; 
Option 2: Download the following files from this Repository: index, iris.data, iris.names, bezdekIris.data

Step 2: Set Up the Project - Create a new directory on your local machine for the project.
Place the downloaded files (index, iris.data, iris.names, bezdekIris.data) into this directory.

Step 3: Clone the Repository (Optional) - If you prefer, you can clone the entire repository to your local machine using `git clone https://github.com/yourusername/iris-classification.git` in your terminal or command prompt and navigate to the cloned directory using `cd iris-classification`. This step is optional and can be skipped if you have already downloaded the necessary files.

Step 4: Install Required Libraries and Run the Project - In the Jupyter Notebook interface, install the necessary libraries, and navigate to the directory where you placed the files and open the IrisClassification.ipynb file. Run the notebook cells sequentially to execute the code and perform iris classification.


# A Note on Machine Learning Techniques 
![ML Techniques](https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2018/03/Types-of-Machine-Learning-Waht-is-Machine-Learning-Edureka-2.png)
The above image explains the three types of Machine Learning Models:-

1. **Supervised Learning:** This includes regression and classification. The following models are used for each-

* Regression: Linear regression, polynomial regression, decision tree, random forest

* Classification: Decision tree, random forest, KNN, trees, logistic regression, Naive-Bayes, SVM.

  

2. **Unsupervised Learning:** This includes clustering, association analysis, and Hidden Markov Model-

* Clustering: SVD, PCA, K-Means

* Association Analysis: Apriori, FP-Growth

* Hidden Markov Model (a statistical model that can be used to describe the evolution of observable events that depend on internal factors, which are not directly observable)

  

3. **Reinforcement Learning**

![ML Application Examples](https://miro.medium.com/v2/resize:fit:2796/format:webp/1*FUZS9K4JPqzfXDcC83BQTw.png)

## Procedure Followed
After importing the dataset, we visualise it using *matplotlib* in various ways to properly understand what is going on. 

The next step is to preprocess the dataset to get it ready for our project. A description of the preprocessing procedure for all three algorithms used is as follows:

1. Decision Tree - This algorithm does not require much preprocessing -- only the label encoder. We start with separating the independent variables (all four into one variable 'x') from the target variable 'y'.

> (i) Label Encoding: Converting labels into numerical form so that it is understood by the machine learning model. In this project, the labels , 'Iris-setosa', 'Iris-versicolor', and 'Iris-virginica', have been converted to 0, 1, and 2 respectively. Similarly, the labels for _x_ have been discarded and the table has been replaced by a numpy array.
> (ii) One-Hot Encoding: With one-hot, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns. Each integer value is represented as a binary vector. For example, the labels , 'Iris-setosa', 'Iris-versicolor', and 'Iris-virginica', can be converted to [1, 0, 0], [0, 1, 0], and [0, 0, 1] respectively. We are not using one-hot encoding in this project. Label Encoding is suitable when there is an intrinsic order in the categories, whereas One-Hot Encoding is better for nominal categories, i.e., if there is no inherent order or ranking to the categorical data, OneHotEncoder is more appropriate, while LabelEncoder is appropriate when the categorical data has an inherent order or ranking.
> (iii) Standard Scalar: 

After preprocessing, we split the dataset into training and testing datasets. Then we get an idea of the predictions by visualising using a decision tree. We will follow the same procedure for the other two algorithms. 

Once the dataset is split, we train the model, test it and visualise the decision tree. We can also predict outputs for random inputs.

## Libraries Used
1. Pandas - Used for analyzing, cleaning, exploring, and manipulating data sets.
2. Numpy - Used for working with arrays and other computational tasks. It also has functions for working in the domains of linear algebra, fourier transform, and matrices.
3. Plotly - Used for data visualization and supports various graphs like line charts, scatter plots, bar charts, histograms, area plots, etc. Plotly produces interactive graphs, can be embedded on websites, and provides a wide variety of complex plotting options. Plotly Express is a terse, consistent, high-level API for creating figures. It is basically a wrapper of Plotly. Plotly’s `plotly.offline` allows you to generate graphs offline and save them in local machine.
4. Cufflinks - Used to bind Pandas and Plotly together so that we can use Plotly in Pandas.
5. Matplotlib - Matplotlib is a popular plotting library in Python used for creating high-quality visualizations and graphs. It offers various tools to generate diverse plots, facilitating data analysis, exploration, and presentation. `matplot.pyplot` is a collection of methods within matplotlib which allows user to construct 2D plots easily and interactively.
6. Scikit-learn (sklearn) - Contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
7. GraphViz: This is an open source graph visualization software. Graph visualization is a way of representing structural information as diagrams of abstract graphs and networks.
8. OS: The OS comes under Python's standard utility modules. This module offers a portable way of using operating system dependent functionality. The Python OS module lets us work with the files and directories.

> NOTE: `%matplotlib` is a magic function in IPython. IPython has a set of predefined ‘magic functions’ that you can call with a command line style syntax. There are two kinds of magics, line-oriented and cell-oriented. Line magics are prefixed with the % character and work much like OS command-line calls: they get as an argument the rest of the line, where arguments are passed without parentheses or quotes. Lines magics can return results and can be used in the right hand side of an assignment. Cell magics are prefixed with a double %%, and they are functions that get as an argument not only the rest of the line, but also the lines below it in a separate argument.
`%matplotlib inline` sets the backend of matplotlib to the 'inline' backend. With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.
