# Iris Classification Using Machine Learning

This project is based on Iris Flower Classification based on three Machine Learning methods, i.e., **Decision Tree** (a supervised learning regression and classification model), **K-Nearest Neighbor** (a supervised learning classification model), and **K-Means Clustering** (an unsupervised learning clustering model for continuous data).

## Problem Statement and Dataset

This is a classification problem wherein we classify irises into one of three classes - Iris Setosa, Iris Versicolour, and Iris Virginica - based on four attributes. These factors are sepal length, sepal width, petal length, and petal width. The data set used includes 50 instances of each of the three iris classes for a total of 150 instances. While one of the classes is linearly separable, the other two are not. The dataset used is: https://archive.ics.uci.edu/dataset/53/iris.

![Iris Classification](https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png)

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

The next step is to preprocess the dataset to get it ready for our project. This includes <enter steps>

After preprocessing, we split the dataset into training and testing datasets. Then we get an idea of the predictions by visualising using a decision tree. We will follow the same procedure for the other two algorithms. 

## Libraries Used
1. Pandas - Used for analyzing, cleaning, exploring, and manipulating data sets.
2. Numpy - Used for working with arrays and other computational tasks. It also has functions for working in the domains of linear algebra, fourier transform, and matrices.
3. Plotly - Used for data visualization and supports various graphs like line charts, scatter plots, bar charts, histograms, area plots, etc. Plotly produces interactive graphs, can be embedded on websites, and provides a wide variety of complex plotting options. Plotly Express is a terse, consistent, high-level API for creating figures. It is basically a wrapper of Plotly. Plotly’s `plotly.offline` allows you to generate graphs offline and save them in local machine.
4. Cufflinks - Used to bind Pandas and Plotly together so that we can use Plotly in Pandas.
5. Matplotlib - Matplotlib is a popular plotting library in Python used for creating high-quality visualizations and graphs. It offers various tools to generate diverse plots, facilitating data analysis, exploration, and presentation. `matplot.pyplot` is a collection of methods within matplotlib which allows user to construct 2D plots easily and interactively.

> NOTE: `%matplotlib` is a magic function in IPython. IPython has a set of predefined ‘magic functions’ that you can call with a command line style syntax. There are two kinds of magics, line-oriented and cell-oriented. Line magics are prefixed with the % character and work much like OS command-line calls: they get as an argument the rest of the line, where arguments are passed without parentheses or quotes. Lines magics can return results and can be used in the right hand side of an assignment. Cell magics are prefixed with a double %%, and they are functions that get as an argument not only the rest of the line, but also the lines below it in a separate argument.
`%matplotlib inline` sets the backend of matplotlib to the 'inline' backend. With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.
