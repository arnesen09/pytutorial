
# coding: utf-8

# ### Introduction to Python for Data Analysis
# **Author: Dane.Arnesen**  
# **Created: 2017-10-17**  
# **Intended Audience: New to Python**  
# 
# **So what is Python?**  
# Python is general purpose programming language that has been around since the early 90s. Python is an interpreted language (code is executed directly, not compiled) that emphasizes code readability. If you're familiar with languages like Java this will become quickly apparent.
# 
# In recent years, Python has been gaining popularity in the world of data science. Why? Because, Python is awesome for data manipulation, data visualization, predictive modeling and more...
# 
# **Getting started**  
# Python may be used locally on your desktop, in the cloud (AWS or similar platforms), and even on Hadoop (PySpark). For those of you who want to run Python locally, I recommend downloading/installing Anaconda:
# 
# https://www.anaconda.com/download/
# 
# Anaconda is a Python data science platform. It comes with a majority of the packages you'll ever need, as well as various tools to work with Python. 
# 
# This particular demo is delivered in a Jupyter notebook. However, Anaconda also comes with a more traditional IDE called Spyder. 
# 
# **Online Resources**  
# - Free online interactive Python tutorial: https://www.learnpython.org/  
# - Pandas API reference: https://pandas.pydata.org/pandas-docs/stable/api.html
# - Numpy API reference: https://docs.scipy.org/doc/numpy-1.13.0/reference/
# - Seaborn reference: http://seaborn.pydata.org/index.html
# - Scipy API reference: https://www.scipy.org/
# - Stack Overflow is your best friend

# #### Basic commands
# We'll start with some basics. First, let's import a package called **pandas**. We import pandas and give it an alias 'pd'. Pandas is one of the more common python modules, and it enables you to work with your data as a DataFrame.

# In[4]:

# Import packages using the import command. 
import pandas as pd
import os

# Chaning my working directory
os.chdir('C://Users/dane.arnesen/Documents/Projects/pytutorial/')


# One of the first things you'll want to do with any data science project is import some data. There are a bunch of ways to do this, but in this tutorial we will use the pandas package. 
# 
# For this tutorial, we will import the Iris dataset. Iris is a benchmark dataset in the data science world. It only has 150 rows and 6 columns, and so it is a good way to test out new code and/or algorithms. 

# In[5]:

# Use Pandas to import a csv file into a dataframe
iris = pd.read_csv('data/iris.csv')

# Check the dataframe to see how many rows and columns it contains
print(iris.shape)


# Once you've successfully imported your data, the next step is to do some descriptive analysis. First, let's just take a peak at the top 5 rows of the data.
# 
# You'll notice the dataset has 6 columns, as well as an index.

# In[3]:

# Take a peak at the first couple of rows in the dataframe
iris.head()


# If you didn't already know, Iris is a type of flower. There are three different species of the Iris flower: Setosa, Versicolor, and Virginica. The three specicies of Iris may be differentiated by using Sepal Length, Sepal Width, Petal Length, and Petal Width. 

# In[4]:

# What are the different species of Iris and how many rows are associated with each
iris.groupby('Species')['Species'].count()


# In[5]:

# Doing some basic descriptives
iris.groupby('Species')['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'].describe()


# Sometimes you may want to drop certain columns from a dataframe. In this case, I want to drop the Id column because it isn't relevant to the analysis.

# In[6]:

# Dropping the Id column from the dataframe
iris.drop(['Id'], inplace=True, axis=1)

# Checking the shape of the dataframe
print(iris.shape)


# #### Visualizing your data
# 
# Often times in an analysis it is helpful to visualize your data. Python has a number of packages to that end. In this tutorial, I will use a package called **seaborn**.

# In[7]:

# When using Jupyter notebooks, you have to specify the following command in order to make your plots show up
get_ipython().magic('matplotlib inline')

# Importing the seaborn package and creating an alias called sns
import seaborn as sns

# Importing pyplot from the matplotlib package. This will be used for things like tweaking plot size, etc.
from matplotlib import pyplot


# Seaborn has a ton of functions that are awesome for statistical visualization. I'm not going to go into a ton of detail in this tutorial, but feel free to visit the following website on your own time:
# 
# https://seaborn.pydata.org/index.html
# 
# Why is visualization important? Do you notice anything in the chart below that wasn't evident in any of the tables above?

# In[8]:

# Create a scatterplot matrix
sns.pairplot(iris, hue='Species')


# It is also possible to customize a chart's formatting. Here, we'll remove the gray background and make the plot fit into a 10x10 grid. 

# In[9]:

# Getting rid of the gray background
sns.set_style("whitegrid")

# Initializing a pyplot figure with size 10 x 10
fig, ax = pyplot.subplots(figsize=(10,10))

# Creating a box plot
sns.boxplot(x='Species', y='PetalWidthCm', data=iris, ax=ax)

# Making the chart a little cleaner to read
sns.despine(offset=10, trim=True)


# #### Subsetting
# Often times you'll want to look at subsets of your original dataset. Again, there are many different ways to accomplish this task, but I'm going to stick with pandas for this tutorial.

# In[10]:

# Create a new dataframe containing only rows where the Species is Setosa
sub1 = iris[iris['Species'] == 'Iris-setosa']

# Create a new dataframe containing rows where petal width is less than 1 and petal length is less than 2
sub2 = iris[(iris['PetalWidthCm'] < 1) & (iris['PetalLengthCm'] < 2)]

# Create a new dataframe containing rows where petal width is less than 1 or petal length is less than 2
sub3 = iris[(iris['PetalWidthCm'] < 1) | (iris['PetalLengthCm'] < 2)]

print(sub1.shape)
print(sub1.shape)
print(sub1.shape)


# **Working with matrices**   
# It is possible, and often preferable, to work with your data at the matrix level as opposed to the dataframe level. Pandas makes it easy to convert your dataframe to a matrix.

# In[11]:

# Converting your dataframe to a matrix
iris_mat = iris.as_matrix()

# Displaying rows 1-10 and all columns. Note: rows indices start at 0.
print(iris_mat[0:9,:])


# In[12]:

# Display row 21 and the first 2 columns. Note the column indices are not inclusive. 
print(iris_mat[20,0:2])


# You can also convert your matrix back to a dataframe.

# In[13]:

# Convert a matrix to a dataframe
iris_new = pd.DataFrame(iris_mat, columns=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species'])

# Display the top 20 rows
iris_new.head(20)


# **Creating new attributes**  
# It is also easy to add new, derived attributes to your dataframe.

# In[14]:

# Creating a new field called Sepal which is the product of sepal length and sepal width
iris_new['Sepal'] = iris_new['SepalLength'] * iris_new['SepalWidth']

# Getting the Sepal values at the 25th, 50th, and 75th percentiles
iris_new['Sepal'].quantile([0.25, 0.5, 0.75])


# **Scipy**  
# The Scipy package comes with a ton of statistical tools for data science. In the example below we will test whether or not there is a statistically significant difference between Petal Width of two species of Iris.

# In[15]:

from scipy import stats

# Grabbing the petal width for all setosa specicies
virginica = iris_new[iris_new['Species'] == 'Iris-virginica']['PetalWidth']

# Grabbing the petal width for all versicolor specicies
versicolor = iris_new[iris_new['Species'] == 'Iris-versicolor']['PetalWidth']


# We will perform a Mann-Whitney-Wilcoxon (MWW) RankSum test, which does not assume a guassian distribution of the data. If the p-value is less than 0.05 we can confidently say there is a difference between the distributions of the two groups.

# In[16]:

# Perform a MWW test
z_stat, p_val = stats.ranksums(virginica, versicolor)

# Print the p-value. Note I have specified I want to print a floating point decimal with 15 decimals after the period
print('MWW RankSum p-value %0.15f' % p_val)


# Again using seaborn, we can visualize the distributions. Visually, you can see how different the distributions are. Ignore the deprecation warning.

# In[17]:

# Visualizing the distributions
sns.distplot(virginica)
sns.distplot(versicolor)


# **Next Steps**  
# I hope you enjoyed this Python tutorial. Feel free to reach out with any questions that you may have. I encourage you to explore the online tutorials, as well as Stack Overflow. Chances are if you have a question, someone else has already asked it. Happy coding.

# # Header

# In[ ]:

# comments

