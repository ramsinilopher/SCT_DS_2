#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data =pd.read_csv(r"C:\Users\ramsi\Downloads\archive (4)\Titanic-Dataset.csv")


# # Data Information

# In[3]:


data.head()


# In[4]:


data.dtypes


# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


data.describe()


# # Checking for null values

# In[8]:


data.isnull().sum()


# # Handling missing values

# In[9]:


# Replace missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)


# In[10]:


# Replace missing Embarked  with the mode of the Fare column
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# In[11]:


# Drop the Cabin column
data.drop(columns = ['Cabin'],inplace = True)


# In[12]:


# Check for remaining missing values
print(data.isnull().sum())


# In[13]:


data.drop_duplicates()


# # Encode categorical variables

# In[14]:


data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


# # Exploratory Data Analysis (EDA)

# In[15]:


# 1. Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# Observation: The distribution of Age shows a mix of younger and older passengers, with a peak around 20–30 years.
# Insight: Majority of passengers were young adults.
# 

# In[16]:


# 2. Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()


# Observation: More passengers did not survive compared to those who did.

# In[17]:


# 3. Survival by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Survival Rate')
plt.show()


# Observation: Females had a significantly higher survival rate than males.
# Insight: Gender played a critical role in survival, likely due to "women and children first" protocols.
# 

# In[18]:


# 4. Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()



# Observation: First-class passengers had the highest survival rate, followed by second-class, with third-class having the lowest.
# Insight: Socioeconomic status influenced survival chances.

# In[19]:


# 5. Correlation Heatmap
plt.figure(figsize=(10, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Observation:
# Survived shows a strong positive correlation with Sex.
# Fare correlates positively with survival, while Pclass correlates negatively.
# Insight: Higher fares and first-class status improved survival odds

# In[20]:


# 6. Survival by Gender and Passenger Class
plt.figure(figsize=(8, 6))
sns.catplot(x='Pclass', y='Survived', hue='Sex', kind='bar', data=data)
plt.title('Survival by Gender and Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()


# Observation:
# Female passengers in first and second class had the highest survival rates.
# Males in third class had the lowest survival rates.
# 

# In[21]:


# 7. Pairplot to Explore Relationships
sns.pairplot(data, hue='Survived', vars=['Age', 'Fare', 'Pclass'])
plt.show()


# Observation: Relationships among Age, Fare, and Pclass were visualized:
# High fares correspond to first-class passengers.
# Survival patterns are distinguishable in pairwise plots.
# Insight: Multi-dimensional relationships highlight how these variables interact.

# # SUMMARY
# Gender: Women had a much higher survival rate.
# Passenger Class: Socioeconomic status significantly influenced survival, with first-class passengers being the safest.
# Fare: Higher fare passengers had better survival odds.
# Age: Younger passengers showed a slightly higher survival rate.

# In[ ]:




