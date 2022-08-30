#!/usr/bin/env python
# coding: utf-8

# ## Importando Bibliotecas

# In[5]:


import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# ### Leitura dos DataSet

# In[4]:


batimentos = pd.read_csv('https://gist.githubusercontent.com/davidneves11/d72e7f49ab01c856acc5d07be4b1a9dd/raw/37631e3a40da92e6261c00fffdf0fb9b869b35dd/batimentos%2520cardiacos.csv')
batimentos.head()


# In[6]:


sns.set()                          #plotei um grafico para visualizar os dados 
plt.figure(figsize=(100, 50))

sns.lmplot(x='Peso', y='Batimentos cardiacos', data=batimentos,line_kws={'color':'red'})
plt.show()


# In[ ]:


x = batimentos[['Peso','Idade']]   #defini os dados para treino
x


# In[ ]:


x = batimentos[['Peso','Idade']]
x


# In[ ]:


y = batimentos['Batimentos cardiacos']
y


# In[7]:


from sklearn.model_selection import train_test_split  #importei o slenar import teste

SEED=4500

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state = SEED)


# In[8]:


from sklearn import linear_model  #Importei o algoritmo de regressão linear.

rgs = linear_model.LinearRegression(fit_intercept=False, normalize=True)


# In[ ]:


rgs.fit(x_treino, y_treino)       #treinei o modelo


# In[ ]:


rgs.score(x_teste, y_teste)    #Calculei a acurácia com o score


# In[ ]:




