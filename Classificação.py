#!/usr/bin/env python
# coding: utf-8

# ## Importando Bibliotecas

# In[9]:


import sklearn
import pandas as pd


# In[ ]:





# ### Leitura dos DataSet

# In[10]:


diabetes = pd.read_csv('diabetes.csv')
diabetes


# In[11]:


batimentos = pd.read_csv('batimentos cardiacos.csv')
batimentos


# In[12]:


colesterol = pd.read_csv('colesterol.csv')
colesterol


# In[ ]:





# ### Separando os dados para teste e treino

# - separei o que vai ser dado de treino e teste 
# - ajustei os dados para ele aprender
# - faço a medição da acurácia

# In[ ]:





# In[13]:


x = diabetes['idade']       #Inclui os dados de teste e treino dentro de uma variavel
y = diabetes['resultado']


# In[ ]:





# # Classificação

# In[14]:


X = diabetes.drop('resultado', axis = 1 ) #separei a label resultados e manter só as features


# In[22]:


y = diabetes['resultado']


# In[26]:


from sklearn.model_selection import train_test_split   #chamo minha função de treino

SEED = 4121988

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state = SEED)


# In[27]:


from sklearn.tree import DecisionTreeClassifier     #importo a arvore de decisao para teste

clf_arvore = DecisionTreeClassifier(random_state=SEED, max_depth=3)


# In[28]:


clf_arvore.fit(x_treino, y_treino)                #coloco meu teste para treino


# In[29]:


clf_arvore.score(x, y)              #calculo acurácia 


# In[30]:


from sklearn.dummy import DummyClassifier   #importei o dummy para entender se acurácia está boa

clf_dummy = DummyClassifier(strategy='most_frequent')

clf_dummy.fit(x_treino, y_treino)

clf_dummy.score(x_teste, y_teste)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




