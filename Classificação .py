#!/usr/bin/env python
# coding: utf-8

# ### Import as Bibliotecas

# In[3]:


import pandas as pd
import seaborn as sns


# In[2]:


colesterol = pd.read_csv('https://gist.githubusercontent.com/davidneves11/01b2963f7a8dfd87d79010fbf847b221/raw/685870f4365bcda4e5bb9e342285e0aac37dd556/colesterol.csv')
colesterol.head()


# In[4]:


sns.scatterplot(x = 'pressao_sanguinea_repouso', y = 'colesterol', data = colesterol) #lendo os dados com grafico de dispersao


# In[ ]:


from sklearn.cluster import KMeans     #defini o número de clusters   
kmeans = KMeans(n_clusters = 2, random_state = 9)


# In[ ]:


kmeans.fit(x)
kmeans.labels_                     #treino modelo que defini como X


# In[5]:


sns.scatterplot(x='pressao_sanguinea_repouso', y='colesterol', data= colesterol, hue = kmeans.labels_) #adicionei coluna clusters na base de dados


# In[ ]:


colesterol.groupby('clusters')['colesterol'].mean() #clusters fazendo uma média do colesterol


# In[ ]:





# In[ ]:





# In[ ]:




