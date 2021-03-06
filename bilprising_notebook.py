
# coding: utf-8

# # Bli en Data Scientist på 20 minutter
# ** JavaZone 2017 **  
# ** simen.eide@finn.no **  
# ** twitter.com/simeneide ** 

# <img class="center-block" src="images/berlingo.png"/>

# <img class="center-block" src="images/caddy.png"/>

# # Tutorial: Finn salgspris på bil

# 
# ### 0. Finn et use-case
# ### 1. Få tak i data
# ### 2. Tren modell på treningsdata
# ### 3. Test modell på noe annet

# ## 1. Få tak i data
# 

# In[1]:


# Alle pakkene kan installeres ved å skrive pip install *pakkenavn* på kommandolinjen.

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from ggplot import *

# må kommenteres ut hvis det kjøres som python-script:
# get_ipython().magic('matplotlib inline')

caddy = pd.read_csv('caddy_jz.csv')
caddy.sample(10)


# In[2]:


ggplot(aes(x='Milage',y='ObjectPrice'), data = caddy) + geom_point()


# In[3]:


ggplot(aes(x='YearModel',y='ObjectPrice'), data = caddy) + geom_point()


# ## Trene / Test
# Del data inn i ett sett dedikert for trening, og ett sett dedidkert for å teste hvor bra modellen var

# In[4]:


np.random.seed(5)
random_numbers = np.random.random(caddy.shape[0])
print(random_numbers[0:10])


# In[5]:


train_index = random_numbers < 0.7

train = caddy[ train_index == True]
test = caddy[ train_index == False]

print('Training set: \t Rows: %d Columns: %d' % train.shape)
print('Test set: \t Rows: %d Columns: %d' %test.shape)


# In[6]:


train.head(5)


# ## Tren en modell

# In[7]:


model = RandomForestRegressor(n_estimators=1000, n_jobs = 5, min_samples_split=10,
                              min_samples_leaf = 5)

def prepare_data(dat):
    X = dat.drop(['ObjectPrice'], axis = 1).fillna(0)
    y = dat[['ObjectPrice']]['ObjectPrice']
    return X, y

X, y = prepare_data(train)
model.fit(X, y)


# ## Sjekk hvor bra modellen funker

# In[8]:


X, y = prepare_data(test)
yhat = model.predict(X)


# In[9]:


result = pd.DataFrame({'truth' : y, 
                    'prediction' : yhat})
result['percentage_diff'] = ((result['prediction'] - result['truth']) / result['truth']*100).round()

print(result.sample(5))


# In[10]:


ggplot(aes("truth","prediction"),data=result) + geom_point() + geom_abline()


# In[11]:


print('Mean Absolute Error: %.1f percent.' % abs(result['percentage_diff']).mean())


# ### Min bil

# In[12]:


prospect_car = caddy[(caddy.ObjectPrice == 265000.0) & (caddy.Milage==37.434)].copy()
X, y = prepare_data(prospect_car)

print('Model Prediction of my car: %d kr' % model.predict(X))


# Oppdaget at bilen jeg så på hadde feil-tolket fritekstfeltet "Effect". Fyll inn 140hk som er maks på disse bilene:

# In[13]:


prospect_car


# In[14]:


prospect_car.Effect = 140
X, y = prepare_data(prospect_car)

print('Model Prediction of my car: %d kr' % model.predict(X))


#   
# # Data Scientist Procedure:
# ### 0. Finn et use-case
# ### 1. Få tak i data
# ### 2. Tren modell på treningsdata
# ### 3. Test modell på noe annet    

# 
# ## github.com/simeneide/car-pricing
# 
# ## simen.eide@finn.no
# 
# ## twitter.com/simeneide 
# 
# ## finn.no/apply-here
# 
# <img class="center-block" src="images/caddy_bed.jpg"/>
# 
