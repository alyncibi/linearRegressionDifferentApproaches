
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


# model: 1D Linear regression
# Formula: yi = theta0 + theta1.xi1 , for the i-th example of the training set
# goal: predict theta0 and theta1; the parameters of the model
# algortihm1: normal function direct formula, to calculate the best (theta0,theta1)
# algorithm2: gradient descent, to calculate the best (theta0,theta1)


# In[3]:


# we generate the dataset : a column vector of 100 random input 


# In[4]:


x = np.random.rand(100,1)


# In[5]:


# we generate a random output vector


# In[6]:


y = np.random.rand(100,1) 


# In[7]:


# we generate a 2nd ouput vector, which is known ( for later comparaison ) as : yy = 4 + 3*x , so theta is know right ?


# In[8]:


yy = 4 + 3 * x


# In[9]:


# algorithm1


# In[10]:


# We prepare th input vector to linear algebra operation (Normal function algorithm # another way of achieving it)


# In[11]:


X = np.c_[np.ones((100,1)),x]


# In[12]:


# We calculate the optimal theta, using normal function algorithm, which gives direct analytic access to theta_best


# In[13]:


# case 1


# In[14]:


theta_best = np.linalg.inv(X.T @ X) @ X.T @ y


# In[15]:


# case 2

theta_best_2 = np.linalg.inv(X.T @  X) @ X.T @ yy
# In[16]:


print(theta_best)


# In[17]:


plt.plot(x,y,"b.")
# we plot the prediction
plt.plot(x,X@theta_best,"r.")


# In[18]:


print(theta_best_2)


# In[ ]:


plt.plot(x,yy,"b.")
# we plot the prediction
plt.plot(x,X @ theta_best_2,"r.")
# plots are superposed ;)


# In[ ]:


# algorithm2


# In[ ]:


eta = 0.1


# In[ ]:


n_iterations = 1000


# In[ ]:


m = len(X)


# In[ ]:


theta = np.random.randn(2,1)


# In[ ]:


for i in range(n_iterations):
    gradients = 2/m * X.T @ (X @ theta - y)
    theta = theta - eta * gradients
    


# In[ ]:


print(theta) # should be near to theta_best


# In[ ]:


plt.plot(x,X @ theta, "r.")
plt.plot(x,y,"b.")


# In[ ]:


theta_2 = np.random.randn(2,1)


# In[ ]:


# for j in range(n_iterations):
    gradients = 2/m * X.T @ (X @ theta_2 - yy)
    theta_2 = theta_2 - eta * gradients


# In[ ]:


print(theta_2) # should be near to theta_best_2 i.e (4,3)


# In[ ]:


plt.plot(x,yy,"b.")
plt.plot(x,X @ theta_2, "r.")

