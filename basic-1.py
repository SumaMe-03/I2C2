#!/usr/bin/env python
# coding: utf-8

# In[21]:


import turicreate as tc
import pickle


# In[4]:


from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import HoverTool
output_notebook()


# In[22]:


trial = tc.SFrame.read_csv('2018_data.csv')
trial


# In[23]:


trial.groupby('traits',operations={'count':tc.aggregate.COUNT()}).sort('count', ascending= False).head(5)


# In[24]:


giraffe_reviews = trial[trial['State/UT']== 'Arunachal Pradesh']
giraffe_reviews['traits'].show()


# In[25]:


trial['word_count'] = tc.text_analytics.count_words(trial['traits_written'])
trial.head(5)


# In[26]:


selected_words = ['sea', 'coast', 'high', 'vegetation', 'altitude', 'climate', 'change',
                  'annual', 'rise', 'tourism','humidity', 'rate','urbanization','population', 
                  'lack', 'water', 'tap', 'malnutrition', 'temperature', 
                  'contamination', 'nitrate', 'accessibility', 'rainfall']
# 'climate', 'change', 'annual', 'rise', 'tourism', 'humidity', 'rate', 
# Loop through word counts to create a classifier for only a few words 
# Created an individual column for each item 
for word in selected_words:
    trial[word] = trial['word_count'].apply(lambda counts: counts.get(word, 0))

trial.head(5)


# In[27]:


trial['traits'].show


# In[28]:


trial = trial[trial['Dengue cases']!= 0]

#positive sentiment = 4-star or 5-star reviews
trial['critical'] = trial['Dengue cases'] >= 2000

trial.head(5) 


# In[29]:


trial['Dengue cases'].show()


# In[30]:


train_data,test_data = trial.random_split(.7,seed=0) 


# In[31]:


training_model = tc.logistic_classifier.create(train_data,target='critical', features=['Dengue cases'],validation_set=test_data)


# In[32]:


predictions = training_model.classify(test_data)
print (predictions)


# In[33]:


roc = training_model.evaluate(test_data, metric= 'roc_curve')
roc


# In[34]:


result = training_model.evaluate(test_data)
print ("Accuracy             : {}".format(result['accuracy']))
print ("Area under ROC Curve : {}".format(result['auc']))
print ("Confusion Matrix     : \n{}".format(result['confusion_matrix']))
print ("F1_score             : {}".format(result['f1_score']))
print ("Precision            : {}".format(result['precision']))
print ("Recall               : {}".format(result['recall']))
print ("Log_loss             : {}".format(result['log_loss']))


# In[36]:


trialdata = trial.copy()
trialdata['predicted_traits'] = training_model.predict(trialdata, output_type = 'probability')
s = input("Enter state or Union Territory: ")
giraffe_reviews = trialdata[trialdata['State/UT']== s]
giraffe_reviews = giraffe_reviews.sort('predicted_traits', ascending=False)
print('Traits of Dengue in '+ s +': ', giraffe_reviews[0]['traits_written'])
print('\n\n')


# In[42]:


# define your object here
class training_model:
    def __init__(self):
        self.attribute = 'value'
        self._base_ptr = None
        self.thisptr = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_base_ptr']
        del state['thisptr']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

# pickle your object using the custom pickling method
import pickle
with open('basic-1.pkl', 'wb') as f:
    pickle.dump(training_model(), f)


# In[ ]:




