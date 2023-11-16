#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from numpy import loadtxt
from keras.models import load_model
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard
import time


# In[ ]:


# CREATE DIRECTORY OF AUDIO FILES 
audio = "C:/Users/jihed/OneDrive/Desktop/jihed/Speech_Emotion_Recognition-master/Speech_Emotion_Recognition-master/Audio_Song_Actors_01-24"
actor_folders = os.listdir(audio) #list files in audio directory
actor_folders.sort() 
actor_folders[0:5]


# In[ ]:


# CREATE FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
emotion = []
gender = []
actor = []
file_path = []
for i in actor_folders:
    filename = os.listdir(audio +'/'+ i) #iterate over Actor folders
    for f in filename: # go through files in Actor folder
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        actor.append(int(part[6]))
        bg = int(part[6])
        if bg%2 == 0:
            bg = "female"
        else:
            bg = "male"
        gender.append(bg)
        file_path.append(audio+ '/' + i + '/' +f)


# In[ ]:


# PUT EXTRACTED LABELS WITH FILEPATH INTO DATAFRAME
audio_df = pd.DataFrame(emotion)
audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
audio_df.columns = ['gender','emotion','actor']
audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)

pd.set_option('display.max_colwidth', -1)
audio_df.sample(10)


# In[ ]:


# LOOK AT DISTRIBUTION OF CLASSES
audio_df.emotion.value_counts().plot(kind='bar')


# In[ ]:


df = pd.DataFrame(columns=['mel_spectrogram'])

counter = 0

for index, path in enumerate(audio_df.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128, fmax=8000)
    db_spec = librosa.amplitude_to_db(spectrogram)
    log_spectrogram = np.mean(db_spec, axis=0)
    df.loc[counter] = [log_spectrogram]
    counter += 1

df.head()




# In[ ]:


# ITERATE OVER ALL AUDIO FILES AND EXTRACT LOG MEL SPECTROGRAM MEAN VALUES INTO DF FOR MODELING 
df1 = pd.DataFrame(columns=['mfcc'])

counter=0

for index,path in enumerate(audio_df.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
    
    #get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
    db_spec = librosa.power_to_db(spectrogram)
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    mfcc=np.mean(mfcc,axis=0)
    df1.loc[counter] = [mfcc]
    counter=counter+1   


df1.head()


# In[ ]:


# ITERATE OVER ALL AUDIO FILES AND EXTRACT LOG MEL SPECTROGRAM MEAN VALUES INTO DF FOR MODELING 
df4 = pd.DataFrame(columns=['zcr'])

counter=0

for index,path in enumerate(audio_df.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
    zcr = librosa.feature.zero_crossing_rate(y=X)
    zcr = np.mean(zcr, axis= 0)
    
    df4.loc[counter] = [zcr]
    counter=counter+1   


df4.head()


# In[ ]:


#combining the two dataframes
for index, row in df.iterrows():
    for index1, row1 in df1.iterrows():
        if index==index1:
            np.append(df['mel_spectrogram'].iloc[index],df1['mfcc'].iloc[index1])


# In[ ]:


#combining the two dataframes
for index, row in df.iterrows():
    for index1, row1 in df4.iterrows():
        if index==index1:
            np.append(df['mel_spectrogram'].iloc[index],df4['zcr'].iloc[index1])


# In[2]:


# TURN ARRAY INTO LIST AND JOIN WITH AUDIO_DF TO GET CORRESPONDING EMOTION LABELS
df_combined = pd.concat([audio_df,pd.DataFrame(df['mel_spectrogram'].values.tolist())],axis=1)
df_combined = df_combined.fillna(0)
df_combined.head()
# DROP PATH COLUMN FOR MODELING
df_combined.drop(columns='path',inplace=True)


# In[ ]:


# TRAIN TEST SPLIT DATA
train,test = train_test_split(df_combined, test_size=0.2, random_state=0,
                               stratify=df_combined[['emotion','gender','actor']])

X_train = train.iloc[:, 3:]
y_train = train.iloc[:,:2].drop(columns=['gender'])
print(X_train.shape)

X_test = test.iloc[:,3:]
y_test = test.iloc[:,:2].drop(columns=['gender'])
print(X_test.shape)


# In[ ]:


# NORMALIZE DATA standard deviation (z score)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std


# In[ ]:


# TURN DATA INTO ARRAYS FOR KERAS
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[ ]:


#OneHotEncoding
from tensorflow.keras.utils import to_categorical
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

print(y_test[0:10])


# In[ ]:


# create the RNN model
model = Sequential()

model.add(SimpleRNN(units=1024, input_shape=(X_train.shape[1], 1), 
                    return_sequences=True, 
                    kernel_regularizer=regularizers.l2(1e-4)))

# add the second SimpleRNN layer with 512 units
model.add(SimpleRNN(units=512, return_sequences=False))

# add the output layer with 6 neurons
model.add(layers.Dense(6, activation='sigmoid',))
model.add(layers.Flatten())

# compile the model with categorical_crossentropy and SGD optimizer
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

#visualizing with TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}".format("RNN,1024units"))
model.summary()
model_history=model.fit(X_train, y_train,batch_size=32, epochs=100, validation_data=(X_test, y_test),callbacks=[tensorboard])

