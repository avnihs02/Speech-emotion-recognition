import librosa
import soundfile
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import preprocessing 

from sklearn.preprocessing import MinMaxScaler
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'fearful', 'disgust','neutral','sad','angry','surprised']



def extract_feature(file_name, mfcc, chroma,spectral_centroid,spectral_bandwidth,spectral_rolloff,spectral_contrast,rms,spectral_flatness):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
            
            
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))             
           
            
        if spectral_centroid:
            spectral_centroid=np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate))
            result=np.hstack((result, spectral_centroid)) 
        
        if spectral_bandwidth:
           spectral_bandwidth=np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate).T)
#           print(spectral_bandwidth)
           result=np.hstack((result, spectral_bandwidth)) 
           
        if spectral_rolloff:
           spectral_rolloff=np.mean(librosa.feature.spectral_rolloff(y=X, sr=sample_rate).T)
#           print(spectral_rolloff)
           result=np.hstack((result, spectral_rolloff))
        
        if spectral_contrast:
           spectral_contrast=np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate))
           result=np.hstack((result, spectral_contrast))
           
        if rms:
           rms=np.mean(librosa.feature.rms(y=X).T,axis=0)
           result=np.hstack((result, rms))
           
        if spectral_flatness:
           spectral_flatness=np.mean(librosa.feature.spectral_flatness(y=X))
           result=np.hstack((result, spectral_flatness))
        
        return result

################### feature extraction by calling the files ############################

def load_data():
    X,y=[],[]
    for file in glob.glob("D:\\DBDA\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        feature=extract_feature(file, mfcc=True, chroma=True,spectral_centroid=True,spectral_bandwidth=True,spectral_rolloff=True,spectral_contrast=True,rms=True,spectral_flatness=True)
        X.append(feature)
        y.append(emotion)
        
    return X,y



#X is the feature vector ,,, 58 features extracted for every 1440 audio files 
#y is the response variable 

X_audio,y_emotions=load_data()

import matplotlib.pyplot as plt
import wave
spf = wave.open("D:\\DBDA\\PROJECT\\speech-emotion-recognition-ravdess-data\\Actor_01\\03-01-08-02-01-01-01.wav", "r")

signal = spf.readframes(-1)
signal = np.fromstring(signal, "Int16")

plt.figure(1)
plt.title("waveform for surprised emotion")
plt.plot(signal)
plt.show()

features=[]
for i in range(1,59) :
    features.append("feat"+str(i))    


le = preprocessing.LabelEncoder()
y1=le.fit_transform(y_emotions)


X1=pd.DataFrame(X_audio,columns=features)
y=pd.DataFrame(y1,columns=["emotions"])
X=X1.drop(["feat1"],axis=1)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.05, random_state=2019,stratify=y) 

from sklearn.preprocessing import MinMaxScaler
# Create scaler: scaler
scaler = MinMaxScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled,index=x_train.index,columns=x_train.columns)


#data=pd.concat([X,y],axis=1)
#data.to_csv("D:\\DBDA\\PROJECT\\audio_features.csv")



#print((x_train.shape[0], x_test.shape[0]))


print(f'Features extracted: {x_train.shape[1]}')

############Multiclasss-keras#############
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
# Function to create model, required for KerasClassifier
def create_model():
    	# create model
    model = Sequential()
    model.add(Dense(200, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(190, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(220, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(230, activation='relu'))
    #model.add(Dense(5, activation='relu'))
    #model.add(Dense(15, activation='relu'))
    model.add(Dense(8,activation='softmax')) # Output
    # Compile model	   
    model.compile(loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    
    return model


# fix random seed for reproducibility

x_train,x_test,y_train,y_test=load_data(test_size=0.3)

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


le = preprocessing.LabelEncoder()
y_test=le.fit_transform(y_test)
y_train=le.fit_transform(y_train)


y_test=np.asarray(y_test)
x_train=np.asarray(x_train)
y_train=np.asarray(y_train)
x_test=np.asarray(x_test)

# create model

#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
model = KerasClassifier(build_fn=create_model,verbose=1)
# define the grid search parameters
batch_size =np.asarray([10,30,50,70,90,110,130])
epochs = np.asarray([300,400,500,600,700,800,900,1000,1400])
learn_rate = np.asarray([0.001, 0.01, 0.1, 0.2, 0.3])
momentum = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 0.9])

lr_range = np.linspace(0.001,1,30)



param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3)
grid_result = grid.fit(x_train, y_train,validation_data=(x_test,y_test))
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']


#######################################################################





model = Sequential()
model.add(Dense(200, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(190, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(220, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(230, activation='relu'))
#model.add(Dense(5, activation='relu'))
#model.add(Dense(15, activation='relu'))
model.add(Dense(8,activation='softmax')) # Output


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=2,epochs=250, workers=6)

y_pred_prob = model.predict(x_test)
y_pred_prob.shape

loss, acc = model.evaluate(x_test, y_test,verbose=0)
print()
print('Test loss = {:.4f} '.format(loss))
print('Test acc = {:.4f} '.format(acc))
#BEFORE
#1008/1008 - 0s - loss: 0.1476 - accuracy: 0.9395 - val_loss: 2.9458 -
#val_accuracy: 0.5648
#Test loss = 2.5118 
#Test acc = 0.5648 

#AFTER
#1008/1008 - 0s - loss: 0.9551 - accuracy: 0.5863 - val_loss: 1.7997 - 
#val_accuracy: 0.4514
#Test loss = 1.7997 
#Test acc = 0.4514 


#######early stopping#####
from tensorflow.keras.callbacks import EarlyStopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',
        restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)

#val_accuracy: 0.5764