# In[1]
import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

# In[2]
def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()

# In[3]
display_mfcc('donlod/genres/disco/disco.00035.au')

# In[4]
def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

# In[5]
def generate_features_and_labels():
    all_features = []
    all_labels = []

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for genre in genres:
        sound_files = glob.glob('donlod/genres/'+genre+'/*.au')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    return np.stack(all_features), onehot_labels

# In[6]
features, labels = generate_features_and_labels()

# In[7]
print(np.shape(features))
print(np.shape(labels))

# In[8]
training_split = 0.8

# In[9]
# last column has genre, turn it into unique ids
alldata = np.column_stack((features, labels))

# In[10]
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

# In[11]
print(np.shape(train))
print(np.shape(test))

# In[12]
train_input = train[:,:-10]
train_labels = train[:,-10:]

test_input = test[:,:-10]
test_labels = test[:,-10:]

# In[13]
print(np.shape(train_input))
print(np.shape(train_labels))

# In[14]
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])

# In[15]
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# In[16]
print(model.summary())

# In[17]
model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)

# In[18]
loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

# In[19]
print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

# In[20]
model.predict(train_input[:1])