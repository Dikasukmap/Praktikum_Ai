# In[1]:import lib
# menimport libtari CSV untuk mengolah data ber ekstensi csv
import csv 
#kemudian mengimport librari Image yang berguna untuk dari PIL atau Python Imaging Library yang berguna untuk mengolah data berupa gambar
from PIL import Image as pil_image 
# kemudian mengimport librari keras yang menggunakan method preprocessing yang digunakan untuk membuat neutal network
import keras.preprocessing.image

# In[2]:load all images (as numpy arrays) and save their classes
#membuat variabel imgs dengan variabel kosong
imgs = []
#membuat variabel classes dengan variabel kosong
classes = []
#membuaka file hasy-data-labels.csv yang berada di folede HASYv2 yang di inisialisasi menjadi csvfile
with open('HASYv2/hasy-data-labels.csv') as csvfile:
    #membuat variabel csvreader yang berisi method csv.reader yang membaca variabel csvfile
    csvreader = csv.reader(csvfile)
    # membuat variabel i dengan isi 0
    i = 0
    # membuat looping pada variabel csvreader
    for row in csvreader:
        # dengan ketentuan jika i lebihkecil daripada o
        if i > 0:
            # dibuat variabel img dengan isi keras untuk aktivasi neural network fungsi yang membaca data yang berada dalam folder HASYv2 dengan input nilai -1.0 dan 1.0
            img = keras.preprocessing.image.img_to_array(pil_image.open("HASYv2/" + row[0]))
            # neuron activation functions behave best when input values are between 0.0 and 1.0 (or -1.0 and 1.0),
            # so we rescale each pixel value to be in the range 0.0 to 1.0 instead of 0-255
            #membagi data yang ada pada fungsi img sebanyak 255.0
            img /= 255.0
            # menambah nilai baru pada imgs pada row ke 1 2 dan dilanjutkan dengan variabel img
            imgs.append((row[0], row[2], img))
            # menambahkan nilai pada row ke 2 pada variabel classes
            classes.append(row[2])
            # penambahan nilai satu pada variabel i
        i += 1

# In[3]:shuffle the data, split into 80% train, 20% test
# mengimport library random 
import random
# melakukan random pada vungsi imgs
random.shuffle(imgs)
# membuat variabel split_idx dengan nilai integer 80 persen dikali dari pengembalian jumlah dari variabel imgs
split_idx = int(0.8*len(imgs))
# membuat variabel train dengan isi lebih besar split idx
train = imgs[:split_idx]
# membuat variabel test dengan isi lebih kecil split idx
test = imgs[split_idx:]

# In[4]: 
# mengimport librari numpy dengan inisial np
import numpy as np
# membuat variabel train input dengan np method asarray yang mana membuat array dengan isi row 2 dari data train
train_input = np.asarray(list(map(lambda row: row[2], train)))
# membuat test input input dengan np method asarray yang mana membuat array dengan isi row 2 dari data test
test_input = np.asarray(list(map(lambda row: row[2], test)))
# membuat variabel train_output dengan np method asarray yang mana membuat array dengan isi row 1 dari data train
train_output = np.asarray(list(map(lambda row: row[1], train)))
# membuat variabel test_output dengan np method asarray yang mana membuat array dengan isi row 1 dari data test
test_output = np.asarray(list(map(lambda row: row[1], test)))

# In[5]: import encoder and one hot
# mengimport librari LabelEncode dari sklearn
from sklearn.preprocessing import LabelEncoder
# mengimport librari OneHotEncoder dari sklearn
from sklearn.preprocessing import OneHotEncoder

# In[6]:convert class names into one-hot encoding

# membuat variabel label_encoder dengan isi LabelEncoder
label_encoder = LabelEncoder()
# membuat variabel integer_encoded yang berfungsi untuk mengkonvert variabel classes kedalam bentuk integer
integer_encoded = label_encoder.fit_transform(classes)

# In[7]:then convert integers into one-hot encoding
# membuat variabel onehot_encoder dengan isi OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
# mengisi variabel integer_encoded dengan isi integer_encoded yang telah di convert pada fungsi sebelumnya
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# mengkonvert variabel integer_encoded kedalam onehot_encoder
onehot_encoder.fit(integer_encoded)

# In[8]:convert train and test output to one-hot
# mengkonvert data train output  mengguanakn variabel label_encoder kedalam variabel train_output_int
train_output_int = label_encoder.transform(train_output)
# mengkonvert variabel train_output_int kedalam fungsi onehot_encoder 
train_output = onehot_encoder.transform(train_output_int.reshape(len(train_output_int), 1))
# mengkonvert data test_output mengguanakn variabel label_encoder kedalam variabel test_output_int
test_output_int = label_encoder.transform(test_output)
# mengkonvert variabel test_output_int kedalam fungsi onehot_encoder 
test_output = onehot_encoder.transform(test_output_int.reshape(len(test_output_int), 1))
# membuat variabel num_classes dengan isi variabel label_encoder dan classess
num_classes = len(label_encoder.classes_)
# mencetak hasil dari nomer Class beruapa persen 
print("Number of classes: %d" % num_classes)

# In[9]: import sequential
# mengimport librari Sequential dari Keras
from keras.models import Sequential
# mengimport librari Dense, Dropout, Flatten dari Keras
from keras.layers import Dense, Dropout, Flatten
# mengimport librari Conv2D, MaxPooling2D dari Keras
from keras.layers import Conv2D, MaxPooling2D

# In[10]: desain jaringan
# membuat variabel model dengan isian librari Sequential
model = Sequential()
# variabel model di tambahkan librari Conv2D tigapuluh dua bit dengan ukuran kernel 3 x 3 dan fungsi penghitungan relu dang menggunakan data train_input
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=np.shape(train_input[0])))
# variabel model di tambahkan dengan lib MaxPooling2D dengan ketentuan ukuran 2 x 2 pixcel 
model.add(MaxPooling2D(pool_size=(2, 2)))
# variabel model di tambahkan dengan librari Conv2D 32bit dengan kernel 3 x 3
model.add(Conv2D(32, (3, 3), activation='relu'))
# variabel model di tambahkan dengan lib MaxPooling2D dengan ketentuan ukuran 2 x 2 pixcel 
model.add(MaxPooling2D(pool_size=(2, 2)))
# variabel model di tambahkan librari Flatten
model.add(Flatten())
# variabel model di tambahkan librari Dense dengan fungsi tanh
model.add(Dense(1024, activation='tanh'))
# variabel model di tambahkan librari dropout untuk memangkas data tree sebesar 50 persen
model.add(Dropout(0.5))
# variabel model di tambahkan librari Dense dengan data dari num_classes dan fungsi softmax
model.add(Dense(num_classes, activation='softmax'))
# mengkompile data model untuk mendapatkan data loss akurasi dan optimasi
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# mencetak variabel model kemudian memunculkan kesimpulan berupa data total parameter, trainable paremeter dan bukan trainable parameter
print(model.summary())

# In[11]: import sequential
# mengimport librari keras callbacks
import keras.callbacks
# membuat variabel tensorboard dengan isi lib keras 
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/mnist-style')

# In[12]: 5menit kali 10 epoch = 50 menit
# fungsi model titambahkan metod fit untuk mengetahui perhitungan dari train_input train_output
model.fit(train_input, train_output,
# dengan batch size 32 bit 
          batch_size=32,
          epochs=10,
          verbose=2,
          validation_split=0.2,
          callbacks=[tensorboard])

score = model.evaluate(test_input, test_output, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# In[13]:try various model configurations and parameters to find the best

import time

results = []
for conv2d_count in [1, 2]:
    for dense_size in [128, 256, 512, 1024, 2048]:
        for dropout in [0.0, 0.25, 0.50, 0.75]:
            model = Sequential()
            for i in range(conv2d_count):
                if i == 0:
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))
                else:
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(dense_size, activation='tanh'))
            if dropout > 0.0:
                model.add(Dropout(dropout))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            log_dir = './logs/conv2d_%d-dense_%d-dropout_%.2f' % (conv2d_count, dense_size, dropout)
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)

            start = time.time()
            model.fit(train_input, train_output, batch_size=32, epochs=10,
                      verbose=0, validation_split=0.2, callbacks=[tensorboard])
            score = model.evaluate(test_input, test_output, verbose=2)
            end = time.time()
            elapsed = end - start
            print("Conv2D count: %d, Dense size: %d, Dropout: %.2f - Loss: %.2f, Accuracy: %.2f, Time: %d sec" % (conv2d_count, dense_size, dropout, score[0], score[1], elapsed))
            results.append((conv2d_count, dense_size, dropout, score[0], score[1], elapsed))


# In[14]:rebuild/retrain a model with the best parameters (from the search) and use all data
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# In[15]:join train and test data so we train the network on all data we have available to us
model.fit(np.concatenate((train_input, test_input)),
          np.concatenate((train_output, test_output)),
          batch_size=32, epochs=10, verbose=2)

# In[16]:save the trained model
model.save("mathsymbols.model")

# In[17]:save label encoder (to reverse one-hot encoding)
np.save('classes.npy', label_encoder.classes_)


# In[18]:load the pre-trained model and predict the math symbol for an arbitrary image;
# the code below could be placed in a separate file

import keras.models
model2 = keras.models.load_model("mathsymbols.model")
print(model2.summary())

# In[19]:restore the class name to integer encoder
label_encoder2 = LabelEncoder()
label_encoder2.classes_ = np.load('classes.npy')

def predict(img_path):
    newimg = keras.preprocessing.image.img_to_array(pil_image.open(img_path))
    newimg /= 255.0

    # do the prediction
    prediction = model2.predict(newimg.reshape(1, 32, 32, 3))

    # figure out which output neuron had the highest score, and reverse the one-hot encoding
    inverted = label_encoder2.inverse_transform([np.argmax(prediction)]) # argmax finds highest-scoring output
    print("Prediction: %s, confidence: %.2f" % (inverted[0], np.max(prediction)))

# In[20]: grab an image (we'll just use a random training image for demonstration purposes)
predict("HASYv2/hasy-data/v2-00010.png")

predict("HASYv2/hasy-data/v2-00500.png")

predict("HASYv2/hasy-data/v2-00700.png")

