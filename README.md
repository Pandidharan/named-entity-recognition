# EX-06 Named Entity Recognition
### Aim:
To develop an LSTM-based model for recognizing the named entities in the text. &emsp;&emsp;&emsp; 

### Problem Statement and Dataset:
Named Entity Recognition (NER) is essential in NLP, focusing on identifying and classifying entities like persons, organizations, and locations in text. This project aims to develop an LSTM-based model to predict entity tags for each word in sentences. The dataset includes words, their POS tags, and standard NER tags such as ORGANIZATION, PERSON, LOCATION, and DATE. Accurate NER enhances applications like information extraction, question answering, and sentiment analysis.
### Design Steps:
- Step:1 Import necessary libraries like pandas, NumPy, and TensorFlow/Keras.   
- Step:2 Read the dataset and use forward fill to handle null values.
- Step:3 Create lists of unique words and tags, and count the number of unique entries
- Step:4 Build dictionaries mapping words and tags to their corresponding index values.
- Step:5 Construct a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, and Time Distributed Dense layers, then compile it for training with the dataset.

## PROGRAM
### Name: Pandidharan.G.R
### Register Number: 212222040111

```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data.head(50)
data = data.fillna(method="ffill")
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())
print("Unique tags are:", tags)

num_words = len(words)
num_tags = len(tags)
num_words

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)
sentences[0]

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
word2idx

plt.hist([len(s) for s in sentences], bins=50)
plt.show()

X1 = [[word2idx[w[0]] for w in s] for s in sentences]
type(X1[0])
X1[0]

max_len = 50
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)
X[0]

y1 = [[tag2idx[w[2]] for w in s] for s in sentences]

y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)
X_train[0]

y_train[0]

input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim=num_words , output_dim=50,input_length=max_len)(input_word)
dropout_layer = layers.SpatialDropout1D(0.1)(embedding_layer)
bidirectional_lstm = layers.Bidirectional(layers.LSTM(units = 100, return_sequences = True, recurrent_dropout=0.1))(dropout_layer)

output = layers.TimeDistributed(layers.Dense(num_tags, activation="softmax"))(bidirectional_lstm)

model = Model(input_word, output)

print('Name:Pandidharan.G.R                 Register Number:212222040111     ')
model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=32,
    epochs=3,
)

print('Name:Pandidharan.G.R                 Register Number:212222040111     ')

# Access the training history from the 'history' variable
import pandas as pd
metrics = pd.DataFrame(history.history)

#Plot the Accuracy and validation accuracy
metrics[['accuracy','val_accuracy']].plot()

print('Name: Pandidharan.G.R                Register Number: 212222040111      ')
metrics[['loss','val_loss']].plot()

i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print('Name: Pandidharan.G.R                Register Number:212222040111       ')
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![accuracy](https://github.com/user-attachments/assets/fc5f0d1d-9b0f-4280-bf1f-32fb8a66fa13)

![loss](https://github.com/user-attachments/assets/ae54803d-c921-4aa5-9626-c6ddfc5e11ed)

### Sample Text Prediction

![prediction](https://github.com/user-attachments/assets/0025ab1c-5d67-461d-9e46-44592dd88377)


## RESULT

Thus,an LSTM-based model has been successfully developed for recognizing the named entities in a text.
