# Imports
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.utils import to_categorical, normalize
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam

# Leitura do dataset
dataset = pd.read_csv('dataset/ClimaAustralia.csv')

# Retirando tuplas com valores "NaN"
dataset.dropna(inplace = True)

# Plotagem do gráfico de temperaturas máximas no intervalo de ano 2012-2017
dataset['Location'] = dataset['Location'].astype('category').cat.codes
dataset['WindGustDir'] = dataset['WindGustDir'].astype('category').cat.codes
dataset['WindDir9am'] = dataset['WindDir9am'].astype('category').cat.codes
dataset['WindDir3pm'] = dataset['WindDir3pm'].astype('category').cat.codes
dataset['RainToday'] = dataset['RainToday'].astype('category').cat.codes
dataset['RainTomorrow'] = dataset['RainTomorrow'].astype('category').cat.codes

# Deletando coluna Date do dataset
dataset.drop(labels = ['Date'], axis=1, inplace=True)

# Embaralhando tuplas
dataset = shuffle(dataset)

# Tornando a coluna Location única
dataset['Location'].unique()

# Criando dataset de treino e de validação
X = dataset.drop('RainTomorrow', axis=1)
y = dataset['RainTomorrow']

# Normalização dos dados
X = X.values
X = normalize(X)

# Separação do dataset em 80% treino, 10% validação e 10% teste
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)

# Criação do modelo de RNA
model = Sequential()

model.add(Dense(units=128, input_shape=(22,), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=125, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Execução do treinamento da RNA
history = model.fit(x = X_train, y = y_train, epochs=100, validation_data = (X_val, y_val), verbose=1)

# Plotagem do Acurácia do Modelo
plt.title('Acurácia do Modelo')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()

# Plotagem da Perda do Modelo
plt.title('Perdas do Modelo')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()

# Plotagem da acurácia atingida pelo modelo
loss, accuracy = model.evaluate(X_test, y_test)
acc = accuracy * 100
plt.bar(1, acc)
plt.text(0.92, 45, '{acc:.2f}%'.format(acc = acc), fontsize=20)
plt.title('Accuracy')
plt.xticks([])
plt.ylabel('Percent')
plt.show()