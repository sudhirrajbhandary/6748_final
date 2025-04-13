from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

#print display
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

#main imdb file
path_train = 'data/train_df.csv'
data_train = pd.read_csv(path_train) #reads data
data_train = data_train.fillna(0)

path_validation = 'data/validation_df.csv'
data_validation = pd.read_csv(path_validation) #reads data
data_validation = data_validation.fillna(0)

path_test = 'data/test_df.csv'
data_test = pd.read_csv(path_test) #reads data
data_test = data_test.fillna(0)

selected_columns = ['runtime','budget',
          'Action','Adventure','Animation','Comedy','Crime','Documentary','Drama',
          'Family','Fantasy','History','Horror','Music','Mystery','Romance',
          'Science Fiction','TV Movie','Thriller','War','Western',
          'production_companies_20th Century Fox','production_companies_Columbia Pictures',
          'production_companies_Metro-Goldwyn-Mayer','production_companies_New Line Cinema',
          'production_companies_Paramount','production_companies_Universal Pictures',
          'production_companies_Walt Disney Pictures','production_companies_Warner Bros. Pictures',
          'production_companies_other',
          'Mapped Value_Africa','Mapped Value_Asia','Mapped Value_Australia',
          'Mapped Value_Europe','Mapped Value_France','Mapped Value_Germany',
          'Mapped Value_India','Mapped Value_North America','Mapped Value_South America',
          'Mapped Value_United Kingdom','Mapped Value_United States of America',
          'spoken_languages_English','spoken_languages_French',
          'spoken_languages_Spanish','spoken_languages_other',
          'dir_total', 'dir_blockbuster', 'dir_success', 'dir_flop',
          'actor_total', 'actor_blockbuster', 'actor_success', 'actor_flop',
          'dir_act_total', 'dir_act_blockbuster', 'dir_act_success']

def extract_knn_fields(x):
    #print(len(selected_columns))
    return x[selected_columns]


x_train = extract_knn_fields(data_train)
x_validation = extract_knn_fields(data_validation)
x_test = extract_knn_fields(data_test)

y_train = data_train['success_level']
y_validation = data_validation['success_level']
y_test = data_test['success_level']


#print(len(x_train))
#print(y_train.head())

####
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelEncoder


label_encoder_test = LabelEncoder()
label_encoder_validation = LabelEncoder()
label_encoder_train = LabelEncoder()

y_train_labels = label_encoder_train.fit_transform(y_train)  # Convert textual labels to numeric labels (0, 1, 2)
y_test_labels = label_encoder_test.fit_transform(y_test)  # Convert textual labels to numeric labels (0, 1, 2)

# One-hot encode the labels for multiclass classification
y_train_one_hot = tf.keras.utils.to_categorical(y_train_labels, num_classes=3)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_labels, num_classes=3)

print("y_train_one_hot",y_train_one_hot)
print("y_test_one_hot",y_test_one_hot)

model = Sequential([
    Dense(64, activation='relu', input_shape=(56,)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',  # Adam optimizer
              loss='categorical_crossentropy',  # Loss function for multiclass classification
              metrics=['accuracy'])  # Metrics to track during training

model.fit(x_train, y_train_one_hot, epochs=5)

loss, accuracy = model.evaluate(x_test, y_test_one_hot)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Get predictions
y_pred_train  = model.predict(x_train)
y_pred_test  = model.predict(x_test)
print("y_pred_train: ", y_pred_train)
print("y_pred_test: ", y_pred_test)

y_pred_train = np.argmax(y_pred_train, axis=1)
y_pred_test = np.argmax(y_pred_test, axis=1)

y_pred_train = label_encoder_train.inverse_transform(y_pred_train)
y_pred_test = label_encoder_test.inverse_transform(y_pred_test)

print("y_pred_train",y_pred_train)
print("y_pred_test",y_pred_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row sum

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Blockbuster', 'Flop', 'Success'],
            yticklabels=['Blockbuster', 'Flop', 'Success'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Percentage)')
plt.show()
