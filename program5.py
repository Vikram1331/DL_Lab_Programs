import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

text = "HELLO"


chars = sorted(set(text))
char_to_index = {c:i for i,c in enumerate(chars)}
index_to_char = {i:c for i,c in enumerate(chars)}

print("Character Mapping:")
print(char_to_index)

X = [char_to_index[text[i]] for i in range(len(text)-1)]
y = [char_to_index[text[i+1]] for i in range(len(text)-1)]

X = to_categorical(X, len(chars))
y = to_categorical(y, len(chars))

X = X.reshape((X.shape[0], 1, X.shape[1]))

print("Input Shape:", X.shape)

model = Sequential([
    SimpleRNN(8, activation='tanh', input_shape=(1, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=200, verbose=1)

def predict_next_char(char):
    x = to_categorical(
        char_to_index[char],
        len(chars)
    ).reshape((1,1,len(chars)))

    pred = model.predict(x, verbose=0)

    return index_to_char[np.argmax(pred)]

print("\nTesting Prediction:")

test_char = "H"
next_char = predict_next_char(test_char)

print("Input:", test_char)
print("Predicted Next Character:", next_char)

# Generate text
print("\nGenerated Text:")

char = "H"
generated_text = char

for i in range(4):
    char = predict_next_char(char)
    generated_text += char
print (generated_text)