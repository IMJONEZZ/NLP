from preprocessing import num_encoder_tokens, num_decoder_tokens, decoder_target_data, encoder_input_data, decoder_input_data, decoder_target_data

from tensorflow import keras
# Add Dense to the imported layers
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# UNCOMMENT THE TWO LINES BELOW IF YOU ARE GETTING ERRORS ON A MAC
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Choose a dimensionality
latent_dim = 256

# Choose a batch size
# and a larger number of epochs:
batch_size = 16
epochs = 500

# Encoder training setup
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# Decoder training setup:
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Building the training model:
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("Model summary:\n")
training_model.summary()
print("\n\n")

# Compile the model:
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# print("Training the model:\n")
# Train the model:
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

training_model.save('training_model.h5')
