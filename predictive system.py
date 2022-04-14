import numpy as np
import pickle
loaded_model = pickle.load(open('/Users/atharva/Desktop/trained_model.sav', 'rb'))

input_data=(350, 0, 0, 203, 0, 974, 775, 28)

input_data_as_numpy_array = np.asarray(input_data)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
