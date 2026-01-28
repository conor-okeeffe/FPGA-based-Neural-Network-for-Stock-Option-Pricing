import tensorflow as tf

model = tf.keras.models.load_model("Models/model.h5", compile=False)

print(model.summary())

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__)
