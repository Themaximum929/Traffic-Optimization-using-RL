import tensorflow as tf

model = tf.keras.models.load_model(r'c:\Users\maxch\MaxProjects\NLP\best_model.h5', compile=False)

print("Model inputs:")
for i, inp in enumerate(model.inputs):
    print(f"  Input {i}: {inp.name}, shape: {inp.shape}")

print("\nModel output:")
print(f"  {model.output.name}, shape: {model.output.shape}")

print("\nModel summary:")
model.summary()
