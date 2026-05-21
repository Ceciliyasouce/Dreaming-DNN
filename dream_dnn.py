import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import Input
from tensorflow.keras.models import Model


image_path = sys.argv[1]          
layer_index = int(sys.argv[2])    
num_steps = int(sys.argv[3])      


def squared_sum_loss(activations):
    return tf.reduce_sum(tf.square(activations))

def dream_step(image, step_size):
    with tf.GradientTape() as tape:
        tape.watch(image)
        activations = dream_model(image)
        loss = squared_sum_loss(activations)

    gradients = tape.gradient(loss, image)
    gradients /= tf.math.reduce_std(gradients) + 1e-8

    image.assign_add(step_size * gradients)
    return loss

img = Image.open(image_path).convert("RGB").resize((224, 224))
img = np.array(img).astype(np.float32)
img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

image = tf.Variable(img[None, ...], dtype=tf.float32)

input_tensor = Input(shape=(224, 224, 3))
base_model = tf.keras.applications.MobileNetV2(
    input_tensor=input_tensor,
    include_top=False,
    weights="imagenet"
)

dream_model = Model(
    inputs=input_tensor,
    outputs=base_model.layers[layer_index].output
)

print(f"Using layer {layer_index}: {base_model.layers[layer_index].name}")

step_size = 0.02
os.makedirs("output", exist_ok=True)

for step in range(num_steps):
    
    loss = dream_step(image, step_size)

    if step % 10 == 0:
        print(f"Step {step}, loss = {loss.numpy():.4f}")

        # Save intermediate image
        temp = image.numpy()[0]
        temp = (temp + 1.0) * 127.5
        temp = np.clip(temp, 0, 255).astype(np.uint8)
        Image.fromarray(temp).save(f"output/step_{step}.jpg")


final_img = image.numpy()[0]
final_img = (final_img + 1.0) * 125
final_img = np.clip(final_img, 0, 255).astype(np.uint8)

output_name = "dream_" + image_path
Image.fromarray(final_img).save(output_name)


plt.figure(figsize=(3, 3))
plt.imshow(final_img)
plt.axis("off")
plt.show()