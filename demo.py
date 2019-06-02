import struct
import numpy as np
from scipy import signal as sg
import scipy
import wave
import os
from PIL import Image
import glob
import imageio
import matplotlib.pyplot as plt
import random

import tensorflow as tf
tf.enable_eager_execution()

import pyaudio
import random

import threading
import time
import queue

import sys

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4*4*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
      
    model.add(tf.keras.layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256) # Note: None is the batch size
    
    model.add(tf.keras.layers.Conv2DTranspose(192, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 192)  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))  
    assert model.output_shape == (None, 32, 32, 64)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))  
    assert model.output_shape == (None, 64, 64, 32)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 1)
  
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
      
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
       
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
     
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_sound_clip(checkpoint_to_load, random_vector_for_generation):

    checkpoint_pointer = './training_checkpoints\\ckpt-' + str(checkpoint_to_load)
    checkpoint.restore(checkpoint_pointer)
    
   
    predictions = generator(random_vector_for_generation, training=False)
    generated = np.asanyarray(predictions[0, :, :, 0] * 127.5 + 127.5, dtype=np.uint8)
    data = []
    for y in range(128):
        for x in range(128): 
            calnice = math_nice(generated[y][x])
            part3 = calnice
            part2 = 0
            part1 = 0
            data.append(part1)
            data.append(part2)
            data.append(part3)
    return bytes(data)

def math_nice(s): #fix so the image data
    if s < 128:
        return abs(s - 128)
    else:
        return abs((s - 128) - 255)

def consume(q,Rate):
    P = pyaudio.PyAudio()
    stream = P.open(rate=Rate, format=pyaudio.paInt24, channels=1, output=True)

    while(True):
        name = threading.currentThread().getName()
        print("Thread: {0} start get item from queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')))
        stream.write(q.get())
        print("Thread: {0} finish process item from queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')))
    stream.close()
    P.terminate()
 
 
def producer(q,models_to_use):
    while(True):
        name = threading.currentThread().getName()
        rand = random.randint(1,len(models_to_use)) - 1
        print("Thread: {0} start put item into queue[current size = {1}] at time = {2} Used model = {3} \n".format(name, q.qsize(), time.strftime('%H:%M:%S'),models_to_use[rand]))

        q.put(generate_sound_clip(models_to_use[rand], tf.random_normal([1,100])))
        print("Thread: {0} successfully put item into queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')))
 
if __name__ == '__main__':
    print(sys.argv)
    try:
        rate = int(sys.argv[1])
    except TypeError:
        print("First value is not an integer")
        quit()
    q = queue.Queue(maxsize = 10)
    models_to_use = []
    if len(sys.argv) > 2:
        x = 0
        for inputs in sys.argv:
            if x <= 1:
                x += 1
                continue
            try:
                if (int(inputs) > 0) & (int(inputs) <= 37):
                    models_to_use.append(inputs)
            except TypeError:
                continue
    else: 
        for x in range(36):
            models_to_use.append(x + 1)
        


    threads_num_consumer = 1
    threads_num_producer = 1
    for i in range(threads_num_consumer):
        t = threading.Thread(name = "ConsumerThread-"+str(i), target=consume, args=(q,rate,))
        t.daemon = True
        t.start()
 

    for i in range(threads_num_producer):
        t = threading.Thread(name = "ProducerThread"+str(i), target=producer, args=(q,models_to_use))
        t.daemon = True
        t.start()
 
    q.join()

    while True:
        time.sleep(1)
 
 