import wave
import struct
from PIL import Image
import numpy as np
import scipy.io.wavfile as wavf
import struct
import os
import re
import sys

def toDataSet(fromPath, toPath, dimention_x, dimention_y):
        print("Start converting wav files from ", fromPath, " to ", toPath, " as png data")
        song = 0
        for targetFile in os.listdir(fromPath):
            print("Start convert of file: ", targetFile)
            waveFile = wave.open(fromPath + '\\' + targetFile, 'r')
            print(waveFile.getparams())
            count = 0
            posCounter = 0
            while ((posCounter + 1) * dimention_x * dimention_y) < waveFile.getnframes():
                RGBA = []
                for x in range(dimention_x):
                    buffert = []
                    for y in range(dimention_y):
                        frame = waveFile.readframes(1)
                        frame_nice = math_nice(frame[1])
                        count += 1
                        buffert.append([frame_nice,frame_nice,frame_nice])
                    RGBA.append(buffert)
                RGBA_np = np.asanyarray(RGBA, dtype=np.uint8)
                image_RGBA = Image.fromarray(RGBA_np) 
                image_RGBA.save(toPath + '\\' + 'lofi' + str(song) + '_' + str(posCounter) + '.png')
                posCounter += 1
                pass
            song += 1
            print("File converted compleate! Number of files created: " + str(posCounter))

def math_nice(s): #fix so the image data
    if s < 128:
        return abs(s - 128)
    else:
        return abs((s - 128) - 255)

if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print("Need syntax: python create_dataset.py input_folder output_folder")
        exit()
    toDataSet("./" + sys.argv[1],"./" + sys.argv[2],128,128)