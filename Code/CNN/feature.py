import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
from PIL import Image


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    #cropped_image.show()

def draw_spectogram(path_to_audio, save_path):
    data, sampling_rate = librosa.load(path_to_audio) #os.path.join(root, directory, file_))
    specs = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(specs,ref=np.max), y_axis='mel', fmax=9000,x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()



