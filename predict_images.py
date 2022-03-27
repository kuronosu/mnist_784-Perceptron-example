import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def transform_pixel(pixel):
    return 255 - sum(pixel) / len(pixel)

def pixel_to_terminal(pixel):
    c = 23.9 * (pixel/255)
    # return 255 - int(c) 
    return int(c) + 232

def colored(color, content=' '):
    return f'\x1b[48;5;{pixel_to_terminal(color)}m{content}\x1b[0m'

class Sample:
    def __init__(self, value, img_path):
        self.value = value
        self.img_path = img_path
        self.prediction = None
        self.image = None
        self.data_arr = None
        self.load()
    
    def load(self):
        img = Image.open(self.img_path)
        img_mtx = np.array(img)
        img_arr = []
        for row in img_mtx:
            for pixel in row:
                img_arr.append(transform_pixel(pixel))
        self.image = img
        self.data_arr = np.asarray(img_arr)
    
    def save_data_arr(self, filename):
        im = Image.fromarray(np.reshape(self.image, (28,28)))
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(filename)
    
    def predict_clf(self, clf):
        self.prediction = clf.predict([self.data_arr])[0]

def print_num(num):
    for r in np.reshape(num, (28,28)):
        print(*[colored(v, 'ㅤ') for v in r], sep='')

def load_data():
    samples = []
    for number in range(10):
        sample = Sample(number, f'img/{number}.bmp')
        samples.append(sample)
    return samples

def make_images(samples):
    c = 0
    n = len(samples)
    figure, axarr = plt.subplots(2, n, figsize=(20, 4))
    for sample in samples:
        axarr[0, c].axis('off')
        axarr[0, c].title.set_text(f'Orig: {sample.value}')
        axarr[0, c].imshow(sample.image, cmap=plt.cm.gray)
        axarr[1, c].axis('off')
        axarr[1, c].title.set_text(f'Pred: {sample.prediction}')
        axarr[1, c].imshow(np.reshape(sample.data_arr, (28,28)), cmap=plt.cm.gray)
        c+=1
    figure.tight_layout()
    figure.canvas.draw()
    return figure

def assemble_predictions(samples, clf):
    for sample in samples:
        sample.predict_clf(clf)


if __name__ == '__main__':
    clf = joblib.load('clf.sav')
    samples = load_data()
    assemble_predictions(samples, clf=clf)
    fig = make_images(samples)
    plt.show()
    # Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).save('predictions.png')
    # print_num(samples[1].data_arr)
