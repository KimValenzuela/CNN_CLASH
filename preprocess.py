import numpy as np
from astropy.io import fits
import pandas as pd

class Preprocess(object):

    def __init__(self, data, path):
        self.IDs = data['ID']
        self.y = data.drop('ID', axis = 1)
        self.path = path

    def get_data(self):
        X = self.ids_to_image()
        X = self.normalize(X)
        return X, self.y.to_numpy()

    def ids_to_image(self):
        images = []
        for _id in self.IDs:
            images.append(self.open_image(_id))
        return np.stack(images, axis = 0)

    def normalize(self, data):
        mn = np.min(data)
        mx = np.max(data)
        return ((data - mn)/(mx - mn))

    def open_image(self, ID):
        image = fits.open(f"{self.path}{ID}.fits")
        image = image[0].data
        image = np.resize(image.astype('float32'), (80, 80)) 
        return image.reshape(80, 80, 1)




if __name__ == '__main__':

	dataset = 'CANDELS'
	data = pd.read_csv(f'{dataset}/{dataset}_labels.csv', index_col = None)[:10]
	prep = Preprocess(data, f'{dataset}/stamps/')
	X, y = prep.get_data()