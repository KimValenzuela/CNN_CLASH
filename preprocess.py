import numpy as np
from astropy.io import fits
import pandas as pd
from PIL import Image as img

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
        image = img.fromarray(image)
        image = image.resize((80, 80))
        image = np.asarray(image).reshape(1, 80, 80)
        # image = np.moveaxis(image, 0, -1)
        # print(image)
        # hdu_image = fits.PrimaryHDU(data = image)
        # hdu_image.writeto(f'images_preprocess/{ID}_preprocess.fits', overwrite = True)
        return image


if __name__ == '__main__':

	dataset = 'CANDELS'
	data = pd.read_csv(f'{dataset}/{dataset}_labels.csv', index_col = None)[:10]
	prep = Preprocess(data, f'{dataset}/stamps/')
	X, y = prep.get_data()