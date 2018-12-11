from PIL import Image
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import spectra
import requests
import numpy as np


class DominantColors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):
        # read image
        img = cv2.imread(self.IMAGE)

        # convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # save image after operations
        self.IMAGE = img

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        # returning after converting to integer from float
        return self.COLORS.astype(int)

    def plotHistogram(self):
        # labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS + 1)

        # create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # appending frequencies to cluster centers
        colors = self.COLORS

        # descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            # using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end

        # display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()


def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


# colors = {}
# im = Image.open('captcha.jpeg')
# rgb_im = im.convert('RGB')
# width, height = im.size
# for i in range(0, width):
#     for j in range(0, height):
#         r, g, b = rgb_im.getpixel((i, j))
#         hex_color = rgb2hex(r, g, b)
#         if hex_color in colors:
#             colors[hex_color] += 1
#         else:
#             colors[hex_color] = 1
#
# print(dict(sorted(colors.items(), key=lambda x: x[1])))


def get_main_colors(image_path, n_clusters=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters)
    clt.fit(image)
    return clt.cluster_centers_


def remove_colors(image_path, color, radio=0.30):
    im = Image.open(image_path)
    width, height = im.size
    new_im = Image.new("RGB", (width, height))
    color_range = [
        [color[0] * (1 - radio), color[0] * (1 + radio)],
        [color[1] * (1 - radio), color[1] * (1 + radio)],
        [color[2] * (1 - radio), color[2] * (1 + radio)],
    ]
    rr, gr, br = color_range
    print(color_range)
    rgb_im = im.convert('RGB')
    for i in range(0, width):
        for j in range(0, height):
            r, g, b = rgb_im.getpixel((i, j))
            if r > rr[0] and r < rr[1] and g > gr[0] and g < gr[1] and b > br[0] and b < br[1]:
                new_im.putpixel((i, j), (0, 0, 0))
            else:
                new_im.putpixel((i, j), (255, 255, 255))
    return new_im


file_name = 'captcha.jpeg'
r = requests.get('http://btcache.me/captcha')
open(file_name, 'wb').write(r.content)

img = cv2.imread('captcha.jpeg', 0)
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [Image.open(file_name), th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# import cv2.cv as cv
# import tesseract
# gray = cv.LoadImage('captcha.jpeg', cv.CV_LOAD_IMAGE_GRAYSCALE)
# cv.Threshold(gray, gray, 231, 255, cv.CV_THRESH_BINARY)
# api = tesseract.TessBaseAPI()
# api.Init(".","eng",tesseract.OEM_DEFAULT)
# api.SetVariable("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyz")
# api.SetPageSegMode(tesseract.PSM_SINGLE_WORD)
# tesseract.SetCvImage(gray,api)
# print(api.GetUTF8Text())

# img = Image.open(loadpath + file).convert("L")

# plt.imshow(Image.open(file_name))
# plt.show()

# clusters = 2
# dc = DominantColors(file_name, clusters)
# colors = dc.dominantColors()
# print(colors)
# dc.plotHistogram()
# im = remove_colors(file_name, colors[1])
# plt.imshow(im)
# plt.show()
