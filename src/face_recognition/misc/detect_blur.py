# import the necessary packages
from imutils import paths
import argparse
import cv2
import numpy

from scipy import misc

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def var_of_sobel(image):
        return cv2.Sobel(image, cv2.CV_64F, 1, 0).var()

def double_sobel(image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

        #dnorm = numpy.sqrt(gx**2 + gy**2)
        dm = cv2.magnitude(gx, gy)
        #return numpy.average(dnorm)
        return numpy.sum(dm)

def maxdouble_sobel(image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

        #dnorm = numpy.sqrt(gx**2 + gy**2)
        dm = cv2.magnitude(gx, gy)
        return numpy.average(dm)


def variance_of_laplacian2(image):
        # compute the Laplacian of the image and then return the focus
        eimage = cv2.equalizeHist(image);
        return cv2.Laplacian(eimage, cv2.CV_64F).var()


def max_of_laplacian(gray_image):
        return numpy.max(cv2.convertScaleAbs(cv2.Laplacian(gray_image,3)))


def fft_evaluate(img_gry):
        rows, cols = img_gry.shape
        crow, ccol = rows/2, cols/2
        f = numpy.fft.fft2(img_gry)
        fshift = numpy.fft.fftshift(f)
        fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
        f_ishift = numpy.fft.ifftshift(fshift)
        img_fft = numpy.fft.ifft2(f_ishift)
        img_fft = 20*numpy.log(numpy.abs(img_fft))
        result = numpy.mean(img_fft)
        return result


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")

args = vars(ap.parse_args())

# loop over the input images
for i, imagePath in enumerate(sorted(paths.list_images(args["images"]))):
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eqhist = cv2.equalizeHist(gray)
        metrics = {}

        width, height, _ = image.shape

        cropped = image[50:height-50,50:width-50,:]
        grayc = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)


	lapvar = variance_of_laplacian(gray)
        metrics['lapvar']=lapvar

        lapvar2 = variance_of_laplacian2(gray)
        metrics['lapvar2']=lapvar2

	sobvar = var_of_sobel(gray)
        metrics['sobvar']=sobvar

        doubsob = double_sobel(gray)
        metrics['doubsob']=doubsob

        doubsob2 = double_sobel(image)
        metrics['doubsob2']=doubsob2

        eqdoubsob = double_sobel(eqhist)
        metrics['hist eq doubsob']=eqdoubsob

        maxdoubsob = maxdouble_sobel(image)
        metrics['maxdoubsob']=doubsob2

        lapmax = max_of_laplacian(gray)
        metrics['lapmax']=lapmax

        fft1 = fft_evaluate(gray)
        metrics['fft1']=fft1

        cv2.putText(image, '{t:.3f}'.format(t=doubsob2/1000), (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, 8, False)
        misc.imsave('blur_class/{}.png'.format(i), image)

        metricsc = {}

        lapvar = variance_of_laplacian(grayc)
        metricsc['lapvar']=lapvar

        lapvar2 = variance_of_laplacian2(grayc)
        metricsc['lapvar2']=lapvar2

	sobvar = var_of_sobel(grayc)
        metricsc['sobvar']=sobvar

        doubsob = double_sobel(grayc)
        metricsc['doubsob']=doubsob

        doubsob2 = double_sobel(cropped)
        metricsc['doubsob2']=doubsob2

        eqdoubsob = double_sobel(eqhist)
        metricsc['hist eq doubsob']=eqdoubsob

        maxdoubsob = maxdouble_sobel(cropped)
        metricsc['maxdoubsob']=doubsob2

        lapmax = max_of_laplacian(grayc)
        metricsc['lapmax']=lapmax

        fft1 = fft_evaluate(grayc)
        metricsc['fft1']=fft1

        cv2.putText(cropped, '{t:.3f}'.format(t=doubsob2/1000), (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, 8, False)
        misc.imsave('cropped/{}_{}.png'.format(i, doubsob2/1000), cropped)

	# show the image
	#cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	#cv2.imshow("Image", image)
	#key = cv2.waitKey(0)
        print imagePath, '->', 'Not Blurry' if doubsob > 1000000.0 else 'Blurry'
        print metrics
        print metricsc, '\n'