import cv2
import glob

path = "/Users/sarthakbhagat/Desktop/ML_Data/val/"
files = glob.glob(path + '/*.jpg')

for i in range(len(files)):
	img = cv2.imread(files[i])
	right = img[:, :600, :]
	left = img[:, 600:, :]

	cv2.imwrite('/Users/sarthakbhagat/Desktop/ML_data_processed/val/map/' + str(i) + '.jpg', left)
	cv2.imwrite('/Users/sarthakbhagat/Desktop/ML_data_processed/val/img/' + str(i) + '.jpg', right)
