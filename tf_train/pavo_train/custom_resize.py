# Author: Emilio Chavez M.
# date: Jan 25, 2019

# This simple script helps with custom resizing of a folder with images.
# The main intention is to use them for training a custom object detection with  
# Tensorflow Object detection API.

import sys, getopt
import os
import cv2


path_to_source_folder = None
path_to_out_folder = None


# Print helper mesage
def usage():
	sys.exit('Error specifying input or output folder \n' \
		+ 'custom_resize.py -i <inputfolder> -o <outputfolder>')

def retrieve_paths(argv):
	global path_to_out_folder
	global path_to_source_folder

	# Verify params
	try:
		if (not (len(argv)>3)):
			usage()

		opts, args = getopt.getopt(argv,"hi:o:",["in_folder=","out_folder="])
	except getopt.GetoptError:
		usage()

	# Get folder names
	for opt, arg in opts:
		if opt == '-h':
			usage()
		elif opt in ("-i", "--in_folder"):
			path_to_source_folder = arg
		elif opt in ("-o", "--out_folder"):
			path_to_out_folder = arg	


def main(argv):

	# Get in and out folders for the images
	retrieve_paths(argv)

	print('Input path:\t', path_to_source_folder)
	print('Output path:\t', path_to_out_folder)

	list_of_images_paths = os.listdir(path_to_source_folder)
	print ("list of imgs: ", list_of_images_paths)
	for img_path in list_of_images_paths:
		# Load img
		print ("Loading : ", path_to_source_folder +img_path)
		cur_img = cv2.imread(path_to_source_folder + img_path, cv2.IMREAD_COLOR)
		print ('Current img size : ', cur_img.shape[:2])
		resize_img = []

		# load height width
		im_row, im_col = cur_img.shape[:2] # rows, colums
		ratio = im_row/im_col
		
		print ('image dim: ', im_row*im_col, 'image ratio: ', ratio)
		if (im_row*im_col < 800*600):
			print ('image OK.')
			cv2.imwrite(path_to_out_folder + img_path, cur_img)
		else:
			# Portrait
			if (ratio > 1):
				fac_x = float(800/im_col)
				resize_img = cv2.resize(cur_img,None,fx=fac_x, fy=fac_x)
			# Landscape
			else:
				fac_y = float(800/im_row)
				resize_img = cv2.resize(cur_img,None,fx=fac_y, fy=fac_y)
		

			print ('Saving img w/ new shape ', resize_img.shape[:2])
			cv2.imwrite(path_to_out_folder + img_path, resize_img)
			print('saved\n')

if __name__ == '__main__':
	main(sys.argv[1:])