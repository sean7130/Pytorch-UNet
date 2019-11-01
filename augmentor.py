import Augmentor
import cv2
import os

def augment_pair(full_img_path, full_mask_path, dest_folder_path, b=30, FISH_EYE=True, PERSPECTIVE=True):
	""" augmenting tool using the Augmentor library. Supports augumeting two images at the same time.
	"""
	img1 = cv2.imread(full_img_path)
	img2 = cv2.imread(full_mask_path)
	image_name = os.path.basename(full_img_path)
	mask_name = os.path.basename(full_mask_path)

	# Initialize pipeline
	p = Augmentor.DataPipeline([[img1,img2]])

	# Apply augmentations
	p.rotate(1, max_left_rotation=5, max_right_rotation=5)
	p.shear(1, max_shear_left = 5, max_shear_right = 5)
	p.zoom_random(1, percentage_area=0.9)

	if FISH_EYE:
		p.random_distortion(probability=1, grid_width=5, grid_height=5, magnitude=5)
		p.gaussian_distortion(probability=1, grid_width=5, grid_height=5, magnitude=5, corner='bell', method='in')

	if PERSPECTIVE:
		p.shear(1, 10, 10)
		p.skew_tilt(1, magnitude=0.2)

	images_aug = p.sample(b)

	if dest_folder_path not in os.listdir():
		os.mkdir(dest_folder_path)
		os.mkdir(os.path.join(dest_folder_path, 'imgs'))
		os.mkdir(os.path.join(dest_folder_path, 'masks'))

	for i in range(len(images_aug)):
		mask_gray = cv2.cvtColor(images_aug[i][1], cv2.COLOR_BGR2GRAY)
		# Assumes .jpg extension name for masks
		cv2.imwrite(os.path.join(dest_folder_path, 'imgs', image_name[:-3]+str(i)+".jpg"), images_aug[i][0])
		cv2.imwrite(os.path.join(dest_folder_path, 'masks', mask_name[:-3]+str(i)+".jpg",), mask_gray)

# if __name__ == "__main__":
# 	img_path = "cat_data/Train/input/cat.0.jpg"
# 	mask_path = "cat_data/Train/mask/mask_cat.0.jpg"
# 	dest_folder_path = 'augment_batch_test'
# 
# 	augment_pair(img_path, mask_path, dest_folder_path)
# 
# 	pass
