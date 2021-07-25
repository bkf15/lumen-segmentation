#Python implementation of color deconvolution 2 fiji script
# adopted from matlab code: https://beardatashare.bham.ac.uk/getlink/fiBnhf64xssimUwYTSzKJ8Fy/colour_deconvolution2_matlab.zip
# adapted by Brian Falkenstein 11/2020

import numpy as np
from math import sqrt, log, floor

def color_deconvolution(img):
	"""
		Deconvolves a full color stain image into hematoxolin and eosin channels 
		Unique from other python color deconvolutions, like skimages rgh2hed, in that the output
		is a 3 channel color representation of the stain. Currently only supports H&E deconvolution


        Parameters
        ----------
        img: 3 channel ndarray with dtype uint8 

        TODO: more documentation
	"""

	#Note: I am simply copying the naming conventions used in the matlab script
	
	img = img.copy()

	#STAIN VECTORS FOR H&E DECONVOLUTION (can add support for more later)
	MODx = [0.644211, 0.092789, 0]
	MODy = [0.716556, 0.954111, 0]
	MODz = [0.266844, 0.283111, 0]

	#Normalize columns to length 1 in 3D space
	leng = [0, 0, 0]
	cosx = [0, 0, 0]
	cosy = [0, 0, 0]
	cosz = [0, 0, 0]
	for i in range(3):
		leng[i] = sqrt(MODx[i]*MODx[i] + MODy[i]*MODy[i] + MODz[i]*MODz[i])
		if not (leng[i] == 0):
			cosx[i] = MODx[i]/leng[i]
			cosy[i] = MODy[i]/leng[i]
			cosz[i] = MODz[i]/leng[i]

	#translation matrix
	if cosx[1] == 0:
		if cosy[1] == 0:
			if cosz[1] == 0:  #2nd color is unspecified
				cosx[1] = cosz[0]
				cosy[1] = cosx[0]
				cosz[1] = cosy[0]

	if cosx[2] == 0:
		if cosy[2] == 0:
			if cosz[2] == 0: #3rd color is unspecified
				#3rd column will be cross product of first 2
				#fiji implementation allows for computation of 3rd color via Ruifroks method
				# but this is unnecessary for extracting just H&E 
				cosx[2] = cosy[0] * cosz[1] - cosz[0] * cosy[1];
				cosy[2] = cosz[0] * cosx[1] - cosx[0] * cosz[1];
				cosz[2] = cosx[0] * cosy[1] - cosy[0] * cosx[1];

	#renormalize 3rd column
	leng = sqrt(cosx[2]*cosx[2] + cosy[2]*cosy[2] + cosz[2]*cosz[2])
	if leng != 0 and leng != 1:
		cosx[2] = cosx[2]/leng
		cosy[2] = cosy[2]/leng
		cosz[2] = cosz[2]/leng

	COS3x3Mat = np.matrix([
				[cosx[0], cosy[0], cosz[0]], 
				[cosx[1], cosy[1], cosz[1]],
				[cosx[2], cosy[2], cosz[2]]
				])

	#Note: I am skipping lines 390-459 of the matlab code, since
	# the determinant of the COS3x3Mat matrix is > 0 (~0.5). I think that
	# bit of code is trying to make the matrix invertible, but it already is
	# for H&E stain matrix 
	#print(np.linalg.det(COS3x3Mat))

	#Invert the matrix
	# Note that this is done manually in the matlab code.
	Q3x3Mat = np.linalg.inv(COS3x3Mat)
	Q3x3MatInverted = COS3x3Mat    #Just following the matlab code...

	#Compute transmittance 
	rowR = img.shape[0]
	colR = img.shape[1]

	#These are the 1 channel transmittances of each dye 
	Dye1_transmittance = np.zeros([rowR, colR])
	Dye2_transmittance = np.zeros([rowR, colR])
	Dye3_transmittance = np.zeros([rowR, colR])

	for r in range(rowR):
		for c in range(colR):
			RGB1 = img[r, c]
			RGB1[RGB1==0] = 1 #Avoid log0
			ACC = -np.log(RGB1 / 255)
			transmittances = 255 * np.exp(-ACC*Q3x3Mat)
			transmittances = transmittances[0,:]
			transmittances[transmittances>255] = 255

			Dye1_transmittance[r,c] = transmittances[0,0]
			Dye2_transmittance[r,c] = transmittances[0,1]
			Dye3_transmittance[r,c] = transmittances[0,2]

	#Construct lookup tables to convert 1 channel dye images to 
	# 	3 channel RGB representations 
	rLUT = np.zeros([256,3])
	gLUT = np.zeros([256,3])
	bLUT = np.zeros([256,3])

	for i in range(3):
		for j in range(256):
			if cosx[i] < 0:
				rLUT[255-j, i] = 255 + (j * cosx[i])
			else:
				rLUT[255-j, i] = 255 - (j * cosx[i])

			if cosy[i] < 0:
				gLUT[255-j, i] = 255 + (j * cosy[i])
			else:
				gLUT[255-j, i] = 255 - (j * cosy[i])

			if cosz[i] < 0:
				bLUT[255-j, i] = 255 + (j * cosz[i])
			else:
				bLUT[255-j, i] = 255 - (j * cosz[i])

	#Apply the lookup table to first dye (Hematoxilin)
	Dye1_color_im = np.zeros(img.shape)
	for r in range(rowR):
		for c in range(colR):
			#print(floor(Dye1_transmittance[r,c]))
			Dye1_color_im[r,c,0] = rLUT[floor(Dye1_transmittance[r,c]),0]
			Dye1_color_im[r,c,1] = gLUT[floor(Dye1_transmittance[r,c]),0]
			Dye1_color_im[r,c,2] = bLUT[floor(Dye1_transmittance[r,c]),0]

	Dye1_color_im = Dye1_color_im.astype(np.uint8)

	return Dye1_transmittance, Dye1_color_im

