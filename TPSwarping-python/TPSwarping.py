import numpy as np
import cv2

def WarpImage_TPS(source,target,img):
	tps = cv2.createThinPlateSplineShapeTransformer()

	source=source.reshape(-1,len(source),2)
	target=target.reshape(-1,len(target),2)

	matches=list()
	for i in range(0,len(source[0])):

		matches.append(cv2.DMatch(i,i,0))

	tps.estimateTransformation(target, source, matches)  # note it is target --> source

	new_img = tps.warpImage(img)

	# get the warp kps in for source and target
	tps.estimateTransformation(source, target, matches)  # note it is source --> target
	# there is a bug here, applyTransformation must receive np.float32 data type
	f32_pts = np.zeros(source.shape, dtype=np.float32)
	f32_pts[:] = source[:]
	transform_cost, new_pts1 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2
	f32_pts = np.zeros(target.shape, dtype=np.float32)
	f32_pts[:] = target[:]
	transform_cost, new_pts2 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2

	return new_img, new_pts1, new_pts2

def thin_plate_transform(x,y,offw,offh,imshape,shift_l=-0.05,shift_r=0.05,num_points=5,offsetMatrix=False):
	rand_p=np.random.choice(x.size,num_points,replace=False)
	movingPoints=np.zeros((1,num_points,2),dtype='float32')
	fixedPoints=np.zeros((1,num_points,2),dtype='float32')

	movingPoints[:,:,0]=x[rand_p]
	movingPoints[:,:,1]=y[rand_p]
	fixedPoints[:,:,0]=movingPoints[:,:,0]+offw*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)
	fixedPoints[:,:,1]=movingPoints[:,:,1]+offh*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)

	tps=cv2.createThinPlateSplineShapeTransformer()
	good_matches=[cv2.DMatch(i,i,0) for i in range(num_points)]
	tps.estimateTransformation(movingPoints,fixedPoints,good_matches)

	imh,imw=imshape
	x,y=np.meshgrid(np.arange(imw),np.arange(imh))
	x,y=x.astype('float32'),y.astype('float32')
	# there is a bug here, applyTransformation must receive np.float32 data type
	newxy=tps.applyTransformation(np.dstack((x.ravel(),y.ravel())))[1]
	newxy=newxy.reshape([imh,imw,2])

	if offsetMatrix:
		return newxy,newxy-np.dstack((x,y))
	else:
		return newxy

# the correspondences need at least four points
Zp = np.array([[217, 39], [204, 95], [174, 223], [648, 402]]) # (x, y) in each row
Zs = np.array([[283, 54], [166, 101], [198, 250], [666, 372]])
im = cv2.imread('D://1.jpg')
r = 6

# draw parallel grids
for y in range(0, im.shape[0], 10):
		im[y, :, :] = 255
for x in range(0, im.shape[1], 10):
		im[:, x, :] = 255

new_im, new_pts1, new_pts2 = WarpImage_TPS(Zp, Zs, im)
new_pts1, new_pts2 = new_pts1.squeeze(), new_pts2.squeeze()
print(new_pts1, new_pts2)

# new_xy = thin_plate_transform(x=Zp[:, 0], y=Zp[:, 1], offw=3, offh=2, imshape=im.shape[0:2], num_points=4)

for p in Zp:
	cv2.circle(im, (p[0], p[1]), r, [0, 0, 255])
for p in Zs:
	cv2.circle(im, (p[0], p[1]), r, [255, 0, 0])
cv2.imshow('w', im)
cv2.waitKey(500)


for p in Zs:
	cv2.circle(new_im, (p[0], p[1]), r, [255, 0, 0])
for p in new_pts1:
	cv2.circle(new_im, (int(p[0]), int(p[1])), 3, [0, 0, 255])
cv2.imshow('w2', new_im)
cv2.waitKey(0)

