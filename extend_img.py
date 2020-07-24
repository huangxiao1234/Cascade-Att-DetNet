#先围绕bounding box裁剪，然后resize到原图统一大小相当于方法，label则先将坐标框入图中，放大后再获取坐标

from PIL import Image,ImageDraw
import pandas as pd
import numpy as np 
import os
import cv2
# names = os.listdir('DFUC2020_Training_Release/images/')
# for name in names:
# 	img = Image.open('DFUC2020_Training_Release/images/'+name)
# 	gt = pd.read_csv('DFUC2020_Training_Release/groundtruth.csv')
# 	index = gt.loc[gt['name']==name].values[0]
# 	d_x = (index[3]-index[1])*2
# 	d_y = (index[4]-index[2])*2
# 	size = np.shape(np.array(img))
# 	# print(size)
# 	try:
# 		img = Image.fromarray(np.array(img)[index[2]-d_y:index[4]+d_y,index[1]-d_x:index[3]+d_x])
# 		img = img.resize((size[1],size[0]))
# 		print(np.shape(img))
# 		img.save('DFUC2020_Training_Release/extend_images1/'+name)
# 	except Exception as e:
# 		pass
src = 'DFUC2020_Training_Release2/'
names = os.listdir(src+'images/')
# names=['100002.jpg']
extend_label =[]
for name in names:
	img = Image.open(src+'images/'+name)
	gt = pd.read_csv(src+'groundtruth.csv')
	index = gt.loc[gt['name']==name].values[0]
	print(index)
	d_x = (index[3]-index[1])*2
	d_y = (index[4]-index[2])*2
	size = np.shape(np.array(img))
	# a=ImageDraw.ImageDraw(img)
	# a.rectangle([(index[1],index[2]),(index[3],index[4])],outline ='black',width =1)
	# img.save('DFUC2020_Training_Release/extend_images1/'+name)
	# # print(size)
	try:
		label = np.zeros(size[:2])
		for x in range(index[1],index[3]):
			label[index[4],x]=1
		for x in range(index[1],index[3]):
			label[index[2],x]=1	
		for y in range(index[2],index[4]):
			label[y,index[1]]=1
		for y in range(index[2],index[4]):
			label[y,index[3]]=1	
		label = Image.fromarray(label[index[2]-d_y:index[4]+d_y,index[1]-d_x:index[3]+d_x])

		label = label.resize((size[1],size[0]))
		index2 = np.argwhere(np.array(label)==1)
		xmin = np.min(index2[:,1])
		xmax = np.max(index2[:,1])
		ymin = np.min(index2[:,0])
		ymax = np.max(index2[:,0])		
		extend_label.append([name,xmin,xmax,ymin,ymax])

		img = Image.fromarray(np.array(img)[index[2]-d_y:index[4]+d_y,index[1]-d_x:index[3]+d_x])
		img = img.resize((size[1],size[0]))
		# print(np.shape(img))
		# a=ImageDraw.ImageDraw(img)
		# a.rectangle([(xmin,ymin),(xmax,ymax)],outline ='black',width =1)
		
		# cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 4)
		img.save(src+'extend_images/'+name)
		# img.save('DFUC2020_Training_Release/extend_images1/'+name)
	except Exception as e:
		pass
# print(extend_label)
df = pd.DataFrame(extend_label)
df.columns=['name','xmin','ymin','xmax','ymax']
df.to_csv(src+'groundtruth_extend.csv',index=False)