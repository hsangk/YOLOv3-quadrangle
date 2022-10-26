import numpy as np
import cv2
import os
import glob
from PIL import Image

path = './output/samples_seg/'
image_path = './data/samples_detect/'
pathlist = os.listdir(path)
pathlist2 = os.listdir(image_path)
label_list = [file for file in pathlist if file.endswith('.txt')]
image_list = [file for file in pathlist2 if file.endswith('.png')]

'''
new_path = glob.glob('./data/samples_detect/*')
for i, f in enumerate(new_path):
    ftitle, fext = os.path.splitext(f)
    title = os.path.basename(new_path[i])
    title = title.replace('.png','')
    os.rename(f, './data/samples_detect/'+'{0:05d}'.format(int(title)) + fext)
'''

# with open('./output/noqr(233).txt', 'r') as f:
# idx = f.read()
# idx = idx.split()
'''
for k in range(len(idx)):
    kkk_img = idx[k].replace(".txt",".png")
    kkk = idx[k]
    search = kkk
    search_img = kkk_img
    for word in label_list:
        if search == word:
            label_list.remove(word)
    for word2 in image_list:
        if search_img == word2:
            image_list.remove(word2)
'''

for pair in zip(label_list, image_list):
    label_path = path + pair[0]
    image = image_path + pair[1]
    with open(label_path, 'r') as f:
        label = f.read()
    label = label.split()

    label_x = []
    label_y = []
    for i in range(len(label)):
        label[i] = int(label[i])
        if i%2==0:
            label_x.append(label[i])
        else:
            label_y.append(label[i])

    '''
    new_label = [0]*8
    for j in range(len(new_label)):
        if j == 0 or j == 2:
            new_label[j] = min(label_x)
        elif j == 1 or j == 5:
            new_label[j] = min(label_y)
        elif j == 4 or j == 6:
            new_label[j] = max(label_x)
        else:
            new_label[j] = max(label_y)
    '''

    image2 = cv2.imread(image)
    crop_img = image2[min(label_y):max(label_y), min(label_x):max(label_x)]

    cv2.imwrite('./output/qrcode_src_seg/' + pair[1], crop_img)

    # cv2.imshow('result', crop_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


'''
# failed detection label 
path = './output/qrcode/'
pathlist = os.listdir(path)
label_list = [file for file in pathlist if file.endswith('.png')]
search = '.png'

for i, word in enumerate(label_list):
    if search in word:
        # print('>> modify: ' + word)
        label_list[i] = word.strip(search)
label_list = list(map(int, label_list))

check = range(1,10001)
num = []
for j in check:
    if j not in label_list:
        num.append(j)

#file = open('./output/noqr.txt','w')
with open('./output/noqr.txt','w') as f:
    for i in range(len(num)):
        f.write(str(num[i])+'.txt'+'\n')

print(num)
print(len(num))
'''