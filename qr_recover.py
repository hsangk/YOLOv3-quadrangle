import cv2
import numpy as np
from PIL import Image
import os

def inclination(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1]


def ccw(p1, p2, p3):
    v, u = inclination(p1, p2), inclination(p2, p3)
    if v[0] * u[1] > v[1] * u[0]:
        return True
    return False


def convex_hull(positions):
    convex = list()
    for p3 in positions:
        while len(convex) >= 2:
            p1, p2 = convex[-2], convex[-1]
            if ccw(p1, p2, p3):
                break
            convex.pop()
        convex.append(p3)

    return convex


def keypoint_detection(mask_image):
    gray = np.array(mask_image)
    # pil_image = Image.fromarray(image)

    # img = cv2.imread('./1.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(type(gray))
    # 시-토마스의 코너 검출 메서드

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=10)
    # 실수 좌표를 정수 좌표로 변환
    final_corners = corners
    # corners = np.int32(corners)

    # 좌표에 동그라미 표시
    # for corner in corners:
    #     x, y = corner[0]
    #     cv2.circle(gray, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)
    # # print(type(mask_image))
    # cv2.imshow('Corners', gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    coordinate_list = []
    for final_corner in final_corners:
        x, y = final_corner[0]
        coordinate_list.append(x)
        coordinate_list.append(y)

    return coordinate_list


def perspective(image, label, image_name):
    print(label)
    print(type(image))
    pts = np.zeros((4, 2), dtype=np.float32)

    # print(len(label))
    # label= [x1,y1,x2,y2,x3,y3,x4,y4]
    if len(label)==8:
        pts[0] = [label[0], label[1]]
        pts[1] = [label[2], label[3]]
        pts[2] = [label[4], label[5]]
        pts[3] = [label[6], label[7]]
        pts = np.int32(pts)

        positions = sorted(pts, key=lambda pos: (pos[0], pos[1]))
        convex1 = convex_hull(positions)

        positions.reverse()
        convex2 = convex_hull(positions)

        if len(convex1)==2:
            pts1 = np.float32([convex1[0], convex1[1], convex2[-3], convex2[-2]])
        else:
            pts1 = np.float32([convex1[0], convex1[1], convex1[2], convex2[-2]])

        width=100
        height=100

        # 변환 후 4개 좌표
        pts2 = np.float32([[0, 0], [width - 1, 0],
                           [width - 1, height - 1], [0, height - 1]])

        # 변환 행렬 계산
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        # 원근 변환 적용
        img=np.array(image)
        result = cv2.warpPerspective(img, mtrx, (width, height))

     #   cv2.imshow('scanned', result)
     #   cv2.waitKey()
     #  cv2.destroyAllWindows()
        #cv2.imwrite('D:/shadow_dataset/qr_sha/' + str(n + 1) + '_' + str(qr_num + 1) + '.png', result)
        # qr_image.save('D:/shadow_dataset/qrmake_image/qr_gt/' + str(n + 1) + '_' + str(j + 1) + '.png')
        cv2.imwrite('./output/qrcode_seg/'+image_name, result)


path = './output/samples_seg/'
image_path = './data/samples_detect/'
pathlist = os.listdir(path)
pathlist2 = os.listdir(image_path)
label_list = [file for file in pathlist if file.endswith('.txt')]
image_list = [file for file in pathlist2 if file.endswith('.png')]

for pair in zip(label_list, image_list):
    label_path = path + pair[0]
    image = image_path + pair[1]
    image = Image.open(image).convert("RGBA")
    with open(label_path,'r') as f:
        label = f.read()
    label = label.split()
    
    perspective(image, label, pair[1])


