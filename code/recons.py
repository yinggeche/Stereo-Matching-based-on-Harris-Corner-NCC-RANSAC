import numpy as np
import cv2
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import random
from scipy import linalg
import sys
import math
sigma=1.4
harris_k=0.06
harris_threshold=0.03
harris_mindist=8
NCC_threshold=0.99
NCC_windodim=8
RANSAC_times=1000
F_inliner_threshold=1
e_line_number=100
dense_map_threshold=0
dense_range=3
SIFT_threshold=0.75
def readimage():
  #leftimage = cv2.imread("cast-left.jpg")
  #rightimage = cv2.imread("cast-right.jpg")
  leftimage = cv2.imread("Cones_im2.jpg")
  rightimage = cv2.imread("Cones_im6.jpg")
  return leftimage,rightimage

def converttogray(l_m,r_m):
    gray_left = cv2.cvtColor(l_m, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(r_m, cv2.COLOR_BGR2GRAY)
    return gray_left,gray_right
def SIFTcorner(img1,img2,min_dist,SIFT_threshold):
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    filtered_coords1 = []
    filtered_coords2 = []
    good = []
    for k1 in kp1:
        filtered_coords1.append(np.array([k1.pt[1],k1.pt[0]],dtype='int64'))
    for k2 in kp2:
        filtered_coords2.append(np.array([k2.pt[1],k2.pt[0]],dtype='int64'))
    for m, n in matches:
        if m.distance < SIFT_threshold * n.distance:
            good.append([m.queryIdx,m.trainIdx])
    return filtered_coords1,filtered_coords2,good
def harriscorner(im,threshold,min_dist):
    #get derivate
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    #find wxx,wxy,wyy
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)
    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    harrisim=Wdet/Wtr

    conner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > conner_threshold) * 1
    coords = np.array(harrisim_t.nonzero()).T
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    index = np.argsort(candidate_values)
    allowed_locations = np.zeros(harrisim.shape)
    # non_maximum suppression
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0
    #print(filtered_coords)
    return filtered_coords
def getNCC(leftwindow,rightwindow):
    N_L=sum(sum(leftwindow**2))**(1/2)
    N_R = sum(sum(rightwindow ** 2)) ** (1 / 2)
    ncc=sum(sum((leftwindow/N_L)*(rightwindow/N_R)))
    return ncc
def matchpoints(gray_left,gray_right,leftcorner,rightcorner,windowsize,threshold):
    m=len(leftcorner)
    n=len(rightcorner)
    gray_left=np.array(gray_left,dtype='uint32')
    gray_right=np.array(gray_right,dtype='uint32')
    NCC_value=np.zeros((m,n))
    for i in range (m):
        leftwindow = gray_left[(leftcorner[i][0] - windowsize):(leftcorner[i][0] + windowsize + 1),
                     (leftcorner[i][1] - windowsize):(leftcorner[i][1] + windowsize + 1)]
        for j in range(n):
            rightwindow=gray_right[(rightcorner[j][0]-windowsize):(rightcorner[j][0]+windowsize+1),(rightcorner[j][1]-windowsize):(rightcorner[j][1]+windowsize+1)]
            NCC_value[i,j]=getNCC(leftwindow,rightwindow)
    ncc=NCC_value.tolist()
    matches=[]
    for i in range (m*n):
        m_max = []
        for j in range (m):
            m_max.extend([max(ncc[j])])
        temp=max(m_max)
        if temp<threshold:
            break
        for a in range (m):
            for b in range(n):
                if ncc[a][b]==temp:
                    matches += [[a,b]]
                    ncc[a][b] = 0
                    a=m+1
                    b=n+1
                    break
    return matches

def drawmatchpointimage(image1,image2,kpl,kpr,matches,linenumber):
    #res = cv2.drawMatches(img1_gray,kpl, img2_gray, kpr, matches[:40], None, flags=2)
    width = image1.shape[1] + image2.shape[1]
    height = max(image1.shape[0], image2.shape[0])
    image = np.zeros([height, width,3], dtype='uint8')
    for i in range(0, width):
        for j in range(0, height):
            if i < ((image.shape[1] / 2) - 1):
                image[j][i] = image1[j][i]
                x=i
            else:
                image[j][i] = image2[j][i-x-2]
    #cv2.imwrite(path + name + '.jpg', image)
    plt.ion()
    plt.imshow(image,cmap='gray')
    #plt.plot([p[1] for p in kpl], [p[0] for p in kpl], '+')
    #plt.plot([p[1]+image1.shape[1] for p in kpr], [p[0] for p in kpr], '+')
    for a in range(0, linenumber):
        temp_l=matches[a][0]
        temp_r=matches[a][1]
        plt.plot(kpl[temp_l][1],kpl[temp_l][0],'+')
        plt.plot(kpr[temp_r][1]+image1.shape[1],kpr[temp_r][0],'+')
        plt.plot([kpl[temp_l][1],kpr[temp_r][1]+image1.shape[1]],[kpl[temp_l][0],kpr[temp_r][0]])

    plt.ioff()
    plt.show()
    return 0
def drawinliners(image1,image2,cor_l,cor_r,ncc_match,F,threshold):
    inliners=[]
    draw_p=len(ncc_match)
    if draw_p>=40:
        draw_p=40
    for c in range (draw_p):
      p_l = []
      p_l.extend([cor_l[ncc_match[c][0]]])
      p_l = np.append(p_l, 1)
      p_r = []
      p_r.extend([cor_r[ncc_match[c][1]]])
      p_r = np.append(p_r, 1)
      if np.dot(np.dot(p_l,F),p_r)<threshold:
          inliners+=[ncc_match[c]]
    print(len(inliners))
    drawmatchpointimage(image1,image2,cor_l,cor_r,inliners,len(inliners))
    return
def findFmatrix(x1,x2,n):
    x1=np.array(x1)
    x2=np.array(x2)
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[i, 1] * x2[i, 1], x1[i, 1] * x2[i, 0], x1[i, 1],
                x1[i, 0] * x2[i, 1], x1[i, 0] * x2[i, 0], x1[i, 0],
                x2[i, 1], x2[i, 0], 1]
    # compute linear least square solution
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3, 3)
    # constrain F
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    return F / F[2, 2]
def theransanc(ransactimes,cor_l,cor_r,ncc_match):
    ACC_result=sys.maxsize
    F_result=None
    ransanc_num=50
    for a in range(ransactimes):
        p1=[]
        p2=[]
        for b in range(ransanc_num):
          temp=random.randint(0, len(ncc_match)-1)
          p1.extend([cor_l[ncc_match[temp][0]]])
          p2.extend([cor_r[ncc_match[temp][1]]])
        F=findFmatrix(p1,p2,ransanc_num)
        temp1=0
        for c in range(len(ncc_match)):
            p_l=[]
            p_l.extend([cor_l[ncc_match[c][0]]])
            p_l=np.append(p_l,1)
            p_r = []
            p_r.extend([cor_r[ncc_match[c][1]]])
            p_r = np.append(p_r, 1)
            temp1+=abs(np.dot(np.dot(p_l.T,F),p_r))
        if temp1<ACC_result:
            ACC_result=temp1
            F_result=F
            #print([a,ACC_result])
    return F_result
def calculateepipole(point,F,m,n,linepoints):
    e_line=np.dot(F,point.T)
    t = np.linspace(NCC_windodim, n-NCC_windodim-1, linepoints)
    lt = np.array([(e_line[2] + e_line[0] * tt) / (-e_line[1]) for tt in t])
    ndx = (lt >= 0) & (lt < m)
    t = np.reshape(t, (linepoints, 1))
    return t[ndx],lt[ndx]
def drawepipole(image1,image2,point,F):
    m, n= image2.shape
    width = image1.shape[1] + image2.shape[1]
    height = max(image1.shape[0], image2.shape[0])
    image = np.zeros([height, width, 3], dtype='uint8')
    for i in range(0, width):
        for j in range(0, height):
            if i < ((image.shape[1] / 2) - 1):
                image[j][i] = image1[j][i]
                x = i
            else:
                image[j][i] = image2[j][i - x - 2]
    line_x,line_y=calculateepipole(point,F,m,n,e_line_number)
    plt.ion()
    plt.imshow(image)
    plt.plot(point[1],point[0],'+')
    plt.plot(line_x+n, line_y, linewidth=2)
    plt.ioff()
    plt.show()
def computedensemap(leftimage,rightimage,F,fullimage):
    v=rgbtohsv(fullimage)[:,:,2]
    m, n= leftimage.shape
    horizontal_map=np.zeros((m,n))
    vertical_map=np.zeros((m,n))
    vector_map=np.zeros((m,n,3))
    leftimage = np.array(leftimage, dtype='uint32')
    rightimage = np.array(rightimage, dtype='uint32')
    temp_right=np.zeros((m+2*NCC_windodim,n))
    temp_right[NCC_windodim:-NCC_windodim,:]=rightimage
    #print(temp_right.shape)
    #print(temp_right[0])
    max_h=0
    max_v=0
    max_sat=0
    # calculate epipole line
    for i in range(NCC_windodim,m-NCC_windodim-1):
    #for i in range(200,300):
        for j in range(NCC_windodim,n-NCC_windodim-1):
        #for j in range(100,300):
            line_x,line_y=calculateepipole(np.array([j,i,1]),F,m,n,e_line_number)
    #         print([line_x,line_y])
            leftwindow = leftimage[(i-NCC_windodim):(i+NCC_windodim+1),(j-NCC_windodim):(j+NCC_windodim+1)]
            # search NCC on the line
            temp_ncc=0
            temp_match=[]
            for a in range(len(line_x)):
               pointx=int(line_x[a])
               pointy=int(line_y[a])
               for b in range(-dense_range,dense_range+1):
                   rightwindow = temp_right[(pointy):(pointy+2*NCC_windodim+1),(pointx-NCC_windodim):(pointx+NCC_windodim+1)]
                   temp=getNCC(leftwindow,rightwindow)
                   if temp>temp_ncc and temp>dense_map_threshold:
                          temp_ncc=temp
                          temp_match=[pointy,pointx]

            if temp_match!=[]:
                dif_x=abs(j-temp_match[1])
                dif_y=abs(i-temp_match[0])
                horizontal_map[i][j]=dif_x
                vertical_map[i][j]=dif_y
                hue=0
                if dif_x!=0:
                   hue=math.degrees(math.atan(dif_y/dif_x))
                if hue<0:
                    hue=360-abs(hue)
                sat=(dif_x**2+dif_y**2)**(1/2)
                #inten=125
                #print(vector_map.size)
                vector_map[i][j][0] = hue
                vector_map[i][j][1] = sat
                #vector_map[i][j][2] = inten
                if dif_x>max_h:
                   max_h=dif_x
                if dif_y>max_v:
                    max_v=dif_y
                if sat>max_sat:
                    max_sat=sat
        print(i)
    horizontal_map*=(255/max_h)
    vertical_map*=(255/max_v)
    #print(vertical_map[:0])
    for i in range(NCC_windodim,m-NCC_windodim-1):
        for j in range(NCC_windodim,n-NCC_windodim-1):
            if horizontal_map[i][j] == 0:
                temp=horizontal_map[i-1][j-1]+horizontal_map[i-1][j]+horizontal_map[i-1][j+1]+horizontal_map[i][j-1]+horizontal_map[i][j+1]+horizontal_map[i+1][j-1]+horizontal_map[i+1][j]+horizontal_map[i+1][j+1]
                temp/=8
                if temp>=10:
                    horizontal_map[i][j]=int(255-temp)
            else:
                horizontal_map[i][j]=int(255-horizontal_map[i][j])
            if vertical_map[i][j] == 0:
                temp=vertical_map[i-1][j-1]+vertical_map[i-1][j]+vertical_map[i-1][j+1]+vertical_map[i][j-1]+vertical_map[i][j+1]+vertical_map[i+1][j-1]+vertical_map[i+1][j]+vertical_map[i+1][j+1]
                temp/=8
                if temp>=10:
                    vertical_map[i][j]=int(255-temp)
            else:
                vertical_map[i][j]=int(255-vertical_map[i][j])
    vector_map[:,:,2]=v
    vector_map[:,:,1]*=(255/max_sat)
    plt.ion()
    plt.imshow(horizontal_map,cmap ='gray')
    plt.ioff()
    plt.show()
    plt.ion()
    plt.imshow(vertical_map, cmap='gray')
    plt.ioff()
    plt.show()
    plt.ion()
    plt.imshow(vector_map, cmap='hsv')
    plt.ioff()
    plt.show()
        # horizontal disparity component
        # vertical disparity component
        # disparity vector using color, where the direction of the vector is coded by hue, and the length of the vector is coded by saturation
    return 0
def rgbtohsv(img):
    m, n, k = img.shape
    r, g, b = cv2.split(img)
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = np.zeros((m, n), np.float32)
    S = np.zeros((m, n), np.float32)
    V = np.zeros((m, n), np.float32)
    HSV = np.zeros((m, n, 3), np.float32)
    for i in range(0, m):
        for j in range(0, n):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            V[i, j] = mx
            if V[i, j] == 0:
                S[i, j] = 0
            else:
                S[i, j] = (V[i, j] - mn) / V[i, j]
            if mx == mn:
                H[i, j] = 0
            elif V[i, j] == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn))
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn)) + 360
            elif V[i, j] == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / (V[i, j] - mn) + 120
            elif V[i, j] == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / (V[i, j] - mn) + 240
            H[i, j] = H[i, j] / 2
    HSV[:, :, 0] = H[:, :]
    HSV[:, :, 1] = S[:, :]
    HSV[:, :, 2] = V[:, :]
    return HSV
def main():
    #readinimage
    leftimage,rightimage=readimage()
    gray_left,gray_right=converttogray(leftimage,rightimage)
    #match key_points
    cor_l,cor_r,ncc_match=SIFTcorner(gray_left,gray_right,harris_mindist,SIFT_threshold)
    # cor_l=harriscorner(gray_left,harris_threshold,harris_mindist)
    # cor_r=harriscorner(gray_right,harris_threshold,harris_mindist)
    print([len(cor_l),len(cor_r)])
    # ncc_match=matchpoints(gray_left,gray_right,cor_l,cor_r,NCC_windodim,NCC_threshold)
    print(len(ncc_match))
    #drawmatchpointimage(leftimage,rightimage,cor_l,cor_r,ncc_match,50)
    # #8-point and RANSAC solve F matrix
    F=theransanc(RANSAC_times,cor_l,cor_r,ncc_match)
    drawinliners(leftimage,rightimage,cor_l,cor_r,ncc_match,F,F_inliner_threshold)
    # # #compute dense disparity map
    expoint=[100,200,1]
    expoint=np.array(expoint)
    drawepipole(gray_left,gray_right,expoint,F)
    computedensemap(gray_left,gray_right,F,leftimage)
main()