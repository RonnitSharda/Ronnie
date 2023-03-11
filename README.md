# Crack Check - Video 

## Outline of the code 

Knn is applied on the video at first then the mask of the screen is calculated.
``` python 
 knnSubtractor = cv2.createBackgroundSubtractorKNN(
        detectShadows=False, history=colorchange, dist2Threshold=20000)
    knn = cv2.VideoWriter(knn_str, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fps, size)
         knnMask = knnSubtractor.apply(frame)
        fg1 = cv2.copyTo(frame, knnMask)
```
This above code applies knn on each frame and saves it in 'fg1'



``` python

def makemask(f, m1):
    """
    param f: the frame
    param m1: The mask that we want to add to
    return: the mask.
    """
    m2 = np.zeros((f.shape), np.uint8)
    m2 = cv2.bitwise_or(m2, f)
    m1 = cv2.addWeighted(m2, 0.17, m1, 1, 0)
    return m1
```
This above code takes two images, one of which is a mask, and it returns a new mask that is the combination of the two.
This is used to make the initial mask of the phone's screen.

The next step involves making the final mask by applying different image morphological operations. 

``` python
def getmask(img, str):
    """
    It takes an image, converts it to grayscale, applies a threshold, erodes and dilates it.
    :param img: The image to be processed
    :param str: The name of the image
    """
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r1, t1 = cv2.threshold(g_img, 75, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    t1 = erosion(t1, kernel, 5)
    t1 = erosion(t1, kernel, 5)
    t1 = dilation(t1, kernel, 5)
    t1 = dilation(t1, kernel, 5)
    for i in range(5):
        t1 = dilation(t1, kernel, 3)
    for i in range(5):
        t1 = erosion(t1, kernel, 3)
```
The next step is to get the screen from the input video by applying the mask we have got from the above functions  and apply connected components analysis to mark the cracks seperately by red color 
``` python
# It reads the video frame by frame and checks if the
# current frame is the frame where the screen is to be cut. If it is, it cuts the screen and saves
# it in the folder 'screens'. It also calls the function connectedcomponents() to find the
# connected components in the screen and saves the image in the folder 'cracks'. It then combines
# the connected components image with the mask and saves it in the folder 'cracks'. It then
# displays the final mask, the screen, the connected components image and the combined image in a
 # single frame and saves it in the folder 'final_output'.
    while (1):
        r, f = c1.read()
        if not r or current_frame == when_to_cut:
            break
        current_frame = current_frame+1
        o2 = cv2.bitwise_and(mask2, f)
    if current_frame == screen_cut or current_frame % seconds == screen_cut:

            print('current_frame = ', current_frame)
            ct = ct+1
            print('ct = ', ct)
            if ct < 7:
                cv2.imwrite('screens/'+s2+'.' +
                            str(ct)+'.jpg', o2)
                
                filename = 'cracks/crack'+s2+'.' + str(ct)+'.jpg'
                file_list.append(filename)
                mask3, pic = connectedcomponents(o2, mask3, s2, ct, filename)
                pic2 = pic
                pic2[np.where((pic2 == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
                mask3 = np.bitwise_or(mask3, pic2)
                t = cv2.bitwise_or(mask3, c_img)
                disp = cv2.hconcat([mask2, o2, pic, t])
                cv2.putText(disp, "Final Mask", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(disp, "Screen", (w+30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(disp, "CC", (w*2+30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(disp, "CC-Combined", (w*3+30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(disp, "Stage 2", (int(w*1.5), h-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3,)
            else:
                disp = cv2.hconcat([mask2, o2, blank, blank])


def connectedcomponents(img, mask, str1, j, filename):
    """
    It takes an image, finds the largest connected component in it, and returns the image with all other
    connected components colored red

    :param img: The input image
    :param mask: The output mask image
    :param str1: The name of the image
    :param j: the number of the image
    :return: The mask and the connected components image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cannyimg = cv2.Canny(img, 30, 120, apertureSize=3, L2gradient=True)
    num_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(
        cannyimg, connectivity=8)
    cannyimg = cv2.cvtColor(cannyimg, cv2.COLOR_GRAY2BGR)
    id = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x1, y1, w1, h1, a1 = stats[id]
    for i in range(1, num_labels):
        current_frame, y, w, h, a = stats[i]
        if current_frame >= x1 and current_frame+w < x1+w1 and y >= y1 and y+h < y1+h1:
            cannyimg[labels == i] = [(0, 0, 255)]
    cv2.imwrite(filename, cannyimg)  # for test 2
    cc = cannyimg
    return mask, cc
```








