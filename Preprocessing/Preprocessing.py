import cv2

listy = ['dewcan.jpg', 'boxangle.jpg', 'droid.jpg', 'droidangle.jpg', 'droidsticker.jpg', 'greenbox.jpg']


def resize_image(filenames):
    for pic in filenames:
        img = cv2.imread('Images/' + pic)
        newimg = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('s'+pic, newimg)


resize_image(listy)
