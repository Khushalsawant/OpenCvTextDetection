# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw
from PIL import Image, ImageEnhance,ImageFilter  ## Need to import PIL package (Python Imaging Library)
import cv2 as cv

from translate import Translator
import nltk.data
from nltk import tokenize
import language_check

tool = language_check.LanguageTool('en-US')

#nltk.download()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

face_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_smile.xml')
eyeglasses_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
frontalface_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_frontalcatface.xml')
frontalcatface_extended_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_frontalcatface_extended.xml')
profileface_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_profileface.xml')

src_path = 'C:/Users/khushal/Pictures/test.jpg'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((10, 10), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)

    # Write image after removed noise
    #cv2.imwrite(src_path + "removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 2)

    # Write the image after apply opencv to do some ...
    #cv2.imwrite(src_path + "thres.png", img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(Image.open(src_path),lang='eng')# + "thres.png"))

    # Remove template file
    # os.remove(temp)

    return result

def detect_faces(Path_of_img):
    # Load an color image in coloured manner
    img = cv.imread(Path_of_img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    scale_factor_np = np.arange(4.1,10.0,0.1)
    min_neighbours_np = np.arange(4, 10, 1)
    smile_scale_factor_np = np.arange(2, 10.0, 0.1)
    smile_min_neighbours_np = np.arange(4,10,1)
    print('scale_factor_np = ', scale_factor_np,'\n min_neighbours = ', min_neighbours_np)
    for j in min_neighbours_np:
        min_neighbours = j
        for i in scale_factor_np:
            scale_factor = i
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbours)
            frontal_faces = frontalface_cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                                                 minNeighbors=min_neighbours)
            frontalface_extended = frontalcatface_extended_cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                                                                    minNeighbors=min_neighbours)
            profilefaces = profileface_cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                                                minNeighbors=min_neighbours)
            if len(faces) != 0:
                print('Faces found:  ', len(faces))
                break
            elif len(frontal_faces) != 0:
                print('Frontal faces = ', len(frontal_faces))
                break
            elif len(frontalface_extended) != 0:
                print('Frontal faces extended= ', len(frontalface_extended))
                break
            elif len(profilefaces) != 0:
                print('Profile Faces = ', len(profilefaces))
                break
        break

    print("scale_factor = ", scale_factor)
    print('min_neighbours = ', min_neighbours)
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #    cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_glass = eyeglasses_cascade.detectMultiScale(roi_gray)
            print("Len of eye_glass = ", len(eye_glass))
            eye_detect = eye_cascade.detectMultiScale(roi_gray)
            print("Len of eye_detect", len(eye_detect))
            for (ex, ey, ew, eh) in eye_glass:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            #for (ex, ey, ew, eh) in eye_detect:
            #    cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            for j in smile_min_neighbours_np:
                min_neighbours = j
                for i in smile_scale_factor_np:
                    scale_factor = i
                    smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbours, minSize=(15, 15),
                                                   flags=cv.CASCADE_SCALE_IMAGE)
                    if len(smile) > 0:
                        break
                break

            print('scale_factor for smile = ', scale_factor,'\n min_neighbours for smile = ', min_neighbours)
            print("Smiles found", len(smile))
            for (ex, ey, ew, eh) in smile:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    elif len(frontal_faces) != 0:
        for (x, y, w, h) in frontal_faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #    cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_glass = eyeglasses_cascade.detectMultiScale(roi_gray)
            print("Len of eye_glass = ", len(eye_glass))
            eye_detect = eye_cascade.detectMultiScale(roi_gray)
            print("Len of eye_detect", len(eye_detect))
            for (ex, ey, ew, eh) in eye_glass:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in eye_detect:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbours, minSize=(15, 15),
                                                   flags=cv.CASCADE_SCALE_IMAGE)
            # print("Smiles found", len(smile))
            for (ex, ey, ew, eh) in smile:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    elif len(frontalface_extended) != 0:
        for (x, y, w, h) in frontalface_extended:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #    cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_glass = eyeglasses_cascade.detectMultiScale(roi_gray)
            print("Len of eye_glass = ", len(eye_glass))
            eye_detect = eye_cascade.detectMultiScale(roi_gray)
            print("Len of eye_detect", len(eye_detect))
            for (ex, ey, ew, eh) in eye_glass:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in eye_detect:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbours, minSize=(15, 15),
                                                   flags=cv.CASCADE_SCALE_IMAGE)
            # print("Smiles found", len(smile))
            for (ex, ey, ew, eh) in smile:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    elif len(profilefaces) != 0:
        for (x, y, w, h) in profilefaces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #    cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_glass = eyeglasses_cascade.detectMultiScale(roi_gray)
            print("Len of eye_glass = ", len(eye_glass))
            eye_detect = eye_cascade.detectMultiScale(roi_gray)
            print("Len of eye_detect", len(eye_detect))
            for (ex, ey, ew, eh) in eye_glass:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in eye_detect:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbours, minSize=(15, 15),
                                                   flags=cv.CASCADE_SCALE_IMAGE)
            # print("Smiles found", len(smile))
            for (ex, ey, ew, eh) in smile:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    else:
        print('No faces detected in the given source image')

    cv.imshow('FACES', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def translate_text(input_str,to_Language):
    translator = Translator(to_lang=to_Language)
    translation = translator.translate(input_str)
    return translation

image_to_str = get_string(src_path)
img_str = image_to_str.strip()
print('--- Start recognize text from image ---')
print(img_str)
print("------ Done -------")
#img_str_nltk = '\n-----\n'.join(tokenizer.tokenize(img_str))

if img_str:
    img_str_nltk1 = tokenize.sent_tokenize(img_str)
    # nltk.extract_test_sentences(str(img_str_nltk1),comment_chars='%.!,')
    # words1 = nltk.word_tokenize(text=img_str_nltk,language='english')
    # print(ts.tokenize(img_str_nltk1))

    # lang_to_translate = input("Enter the language in which you want to translate text eg Hindi, German")
    lang_to_translate = 'german'
    for i in range(len(img_str_nltk1)):
        img_text = str(img_str_nltk1[i])
        print(img_text)
        print('-' * 60)
        img_text_translated = translate_text(img_text, lang_to_translate)
        if i ==0:
            translated_text = img_text_translated
            print(img_text_translated)

else:
    print("Input image doesn't contain any text")


img_face = detect_faces(src_path)

print(translated_text)

text = translated_text #.decode('utf-8')

img0 = np.zeros((512, 1248, 3), np.uint8)
cv2.imwrite('C:/Users/khushal/Pictures/background.jpg', img0)
cv2.destroyAllWindows()

img = Image.open("C:/Users/khushal/Pictures/background.jpg")
draw = ImageDraw.Draw(img)
# font = ImageFont.truetype(<font-file>, <font-size>)
font = ImageFont.truetype("arial.ttf", 26, encoding="unic")
# draw.text((x, y),"Sample Text",(r,g,b))
draw.text((0, 0),text,(255,255,255),font=font)
img.save('C:/Users/khushal/Pictures/text_translated.jpg')


#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img, text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow('image', cv2.imread('C:/Users/khushal/Pictures/text_translated.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()

######################################################

##https: // pythonprogramming.net / blurring - smoothing - python - opencv - tutorial /

# Thresholding OpenCV ==>

import cv2

path_of_img = 'C:/Users/khushal/Pictures/download.jpg'

img = cv2.imread(path_of_img)

# https://pythonprogramming.net/thresholding-image-analysis-python-opencv-tutorial/

'''

he idea of thresholding is to further-simplify visual data for analysis.

First, you may convert to gray-scale, but then you have to consider that grayscale still has at least 255 values.

What thresholding can do, at the most basic level, is convert everything to white or black, based on a threshold value

'''

grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

cv2.imshow('original', img)

cv2.imshow('Adaptive threshold', th)

cv2.waitKey(0)

cv2.destroyAllWindows()

# Image arithmetics and Logic OpenCV ==>


import cv2

import numpy as np

path_of_img1 = 'C:/Users/khushal/Pictures/khushal.jpg'

path_of_img2 = 'C:/Users/khushal/Pictures/download-logo.jpg'

# Load two images

img1 = cv2.imread(path_of_img1)

img2 = cv2.imread(path_of_img2)


print("Shape of img1 = ", img1.shape,"\n Shape of img2 ", img2.shape)
'''

https://pythonprogramming.net/image-arithmetics-logic-python-opencv-tutorial/

'''

img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret1, mask1 = cv2.threshold(img1gray, 115, 255, cv2.THRESH_BINARY_INV)
contours,hierarchy1 = cv2.findContours(mask1,2,1)
cnt1 = contours[0]

# Now create a mask of logo and create its inverse mask
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# add a threshold
ret, mask = cv2.threshold(img2gray, 115, 255, cv2.THRESH_BINARY_INV)
contours,hierarchy = cv2.findContours(mask,2,1)
cnt2 = contours[0]

img_shape_match = cv2.matchShapes(cnt1,cnt2,1,0.0)
print('img_shape_match = ', img_shape_match)

# I want to put logo on top-left corner, So I create a ROI

rows, cols, channels = img2.shape
rows1, cols1, channels1 = img1.shape

if rows1 > rows and cols1 > cols:
    print('Image first is bigger', img1.shape)
    roi = img1[0:rows, 0:cols]
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Image second is bigger', img2.shape)
    roi = img2[0:rows, 0:cols]
    mask_inv1 = cv2.bitwise_not(mask1)
    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv1)
    img1_fg = cv2.bitwise_and(img1, img1, mask=mask1)
    dst = cv2.add(img2_bg, img1_fg)
    img2[0:rows, 0:cols] = dst
    cv2.imshow('res', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



############################################################


'''
Doing reverse pixel using scipy instead of openCV
'''
import scipy.misc
from scipy import misc
from scipy.misc.pilutil import Image

im = Image.open('C:/Users/khushal/Pictures/download1.jpg')
im_array = scipy.misc.fromimage(im)
im_inverse = 255 - im_array
im_result = scipy.misc.toimage(im_inverse)
misc.imsave('C:/Users/khushal/Pictures/result.jpg', im_result)