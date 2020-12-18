import cv2
import pytesseract



pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

img = cv2.imread("images\\text2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#print(img.shape)

# resizing the image to be able to show on screen
rp = 0.5 # resize percentage (0.5 = 50%)
img = cv2.resize(img, (int(img.shape[1] * rp), int(img.shape[0] * rp)))

refPt = []
words = []

def click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        #refPt.append([x,y])
        in_rec = True
        for i in words:
            if i[0]<=x and i[1]<=y and i[2]>=x and i[3]>=y:
                cv2.rectangle(img, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)
                cv2.imshow("image", img)


# Detecting words
hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_data(img)
#print(boxes)

for x,b in enumerate(boxes.splitlines()):

    if x != 0:
        
        b = b.split()
        #print(b)

        if len(b) == 12:
            x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])

            words.append([x,y,x+w,y+h])

            cv2.rectangle(img, (x,y), (w+x, h+y), (0,0,255), 1)
            #cv2.putText(img, b[11], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

cv2.namedWindow("image")
cv2.setMouseCallback("image", click)
cv2.imshow("image", img)
cv2.waitKey(0)
print(refPt)


# detecting characters
#print(pytesseract.image_to_string(img))
# hImg, wImg, _ = img.shape
# boxes = pytesseract.image_to_boxes(img)

# for b in boxes.splitlines():
#     b = b.split(" ")
#     #print(b)
#     x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

#     # editing the image (boxes, predictions)
#     cv2.rectangle(img, (x,hImg-y), (w,hImg-h), (0,0,255), 1)
#     cv2.putText(img, b[0], (x,hImg-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
