import cv2
import numpy as np

string = "335830059_58a4bcecb1.jpg";
orig_img = cv2.imread(string)
txt_file = open(string + ".txt","r")
text=  txt_file.readline()

oeoeooee = "01a_clermont.jpg"
oe2 =oeoeooee.split('.')

print len(oe2)
#print text
ney= text.split(',')
#print ney[1]

new_string = string.split('.')

height, width, channels = orig_img.shape
blank_image = np.zeros((height,width,1), np.uint8)
cv2.rectangle(blank_image,(int(ney[0]),int(ney[1])),(int(ney[2]),int(ney[3])),(255,255,255),-1)


text=  txt_file.readline()
print len(text)

 
cv2.imwrite(new_string[0]+"_GT.jpg",blank_image)
txt_file.close()  
