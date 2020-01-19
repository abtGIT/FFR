import dlib
import numpy as np
import cv2


class FaceDetector:
        
    def frame_return(self,capObj):
        return capObj.read()

    ### for face detection in video file
    ### format for output ((frame_no, ((face_no, face_img)))
    def face_detect(self,file):
        capObj = cv2.VideoCapture(file)
        frame_count = 0
        frames = []
        #detector = dlib.get_frontal_face_detector()
        while(capObj.isOpened()):
            ret, img = self.frame_return(capObj)
            if ret == True:
                frame_count += 1
                frames.append((frame_count,self.face_detect_image(img)))
            else:
                break
        return tuple(frames)
        
    def face_detect_image(self, img):
        detector = dlib.get_frontal_face_detector()
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        faces = []
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
            faces.append((i,img[d.left():d.right(),d.top():d.bottom()]))
        return tuple(faces)

    ### for face detection in image file
    ### format for output ((face_no, face_img))
#     def face_detect_ifile(self,ifile):
#         image = cv2.imread(ifile)
#         return self.face_detect_image(image)
