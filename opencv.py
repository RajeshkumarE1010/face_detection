import cv2
import time
import mediapipe as mp

class FaceDectector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon=minDetectionCon
        self.mpFacedetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFacedetection.FaceDetection(self.minDetectionCon)

    
    # if not success:
    #     print("Error reading frame!")
    #     break
    def findFaces(self,img, draw=True):

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results= self.faceDetection.process(imgRGB)
        # print(results)
        bboxs=[]
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                # print(id,detection)
                # mpDraw.draw_detection(img,detection)
                bboxc= detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                # bbox= int(bboxc.xmin *iw),int(bboxc.ymin *ih)/int(bboxc.width *iw),int(bboxc.height *ih)
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
                bboxs.append([id,bbox,detection.score])
                cv2.rectangle(img, bbox, (0, 0, 255), 2)  # Assuming bbox is now a tuple of integers
                cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

        return img,bboxs  
            
    
def main():
    cap=cv2.VideoCapture("example_omg.mp4")
    pTime=0
    detector= FaceDectector()
    while True:
        success, img=cap.read()
        img, bboxs=detector.findFaces(img)
        cTime=time.time()
        fps= 1 /(cTime - pTime)
        pTime=cTime
        cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),2)
    
        resized=cv2.resize(img,(500,600))
        cv2.imshow("image",resized)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break




if __name__ =="__main__":
    main()

# -------------------------------------------------------------------------------------------------------------

# import dlib # dlib for accurate face detection
# import cv2 # opencv
# import imutils # helper functions from pyimagesearch.com

# # Grab video from your webcam
# stream = cv2.VideoCapture("example_omg.mp4")

# # Face detector
# detector = dlib.get_frontal_face_detector()

# # Fancy box drawing function by Dan Masek
# def draw_border(img, pt1, pt2, color, thickness, r, d):
#     x1, y1 = pt1
#     x2, y2 = pt2

#     # Top left drawing
#     cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
#     cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
#     cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

#     # Top right drawing
#     cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
#     cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
#     cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

#     # Bottom left drawing
#     cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
#     cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
#     cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

#     # Bottom right drawing
#     cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
#     cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
#     cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    
# count = 0

# while True:
#     if count % 3 != 0:
#         # read frames from live web cam stream
#         (grabbed, frame) = stream.read()

#         # resize the frames to be smaller and switch to gray scale
#         frame = imutils.resize(frame, width=700)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Make copies of the frame for transparency processing
#         overlay = frame.copy()
#         output = frame.copy()

#         # set transparency value
#         alpha  = 0.5

#         # detect faces in the gray scale frame
#         face_rects = detector(gray, 0)

#         # loop over the face detections
#         for i, d in enumerate(face_rects):
#             x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()

#             # draw a fancy border around the faces
#             draw_border(overlay, (x1, y1), (x2, y2), (162, 255, 0), 2, 10, 10)

#         # make semi-transparent bounding box
#         cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

#         # show the frame
#         cv2.imshow("Face Detection", output)
#         key = cv2.waitKey(1) & 0xFF
        
#     count +=1
#     # press q to break out of the loop
#     if key == ord("q"):
#         break

# # cleanup
# cv2.destroyAllWindows()
# stream.stop()
# ---------------------------------------------------------------------------------------------------------
# from facenet_code.detection import Detection
# from facenet_code.encoder import Encoder
# from scipy.linalg import svd
# from imutils import paths
# import numpy as np
# import argparse
# import cv2
# import os

# class DetectBlur(object):
#     def __init__(self, video, threshold=0.8):
#         self.video = video
#         self.threshold = threshold
#         print(self.threshold)
#         self.video_frames = []

#         self.detect = Detection()

#         self.process()
    
#     def process(self):
#         self.create_output_folder()
#         self.get_video_frames()
#         self.detect_blur()

#     def create_output_folder(self):
#         if not os.path.isdir('output'):
#             os.mkdir('output')
#         video_name = self.video.split('.')[0]
#         if not os.path.isdir('output/'+video_name):
#             os.mkdir('output/'+video_name)
#         if not os.path.isdir('output/'+video_name+'/'+'frames'):
#             os.mkdir('output/'+video_name+'/'+'frames')

#     def get_blur_degree(self, img, sv_num=10):
#         gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         u, s, v = np.linalg.svd(gray_img)
#         top_sv = np.sum(s[0:sv_num])
#         total_sv = np.sum(s)
#         return top_sv/total_sv

#     # def get_blur_map(self, img, win_size=10, sv_num=3):
#     #     gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     #     new_img = np.zeros((gray_img.shape[0]+win_size*2, gray_img.shape[1]+win_size*2))
#     #     for i in range(new_img.shape[0]):
#     #         for j in range(new_img.shape[1]):
#     #             if i<win_size:
#     #                 p = win_size-i
#     #             elif i>gray_img.shape[0]+win_size-1:
#     #                 p = gray_img.shape[0]*2-i
#     #             else:
#     #                 p = i-win_size
#     #             if j<win_size:
#     #                 q = win_size-j
#     #             elif j>gray_img.shape[1]+win_size-1:
#     #                 q = gray_img.shape[1]*2-j
#     #             else:
#     #                 q = j-win_size
#     #             new_img[i,j] = img[p,q]
#     #     blur_map = np.zeros((gray_img.shape[0], gray_img.shape[1]))
#     #     max_sv = 0
#     #     min_sv = 1
#     #     for i in range(gray_img.shape[0]):
#     #         for j in range(gray_img.shape[1]):
#     #             block = new_img[i:i+win_size*2, j:j+win_size*2]
#     #             u, s, v = np.linalg.svd(block)
#     #             top_sv = np.sum(s[0:sv_num])
#     #             total_sv = np.sum(s)
#     #             sv_degree = top_sv/total_sv
#     #             if max_sv < sv_degree:
#     #                 max_sv = sv_degree
#     #             if min_sv > sv_degree:
#     #                 min_sv = sv_degree
#     #             blur_map[i, j] = sv_degree
#     #     blur_map = (blur_map-min_sv)/(max_sv-min_sv)
#     #     return blur_map

#     def get_video_frames(self):
#         vidcap = cv2.VideoCapture("Times Square Crowd People 2 HD Video Background.mp4")
#         success, image = vidcap.read()
#         count = 0
#         while success:
#             self.video_frames.append(image)    
#             success, image = vidcap.read()

#     def print_box(self, frame, name, blur_degree, face_bb, color):
#         left, top, right, bottom = face_bb
#         width = right - left
#         height = bottom - top

#         if height > width:
#             tam = int(height/4)
#         else:
#             tam = int(width/4)

#         cv2.putText(frame, name, (right + 15, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#         cv2.putText(frame, blur_degree, (right + 15, top + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

#         cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), color, 1)

#         cv2.line(frame, (left, top), (left+tam, top), color, 3)
#         cv2.line(frame, (left, top), (left, top+tam), color, 3)

#         cv2.line(frame, (left, bottom), (left, bottom-tam), color, 3)
#         cv2.line(frame, (left, bottom), (left+tam, bottom), color, 3)

#         cv2.line(frame, (right, top), (right-tam, top), color, 3)
#         cv2.line(frame, (right, top), (right, top+tam), color, 3)

#         cv2.line(frame, (right, bottom), (right-tam, bottom), color, 3)
#         cv2.line(frame, (right, bottom), (right, bottom-tam), color, 3)

#     def detect_blur(self):
#         output_video = None
#         if output_video is None:
#             video_name = self.video.split('.')[0]
#             size = (self.video_frames[0].shape[1], self.video_frames[0].shape[0])
#             fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#             output_video = cv2.VideoWriter('output/'+video_name+'/'+video_name+'.avi',fourcc, 5, size, True)
#         for i, frame in enumerate(self.video_frames):
#             print('[INFO] detecting blur in image '+str(i+1)+'/'+str(len(self.video_frames)))
#             faces = self.detect.find_faces(frame)
#             if len(faces) > 0:
#                 for face in faces:
#                     if face.confidence > 0.9:
#                         text = "Not Blurry"
#                         boxes = face.bounding_box.astype(int)
#                         left, top, right, bottom = boxes
#                         face_image = frame[top:bottom, left:right]
#                         blur_degree = self.get_blur_degree(face_image)
#                         if blur_degree > self.threshold:
#                             text = "Blurry"
#                         self.print_box(frame, text, "{:.2f}".format(blur_degree), boxes, (255,255,255))
#             if output_video is not None:
#                 output_video.write(frame)
#                 cv2.imwrite('output/'+video_name+'/'+'frames/frame_'+str(i+1)+'.jpg', frame)
#         if output_video is not None:
#             output_video.release()


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument('video', type=str, help='the video input to detect blurry faces')
#     ap.add_argument('--threshold', default=0.8, type=float, help='the threshold of blur degree to classify if some face is blurry or not')
#     args = vars(ap.parse_args())

#     DetectBlur(video=args['video'], threshold=args['threshold'])

    
