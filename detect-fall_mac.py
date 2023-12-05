import cv2
import torch
import numpy as np
import math
import requests
import json
from datetime import datetime
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

# 카카오톡 토큰 및 헤더 설정
with open("token.json","r") as kakao:
    tokens = json.load(kakao)
url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
headers = {
    "Content-Type": "application/x-www-form-urlencoded", 
    "Authorization": 'Bearer ' + tokens["access_token"]
}

# Set device for model
device = torch.device("cpu")  # M2 맥북은 CUDA를 지원하지 않으므로 CPU만 사용

# Load model weights
weights = torch.load('yolov7-w6-pose.pt', map_location=device)  # 모델을 CPU에 로드
model = weights['model']
model = model.float()
_ = model.eval()  # 모델을 평가 모드로 설정

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Webcam could not be accessed.")
    exit()

# Define a threshold value for fall detection
fall_threshold = 150  # This is an example value, adjust based on your requirements

def send_message_kakao(token):
    headers = {  # 함수 내에서 headers 재정의
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": 'Bearer ' + token
    }

    template_object = {
        "object_type": "text",
        "text": "경고: 낙상 감지됨. 상황을 확인하세요.",
        "link": {
            "web_url": "http://www.naver.com",  # 웹사이트 URL로 대체하세요
            "mobile_web_url": "http://www.naver.com"  # 웹사이트 URL로 대체하세요
        }
    }
    data = {"template_object": json.dumps(template_object)}
    response = requests.post(url, headers=headers, data=data)
    print("Response from Kakao API:", response.json())  
    return response.json()

last_fall_detection_time = None
# Main loop for frame processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to RGB and process
    orig_image = frame
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, 640, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)  # CPU에서 처리리
    image = image.float()

    # Model inference
    with torch.no_grad():
        output, _ = model(image)

    # Apply non max suppression
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)
    im0 = image[0].permute(1, 2, 0) * 255
    im0 = im0.numpy().astype(np.uint8)
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

    fall_detected = False

    # Process each detected object in the frame
    for idx in range(output.shape[0]):
            #plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

            left_shoulder_y= output[idx][23]
            left_shoulder_x= output[idx][22]
            right_shoulder_y= output[idx][26]
            
            left_body_y = output[idx][41]
            left_body_x = output[idx][40]
            right_body_y = output[idx][44]

            len_factor = math.sqrt(((left_shoulder_y - left_body_y)**2 + (left_shoulder_x - left_body_x)**2 ))

            left_foot_y = output[idx][53]
            right_foot_y = output[idx][56]
            
            if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
            #Plotting key points on Image
              cv2.rectangle(im0,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(0, 0, 255),
                  thickness=5,lineType=cv2.LINE_AA)
              cv2.putText(im0, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)
              fall_detected = True


    # Display the processed frame
    cv2.imshow("Frame", im0)

    current_time = datetime.now()
    
    if fall_detected:
        if last_fall_detection_time is None or (current_time - last_fall_detection_time).total_seconds() > 300:
            send_message_kakao(tokens["access_token"])
            last_fall_detection_time = current_time

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
