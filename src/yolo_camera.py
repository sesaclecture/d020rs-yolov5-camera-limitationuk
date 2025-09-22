import torch # PyTorch, YOLOv5 모델은 이 라이브러리를 기반으로 작동
import cv2 # OpenCV

# PyTorch Hub에서 미리 학습된 YOLOv5 모델을 로드
model = torch.hub.load("ultralytics/yolov5", "yolov5m") 
# torch.hub.load(모델이 호스팅된 GitHub 저장소의 주소, 사용할 YOLOv5 모델의 버전)

cap = cv2.VideoCapture(1) # 비디오 장치를 열어 객체 생성

# TODO: Loop for camera frames
# Read frame (BGR to RGB)
while True:
    ret, frame = cap.read() # cap에서 프레임 읽기
    #ret: 프레임을 읽었는지에 대한 불(bool) 값
    #frame: 읽어온 프레임 이미지 데이터 

    # TODO: break the loop on error
    if ret is False:
        break
    
    # 추론 실행 (BGR -> RGB)
    # OpenCV로 읽어온 BGR 형식의 frame을 RGB 형식으로 변환, YOLOv5 모델은 RGB 형식의 이미지를 사용
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 변환된 RGB 프레임을 YOLOv5 모델에 입력, 탐지된 모든 객체의 정보(위치, 신뢰도, 클래스)가 저장
    results = model(rgb_frame)
    
    # TODO: Bounding box 그리기
    for *box, conf, cls in results.xyxy[0]: # results.xyxy[0]: 첫 번째 프레임의 탐지 결과를 나타내는 배열, [x1, y1, x2, y2, 신뢰도, 클래스]
        # TODO: 인식 결과를 표시하기 위한 좌표를 얻음

        x1, y1, x2, y2 = map(int, box) # map: 반복 가능한 객체에 함수를 적용하고 반환

        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        label = f"{model.names[int(cls)]}: {conf:.2f}"

        # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        # cv2.rectangle(원본이미지, 왼쪽 위 꼭짓점, 오른쪽 아래 꼭짓점, 색상, 두께)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #  cv2.putText(원본이미지, 표시할 텍스트, 텍스트가 시작될 위치, 폰트, 크기, 색상, 두께)
        cv2.putText(frame, label, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        #print(label)

    # TODO: 화면 표시
    cv2.imshow("YOLOv5", frame)
    
    # TODO: 종료를 위한 key 처리
    key = cv2.waitKey(1) & 0xff
    if key == 27: # 27은 ESC 키의 아스키 코드
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()