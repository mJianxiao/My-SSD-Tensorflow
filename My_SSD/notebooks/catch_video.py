import cv2
# user: admin
# pwd: 12345
# main: 主码流
# ip: 192.168.1.64
# Channels: 实时数据
# 1： 通道
cap = cv2.VideoCapture("rtmp://test.srs.clevervision.cn:11935/video/1e206ad72b4c4d75824a341459ac497a")
print(cap.isOpened())
while cap.isOpened():
    success, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
