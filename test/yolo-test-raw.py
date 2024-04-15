from ultralytics import YOLOWorld

model = YOLOWorld('yolov8x-worldv2.pt')
image_path = './images/kitchen.webp'
results1 = model(image_path, conf=0.1)
results2 = model.track(image_path, conf=0.1)

results1[0].save(filename='./images/result1.jpg')
results2[0].save(filename='./images/result2.jpg')