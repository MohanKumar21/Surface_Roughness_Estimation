import numpy as np
import cv2
import torch
import glob as glob
import matplotlib.pyplot as plt 
from model import create_model
OUTS=[]
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=1).to(device)
model.load_state_dict(torch.load(
    './Outputs/model30.pth', map_location=device
))
model.eval()

# directory where all the images are present
DIR_TEST = './test'
test_images = glob.glob(f"{DIR_TEST}/*")
test_images=list(set([i[:-4]+".jpg" for i in test_images]))
print(f"Test instances: {len(test_images)}")

# classes: 0 index is reserved for background
#CLASSES = ['Atmospheric pressure limitation', 'Authorized Representative', 'Batch Code', 'Biological risks', 'Catalogue Number', 'Caution', 'Consult instructions', 'Contains Latex', 'Contains sufficient for -n- tests', 'Control', 'Date of Manufacture', 'Do not resterilize', 'Do not reuse', 'Do not use if package is damaged', 'Drops per milliliter', 'Fluid Path', 'For IVD perfomance evaluation only', 'Fragile Handle with care', 'Humidity Limitation', 'In vitro Diagnostic Medical device', 'Keep Dry', 'Keep away from sunlight', 'Liquid filter with pore size', 'Lower limit of temperature', 'Manufacturer', 'Negative Control', 'Non Pyrogenic', 'Non sterile', 'One way valve', 'Patient Number', 'Positive Control', 'Protect from heat radioactive sources', 'Sampling site', 'Sterile Fluid Path', 'Sterile', 'Sterilized using aseptic techniques', 'Sterilized using ethylene oxide', 'Sterilized using irradiation', 'Sterilized using steam', 'Temperature Limit', 'Upper limit of temperature', 'Use By Date',"glove pairs", "medical device", "powder free","Recyclable","background"]
#CLASSES=['person','dog']
CLASSES=["Workpiece"]
print(CLASSES)
# define the detection threshold... 
# ... any detection having score below this will be discarded

detection_threshold = 0.0001

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    #image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    OUTS.append(outputs)
        
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
      
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        print(pred_classes)
        
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)

    plt.imshow(orig_image)

    cv2.imwrite(f"./test_predictions/{i}.jpg", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')

