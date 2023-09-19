import torch

BATCH_SIZE = 1
RESIZE_TO = 256 
NUM_EPOCHS = 30

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = './final_augmented'

VALID_DIR = './valid'


# CLASSES = ['Atmospheric pressure limitation', 'Authorized Representative', 'Batch Code', 'Biological risks', 'Catalogue Number', 'Caution', 'Consult instructions', 'Contains Latex', 'Contains sufficient for -n- tests', 'Control', 'Date of Manufacture', 'Do not resterilize', 'Do not reuse', 'Do not use if package is damaged', 'Drops per milliliter', 'Fluid Path', 'For IVD perfomance evaluation only', 'Fragile Handle with care', 'Humidity Limitation', 'In vitro Diagnostic Medical device', 'Keep Dry', 'Keep away from sunlight', 'Liquid filter with pore size', 'Lower limit of temperature', 'Manufacturer', 'Negative Control', 'Non Pyrogenic', 'Non sterile', 'One way valve', 'Patient Number', 'Positive Control', 'Protect from heat radioactive sources', 'Sampling site', 'Sterile Fluid Path', 'Sterile', 'Sterilized using aseptic techniques', 'Sterilized using ethylene oxide', 'Sterilized using irradiation', 'Sterilized using steam', 'Temperature Limit', 'Upper limit of temperature', 'Use By Date',"glove pairs", "medical device", "powder free","Recyclable"]
# CLASSES=['person','chair','car','dog','bottle','cat','bird','pottedplant','sheep','boat','aeroplane','tvmonitor','sofa','bicycle','horse','diningtable','motorbike','cow','train','bus']
# CLASSES=['person','dog']
CLASSES=["Workpiece"]
NUM_CLASSES=1


VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR = './Outputs'
SAVE_PLOTS_EPOCH = 2 
SAVE_MODEL_EPOCH = 2