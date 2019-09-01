import pyocr
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np

class TessOCR():
    def __init__(self, imPath = ""):
        self.imPath = imPath
    
    def read_image(self):
        # read using OpenCV
        image = cv2.imread(self.imPath, 0)

        # transform into PIL image to use Tesseract
        self.image = Image.fromarray(np.uint8(image))

        return image
    def extract_text(self):
        
        tools = pyocr.get_available_tools()[0]

        # use option WordBoxBuilder to get bounding boxes
        print("Getting text")
        self.text = tools.image_to_string(self.image, 
                         builder=pyocr.builders.TextBuilder(),
                         lang = tools.get_available_languages()[0])

        print("Getting bounding boxes")
        self.boundingBoxes = tools.image_to_string(self.image, 
                                builder = pyocr.builders.WordBoxBuilder(),
                                lang = tools.get_available_languages()[0]) # lang = 'por'

    def extraction_routine(self):
        
        self.read_image()
        self.extract_text()
        
        return self.text

if __name__ == "__main__":

    imPath = "/home/hrmello/Desktop/Projetos pessoais/ImageDigitalizer/imagemteste.png"
    ocr = TessOCR(imPath=imPath)
    text = ocr.extraction_routine()
    print(text)
    print(ocr.boundingBoxes)
    