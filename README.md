# Document_verification_IDFC
## Visual_Recognition_IDFC
DOCUMENT VALIDATION:

A RCNN based Image Classifier used to classify Aadhaar Card, Pan card and any other document. This model was trained on a dataset of Aadhaar Cards, PAN cards and Other documents like gas bills, voter ID cards, driving licence etc. collected from customer data. The model was trained over several variations of the images such as blurred or tilted images. The model has an accuracy of 94%. 

## Requirements
- keras
- Tensorflow
- OpenCv
- PIL
- Tesseract-OCR
- Google OCR

 
### Training 
- Due to privacy reasons I was given a very limited dataset, so I had to upsample this data to ensure that I could train my model for all possible scenarios. I used the ```ImageDataGenerator``` module from keras and added variations to the dataset like tilting or selective blurring and increased my dataset size for each class.  

- Freezing the 4 layers allowed me to utilize only the convoluted layers and send the output to a custom fully connected neural network which would be adjusted for only specific images. Given that I had a limited dataset I had to include dropout in the fully connected network to prevent overfitting on the data. 

- This process ensured that the model would understand only the specific document types required to be classified.

### Evaluation
- The trained model was then evaluated on accuracy and through manual checking methods to ensure that the right cardsdocuments were being classified. The model has an accuracy of 94%.

- The operations team manually verified several images using the classifier to ensure that the right documents were classified.

## Getting started
- Pass the image path to the PAN validate function
- The image will fed to the model resposible for classification
- If the image is a PAN card then we will proceed towards the OCR stage.
- 'Name', 'Father name', 'PAN NO', 'Date of Birth' will be extracted from the image.
- 'PAN NO' will be validated using regular expression.
- If the 'PAN NO' is invalid then result will be INVALID PAN CARD with a confidence less than 50%.
- If the document passed is not a pan card then the result will be NOT A PAN CARD with a confidence of 0%. 

## Impact of the project
- This project was packaged into a ready to use library which helped save more 50 hours of manual customer document verification. This helped the operations team to focus on other high priority tasks.

## Future Improvements
- The model can be trained on fake documents and this can help in detecting fraudulent documents. 

