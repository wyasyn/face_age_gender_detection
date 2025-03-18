# Age and Gender Prediction from Face Images

This project uses deep learning models to detect age and gender from a given face image. The system uses OpenCV’s DNN module to perform face detection and pre-trained Caffe models to predict the gender and age of the person in the image. The models are based on the face detection and age-gender recognition networks.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project allows you to detect faces from an image and predict the gender and age group of each detected face. The face detection is done using OpenCV's `CascadeClassifier`, and the age and gender prediction is done using pre-trained Caffe models.

### Models Used:

- **Face Detection Model**: `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- **Gender Prediction Model**: `gender_net.caffemodel`
- **Age Prediction Model**: `age_net.caffemodel`

These models are pre-trained and used to classify the gender and age group of detected faces. The following age groups are used:

- (0-2)
- (4-6)
- (8-12)
- (15-20)
- (25-32)
- (38-43)
- (48-53)
- (60-100)

The gender prediction outputs either "Male" or "Female."

## Features

- Detects faces in an image.
- Predicts the gender (Male/Female) and age group of each detected face.
- Draws bounding boxes around the faces and labels them with the predicted gender and age.
- Works with any given image (using OpenCV to load the image).

## Installation

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

### Install Dependencies

1. Install Python 3 if you haven’t already.
2. Install required Python libraries using pip:

```bash
pip install opencv-python opencv-python-headless numpy matplotlib
```

### Download Models

1. **Face Detection Model**:

   - [Download `res10_300x300_ssd_iter_140000_fp16.caffemodel`](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel)
   - [Download `deploy.prototxt`](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)

2. **Age and Gender Prediction Models**:

   - [Download `gender_net.caffemodel`](https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/age_gender/gender_net.caffemodel)
   - [Download `deploy_gender.prototxt`](https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/age_gender/deploy_gender.prototxt)
   - [Download `age_net.caffemodel`](https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/age_gender/age_net.caffemodel)
   - [Download `deploy_age.prototxt`](https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/age_gender/deploy_age.prototxt)

3. Save the downloaded model files into a directory (e.g., `./models/`).

## Usage

1. Clone or download this repository to your local machine.
2. Place the downloaded models in a directory called `models/` or specify the model paths in the code.
3. Replace `"path_to_your_image.jpg"` with the path to the image you want to analyze.
4. Run the script in a Jupyter notebook or a Python environment.

```bash
python age_gender_prediction.py
```

### Expected Output:

The script will display the image with bounding boxes around detected faces and labels showing the predicted gender and age group for each face.

## Example Output:

- The face is detected, and a rectangle is drawn around it.
- The label shows the predicted gender (Male/Female) and age group (e.g., `(25-32)`).

## Contributing

Feel free to fork the repository, make changes, and create pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
