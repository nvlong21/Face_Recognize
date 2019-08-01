# AUFR: Authenticate Using Face Recognition

  ![Gui Application View](icon/aufr.png?raw=true "PyQt GUI")

## Algorithms Implemented
  - Eigenfaces
  - Localbinary Pattern Histograms[LBPH]
  - Fisherfaces

# How to use?
 1. Download miniconda/anaconda.
 2. Create environment.
 3. Installation.	
 4. Clone repository.	
 5. Execute.

### 1. Download
 - Download [Mininconda](https://conda.io/miniconda.html).
 - Download [Anaconda](https://www.anaconda.com/).

### 2. Create Environment
 - ```$ conda create -n cv python=3.*```
 - ```$ conda activate cv```

### 3. Package Installation
 - ```$ conda install pyqt=5.*```
 - ```$ conda install opencv=*.*```
 - ```$ conda install -c michael_wild opencv-contrib```

### 4. Clone Repository
 - Clone ```$ git clone https://github.com/indian-coder/authenticate-using-face-recognition.git aufr```
 - Cd into aufr ```$ cd aufr```

### 5. Execute Application
 - Execute  ```$ python main.py```

	Note:Generate atleat two datasets to work properly.
  
  1. Enter name,and unique key.
  2. Check algorithm radio button which you want to train.
  3. Click recognize button.
  4. Click save button to save current displayed image.
  5. Click record button to save video.

## Resources
  - [OpenCV face Recognition](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)
  - [PyQt5 Documentation](http://pyqt.sourceforge.net/Docs/PyQt5/)
