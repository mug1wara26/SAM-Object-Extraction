# SAM-Object-Extraction

Simple app to extract objects from images using SAM models and save segments to file

## Setup
First clone the repo and thirdparty repos, then install python requirements
```shell
git clone https://github.com/mug1wara26/SAM-Object-Extraction.git && cd SAM-Object-Extraction
git submodule update --init --recursive

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, you need to install the weights for the models
```shell
mkdir weights
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth
wget -P weights/ https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt
```

## Running the code
Currently the code only supports EdgeSAM and MobileSAM, more models are planned to be supported. EdgeSAM seems to perform better and faster so it is recommended. main.py takes the following arguments:

`python main.py <model type> <path to image>`

For example, to run EdgeSAM on truck.jpg

`python main.py edge images/truck.jpg`

## Point mode
App launches in Positive point mode, click on any object to generate a mask, add more points if the mask is too specific
![image](https://github.com/mug1wara26/SAM-Object-Extraction/assets/42673064/dace3995-a551-4b1e-903b-8650e614854a)
![image](https://github.com/mug1wara26/SAM-Object-Extraction/assets/42673064/8c8ce2bd-d602-4bd5-abb4-70d35dd1b1ac)

## Negative points
Press 'n' while on the window to go into negative point mode to remove sections of the object that you don't want. Note that it doesn't perform very well with just points
![image](https://github.com/mug1wara26/SAM-Object-Extraction/assets/42673064/c154bd54-c75a-4261-8dc7-2dd15ad9ff90)

## Box mode
Press 'b' to go to box mode, box prompts seem to work best with the model, positive and negative points can be used to improve the prompt
![image](https://github.com/mug1wara26/SAM-Object-Extraction/assets/42673064/5bbc43b5-2e64-436b-b9d9-9ea3856097fe)
![image](https://github.com/mug1wara26/SAM-Object-Extraction/assets/42673064/c4dab69c-39a4-4d37-b4d5-671940d93e1f)

## Save
Press 's' to save the segmented object into a transparent png file. Image will be saved into a segments folder
![image](https://github.com/mug1wara26/SAM-Object-Extraction/assets/42673064/93d0726f-56ba-4904-a9c4-bf85a314d2c7)

## Acknowledgements
This project uses code and models from [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) which were based off the original [SAM](https://github.com/facebookresearch/segment-anything) model
