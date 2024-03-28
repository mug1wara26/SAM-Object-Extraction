# SAM-Object-Extraction

Simple app to extract objects from images using SAM models and save segments to file

## Setup
First clone the thirdparty repos, then install python requirements
```shell
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

For example, to run edge on truck.jpg

`python main.py edge images/truck.jpg`

## Acknowledgements
This project uses code and models from [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) which were based off the original [SAM](https://github.com/facebookresearch/segment-anything) model
