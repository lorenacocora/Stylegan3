# Stylegan3

If you dont have cuda downloaded, get version 12.4 before you start the steps below

Create venv:
	1. python -m venv venv
	2. venv/Scripts/activate

To make dataset:
	1. pip install instaloader
	2. instaloader profile <profile_name>
	3. git clone https://github.com/dvschultz/dataset-tools.git
	4. cd dataset-tools
	5. pip install -r requirements.txt
	6. python dataset-tools.py -i <folder_location> -o <processed_folder> --process_type crop_to_square --max_size 512
  7. zip the processed folder created


To train:
	1. git clone https://github.com/PDillis/stylegan3-fun.git
	2. Open file: /content/stylegan3-fun/torch_utils/ops/grid_sample_gradfix.py
	   	Edit line 60
           	from: op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
           	to: op, _ = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
	3. cd stylegan3-fun
	4. pip install ninja click numpy requests psutil Pillow scipy setuptools==72.1.0 
	5. Create a 'libs' folder in venv/Scripts. Copy python39.lib from your Python39/libs folder to your venv/Scripts/libs. Make sure you activate venv.
		- python -m pip install light-the-torch
		- ltt install torch
	7. python train.py --data=../datasets/dataset-512.zip --cfg=stylegan2 --gpus=1 --batch=8 --img-snap=1 --resume=ffhq512 --snap-res=1080p --snap=10


To visualize:
	1. pip install imgui==1.4.1 glfw PyopenGL matplotlib 
	2. python visualizer.py
