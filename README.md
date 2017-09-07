# deep_dream_tensorflow
An implement of google deep dream with tensorflow

# Requirement
- Python3
- OpenCV

# Usage
-python3 main.py --input {input path} --output {output path}

If you don't input any image, it will generate a dream image with noise.

# Tips
Gradient ascent region has uncertainty, even same image with same parameters can generate different pictures.

Larger "iter_num" means a more surprising and more different image.

larger receptive field means more semantic information.
