Attached please find two notebooks: 

one for the segmentation model for 384 x 394 images and 

another for processing high res split to tiles. The model architecture is the same and .pth, .onnx, and .h5 are all included (in the checkpoints folder). 


Attached, you can also find two notebooks I have created for onnx and akida model conversion. I run those files in Google Colab but they can be run on machine with appropriate CUDA installation (NVIDIA toolkit, etc).

After onnx conversion the original model still does well altough it has some glitches. The onnx version of the tiling model tile model does almost as good as the original baseline model. However, after Akida conversion / quantization (even using sample images for PQT) the
model does very poorly. I am not sure the issue is with onnx conversion or quantization  is not done properly. 

Please let me know if above description is not clear or if you have questions.

Thanks,
Ali Kayyam (akayyam@brainchip.com)



PS: code and model checkpoints are also available in machine bcl1 at: 

/home/akayyam/segmentation

/home/akayyam/segmentation/checkpoints



Here is also link to G drive containing code and images for ONNX and Akida conversion:

https://drive.google.com/drive/folders/1RBGZohhcCInLVT_TYnTka_ssV4DFwCnQ?usp=sharing 

