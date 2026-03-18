# ESP32

We have attempted to run/configure the model on an (original) ESP32 as well, however, given that it is very old and dated hardware, we have been unable to get it to work. Some of the models that do run on that hardware are fairly dated themselves, and have already been thoroughly investigated.

# RESULTS
Comparison between running the model on a few different hardware platforms:

- PC with GPU
- Laptop
- Raspberry Pi Zero/5

# PRE-PROCESSING
We want to test it on our own data, so we will re-train it slightly with specific subset of some of the dataset we have found. Reduce the size by using: **quantization**. Comparison between the different platforms + before and after processing + before and after training.


# SOURCES:
1. How to re-train using only one label (in our case persons): https://github.com/ultralytics/ultralytics/issues/13534
2. Quantization how-to (using some Microsoft bullshit, onnx): https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2
3. Visualizing the neural network: https://netron.app/
4. Training parameters: https://docs.ultralytics.com/modes/train/