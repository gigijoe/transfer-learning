#from pytorch_model import preprocess_image, postprocess
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

import cv2
from albumentations import (Compose,Resize,)
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensor

import argparse
import time

parser = argparse.ArgumentParser(description='PyTorch TensorRT Evaluate')
parser.add_argument("--image", '-i',type=str,
                        default="image.jpeg", help="image file")
parser.add_argument("--onnx", '-o', type=str,
                        default='model.onnx', help="source onnx file")
parser.add_argument("--engine", '-e', type=str,
                        help="source engine file")
args = parser.parse_args()

def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        Resize(32, 32, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensor(),
    ])

    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(image=input_img)["image"]
    # prepare batch
    batch_data = torch.unsqueeze(input_data, 0)

    return batch_data

ONNX_FILE_PATH = args.onnx
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

ENGINE_FILE_PATH = args.engine

def load_engine(trt_runtime, engine_path):
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)

    return engine

def save_engine(engine, engine_path):
    
    buf = engine.serialize()
    with open(engine_path, 'wb') as f:
        f.write(buf)

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    #network = builder.create_network()
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    else:
        print("fp16 not support !!!")

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        #parser.parse(model.read())
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    if parser.num_errors == 0:
        print(f'Completed parsing of ONNX file, number of layers are {network.num_layers}')

    last_layer = network.get_layer(network.num_layers - 1)
    # Check if last layer recognizes it's output
    if not last_layer.get_output(0):
        # If not, then mark the output using TensorRT API
        network.mark_output(last_layer.get_output(0))

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine, context


def main():
    # initialize TensorRT engine and parse ONNX model
    if args.engine:
        trt_runtime = trt.Runtime(TRT_LOGGER)
        print(f'Loading engine {args.engine}...')
        engine = load_engine(trt_runtime, args.engine)
        context = engine.create_execution_context()
        print("Completed loading Engine")
    else:
        engine, context = build_engine(ONNX_FILE_PATH)
        print('Save model.engine')
        save_engine(engine, 'model.engine')

    print('Start inference ...')
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    infer_start = time.time()
    # preprocess input data
    host_input = np.array(preprocess_image(args.image).numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    infer_end = time.time()
    print(f'Inference time : {infer_end - infer_start}s')
    print(f'host_output = {host_output}')

    output = host_output
    # Result postpro

    index = np.argmax(output)
    print(f'Best : {index}')

    # postprocess results
    #output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    #postprocess(output_data)


if __name__ == '__main__':
    main()

