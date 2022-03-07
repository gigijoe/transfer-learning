#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <unistd.h>
#include <dirent.h> 

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int OUTPUT_SIZE = 2;

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

using namespace nvinfer1;

static Logger gLogger;

void doInference(IExecutionContext & context, float* input, float* output, int batchSize, 
    const int inputIndex, const int outputIndex, void* buffers[2], cudaStream_t & stream)
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void oneInference(IExecutionContext & context, const int inputIndex, const int outputIndex, void* buffers[2], cudaStream_t & stream)
{
    const int batchSize = 1;
    // Do very first inference to avoid slow processing
    float prob[OUTPUT_SIZE];
    // Subtract mean from image
    float one[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        one[i] = 1.0;

    doInference(context, one, prob, batchSize, inputIndex, outputIndex, buffers, stream);    
}

void imageInference(IExecutionContext & context, const char *image_file, const int inputIndex, const int outputIndex, void* buffers[2], cudaStream_t & stream)
{
    const int batchSize = 1;
    float prob[OUTPUT_SIZE];

    // Load image file
    cv::Mat img = cv::imread(image_file);
    if(img.empty()) {
        std::cout << "Error open image file " << image_file << " !!!" << std::endl;
        return;
    }

    // Do inference now
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.f / 255.f);

//transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
// https://blog.csdn.net/hello_dear_you/article/details/119863264
    cv::subtract(img, cv::Scalar(0.4914, 0.4822, 0.4465), img, cv::noArray(), -1);
    cv::divide(img, cv::Scalar(0.2023, 0.1994, 0.2010), img, 1, -1);

// TensorRT requires your image data to be in NCHW order. But OpenCV reads this in the NHWC order.
// https://pfnet-research.github.io/menoh/md_tutorial.html
    std::vector<float> chw(img.channels() * img.rows * img.cols);
    for(int y = 0; y < img.rows; ++y) {
        for(int x = 0; x < img.cols; ++x) {
            for(int c = 0; c < img.channels(); ++c) {
                chw[c * (img.rows * img.cols) + y * img.cols + x] =
                  img.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    float *data = &chw[0];

    // Run inference
    //for (int i = 0; i < 10; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(context, data, prob, batchSize, inputIndex, outputIndex, buffers, stream);
        auto end = std::chrono::system_clock::now();

        if(prob[0] < prob[1])
            printf("\033[0;31m"); /* Red */
        
        std::cout << image_file << std::endl;
        for(unsigned int i = 0; i < OUTPUT_SIZE; i++) {
            std::cout << prob[i] << ", ";
        }
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        printf("\033[0m"); /* Default color */
    //}
}

void imageFolderInference(IExecutionContext & context, const char *image_folder, const int inputIndex, const int outputIndex, void* buffers[2], cudaStream_t & stream)
{
    DIR *d;
    struct dirent *dir;
    d = opendir(image_folder);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if(strcmp(dir->d_name, ".") == 0 ||
                strcmp(dir->d_name, "..") == 0)
                continue;
            char d_path[512]; // here I am using sprintf which is safer than strcat
            snprintf(d_path, 512, "%s/%s", image_folder, dir->d_name);
            //printf("%s\n", d_path);
            imageInference(context, d_path, inputIndex, outputIndex, buffers, stream);
        }
        closedir(d);
    }
}

int main(int argc, char** argv)
{
    const char *engine_file = 0;
    const char *image_file = 0;
    const char *image_folder = 0;
    const char *optstr = "e:i:d:"; // Option -e, -i, -d, ":" means there must be argument behind
    int o;
    while((o = getopt(argc, argv, optstr)) != -1) {
        switch(o) {
            case 'e': engine_file = strdup(optarg);
                break;
            case 'i': image_file = strdup(optarg);
                break;
            case 'd': image_folder = strdup(optarg);
                break;
        }
    }

    if(engine_file == 0 || (image_file == 0 && image_folder == 0)) {
        std::cout << "Usage : ./resnet50 -e engine [ -i image | -d folder ]" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(engine_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    } else {
        std::cout << "Error open engine file " << engine_file << " !!!" << std::endl;
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    printf("Bindings after deserializing:\n");
    for(int bi = 0; bi < engine->getNbBindings(); bi++) {
        if(engine->bindingIsInput(bi) == true) {
            printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
            printf("%s\n", engine->getBindingFormatDesc(bi));
        } else {
            printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
            printf("%s\n", engine->getBindingFormatDesc(bi));
        }
    }

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    const int batchSize = 1;

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    oneInference(*context, inputIndex, outputIndex, buffers, stream);
    
    if(image_file)
        imageInference(*context, image_file, inputIndex, outputIndex, buffers, stream);

    if(image_folder)
        imageFolderInference(*context, image_folder, inputIndex, outputIndex, buffers, stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
