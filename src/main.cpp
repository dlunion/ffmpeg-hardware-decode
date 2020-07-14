

#include "h264_codec.hpp"
#include <thread>

using namespace cv;
using namespace std;


bool save_to_file(const char* file, const void* data, size_t size){

    FILE* f = fopen(file, "wb");
    if(!f) return false;

    auto writed_size = fwrite(data, 1, size, f);
    fclose(f);
    return writed_size == size;
}

void memory_encode_decode(){

    H264Codec::setDevice(0);
    cv::Mat image(480, 640, CV_8UC3);
    auto stream = H264Codec::createMemoryStream();
    auto encoder = H264Codec::createEncoder(stream, 640, 480, 1);

    for(int i = 0; i < 100; ++i){

        image.setTo(0);
        putText(image, format("%d", i), Point(320, 240), 0, 3, Scalar(0, 255), 2, 16);

        encoder->write(image);
        printf("encode: %d\n", i);
    }

    // write trailer and free
    encoder.reset();

    // save to file
    save_to_file("result.mp4", stream->data(), stream->size());


    ////////////////  decode  test
    auto decoder = H264Codec::createDecoderInCuda(stream);
    std::shared_ptr<H264Codec::CudaImage> cuimage;
    int iframe = 0;

    while(decoder->read(cuimage, image, true)){
        printf("decode: %d\n", iframe++);
    }

    imwrite("last.1.image.jpg", image);
}


void file_encode_decode(){

    H264Codec::setDevice(0);
    cv::Mat image(480, 640, CV_8UC3);
    string stream = "encode.mp4";
    auto encoder = H264Codec::createEncoder(stream, 640, 480, 1);

    for(int i = 0; i < 100; ++i){

        image.setTo(0);
        putText(image, format("%d", i), Point(320, 240), 0, 3, Scalar(0, 255), 2, 16);

        encoder->write(image);
        printf("encode: %d\n", i);
    }

    // write trailer and free
    encoder.reset();


    ////////////////  decode  test
    auto decoder = H264Codec::createDecoderInCuda(stream);
    std::shared_ptr<H264Codec::CudaImage> cuimage;
    int iframe = 0;

    while(decoder->read(cuimage, image, true)){
        printf("decode: %d\n", iframe++);
    }

    imwrite("last.2.image.jpg", image);
}

// read part , decode part
void stream_decode(){

    auto stream = H264Codec::createMemoryStream();
    thread(
        [&]{
            FILE* f = fopen("encode.mp4", "rb");
            char buf[1024 * 5];

            fseek(f, 0, SEEK_END);
            int total = ftell(f);
            fseek(f, 0, SEEK_SET);

            while(!feof(f)){

                printf("File position: %.2f KB, total: %.2f KB\n", ftell(f) / 1024.0f, total / 1024.0f);

                int rlen = fread(buf, 1, sizeof(buf), f);
                if(rlen > 0)
                    stream->write(buf, rlen);

                this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            fclose(f);

            stream->send();
            printf("thread done.\n");
        }
    ).detach();

    auto decoder = H264Codec::createDecoderInCuda(stream);
    std::shared_ptr<H264Codec::CudaImage> cuimage;
    int iframe = 0;
    Mat image;

    while(decoder->read(cuimage, image, true)){
        printf("decode: %d\n", iframe++);
    }

    imwrite("last.3.image.jpg", image);
}

int main(){

    H264Codec::setLogLevel();
    memory_encode_decode();
    file_encode_decode();
    stream_decode();

    printf("program done.\n");
    return 0;
}