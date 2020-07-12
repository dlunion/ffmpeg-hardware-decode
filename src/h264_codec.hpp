
#pragma once


#ifndef H264_CODEC_HPP
#define H264_CODEC_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

// reference for cuda_runtime.h  same to cudaStream_t
struct CUstream_st;


namespace H264Codec{

    typedef CUstream_st* CUStream;


    class Encoder{
    public:
        /**
         * write frame
         * 
         * image: 
         *      must be not empty, and CV_8UC3, BGR format
         * 
         * return:
         *      true if write ok, otherwise false
         **/
        virtual bool write(const cv::Mat& image) = 0;
    };


    // bgr format, channels = 3,   bbbgggrrr      float
    struct CudaImage{
        int width = 0, height = 0;
        float* ptr = nullptr;

        /**
         *   Get plane pointer
         * 
         * return:
         *      ptr + width * height * index
         **/
        float* plane(int index) const;

        /**
         *   Get image area
         * 
         * return:
         *      width * height
         **/
        size_t area() const;

        /**
         *   Get image memory bytes count
         * 
         * return:
         *      (width * height) * sizeof(float) * channels,  here channels = 3
         **/
        size_t bytes() const;
    };

    enum CudaNormType : int{
        NoneOfNormType,                 //  nothing to do
        MeanSTD,                    //  out = (x * scale - mean) / std;
        ScaleAdd                    //  out = x * scale + add;
    };

    struct CudaNorm{
        CudaNormType type;

        float mean[3];
        float std[3];
        float scale;
        float add;

        /**
         *      Get CudaNorm, Normialized method: image = (image * scale - mean) / std
         * 
         * return:
         *      CudaNorm instance by MeanSTD
         **/
        static CudaNorm meanStd(float mean[3], float std[3], float scale = 1);


        /**
         *      Get CudaNorm, Normialized method: image = image * scale + add
         * 
         * return:
         *      CudaNorm instance by ScaleAdd
         **/
        static CudaNorm scaleAdd(float scale = 1, float add = 0);


        /**
         *      Get CudaNorm, Normialized method: image = image
         * 
         * return:
         *      CudaNorm instance by NoneOfType
         **/
        static CudaNorm none();
    };

    /**
     * 
     *     Create cuda image instance, and alloc memory
     * 
     * return:
     *      New CudaImage Instance
     **/
    std::shared_ptr<CudaImage> createCudaImage(int width, int height);


    class DecoderInCuda{
    public:
        /**
         * cuda_image: 
         *     If cuda_image is nullptr, the read function will allocate a new return. 
         *     If the memory size of cuda_image does not match the image size, 
         *     reallocate and assign cuda_image
         *     The image is decoded to get yuv data, and then normalized by the norm parameter
         * 
         * raw_image:
         *     If keep_opencv_image is true, copy gpu image to host and convert to raw_image(BGR format, CV_8UC3)
         *     and not normalized
         * 
         * keep_opencv_image:
         *     Tell the decoder if it needs to copy the image to the host(raw_image)
         **/
        virtual bool read(std::shared_ptr<CudaImage>& cuda_image, cv::Mat& raw_image, bool keep_opencv_image = false) = 0;
    };


    class Decoder{
    public:
        /**
         *   Read frame for decoder, image is BGR format CV_8UC3
         * 
         * return:
         *      True if decode got picture, else if failure or end of io.
         **/
        virtual bool read(cv::Mat& image) = 0;
    };


    class IOStream{
    public:

        /**
         *   Write data to stream
         * 
         * data:
         *      Data pointer for write
         * 
         * data:
         *      Data length for write
         **/
        virtual void write(const void* data, size_t size) = 0;

        /**
         *   Set stream end flag.
         **/
        virtual void send() = 0;

        /**
         *   Move the stream cursor to position
         * 
         * offset:
         *      Position's offset
         * 
         * whence:
         *      Special offset begin and means
         *      
         *      value list:
         *      SEEK_SET				cursor = offset,          
         *      SEEK_CUR				cursor = cursor + offset		
         *      SEEK_END				cursor = end of stream
         * 
         * return:
         *      0 if success, otherwise has error(stream not be end, and seek to SEEK_END etc.)
         **/
        virtual int64_t seek(int64_t offset, int whence) = 0;

        /**
         *   Read data from stream, block the invoke if data not ready
         * 
         * data:
         *      Data pointer for write
         * 
         * data:
         *      Data length for write
         * 
         * return:
         *      Fetched data length
         **/
        virtual size_t read(void* data, size_t size) = 0;

        /**
         *   Get stream status, is end of stream
         * 
         * return:
         *      True if stream is ended, otherwise false
         **/
        virtual bool eof() = 0;

        /**
         *   Get stream cursor
         * 
         * return:
         *      That stream cursor
         **/
        virtual int64_t tell() = 0;

        /**
         *   Get stream data pointer
         * 
         * return:
         *      That stream data pointer
         **/
        virtual const void* data() const = 0;

        /**
         *   Get stream data length
         * 
         * return:
         *      That stream data length
         **/
        virtual size_t size() const = 0;

        /**
         *   Get stream data reference
         * 
         * return:
         *      That stream data reference
         **/
        virtual const std::shared_ptr<std::vector<unsigned char>> dataref() const = 0;
    };


    enum SourceType : int{
        NoneOfSourceType,
        File,
        Stream
    };

    class Source{
    public:
        Source();

        /**
         *      Construct source object with file io(SourceFrom::File)
         **/
        Source(const std::string& file);

        /**
         *      Construct source object with stream io(SourceFrom::Stream)
         **/
        Source(const std::shared_ptr<IOStream>& stream);

        /**
         *      Clear the source, and set to NoneOfSourceType
         **/
        void clear();

        /**
         *      Source file path
         **/
        const std::string& file();

        /**
         *      Source stream
         **/
        const std::shared_ptr<IOStream>& stream();

        /**
         *      Source type
         **/
        const SourceType& type();

    private:
        std::shared_ptr<IOStream> stream_;
        std::string file_;
        SourceType type_;
    };

    /**
    *      Create encoder by source
    **/
    std::shared_ptr<Encoder> createEncoder(const Source& source, int width, int height, int fps);

    /**
    *      Create memory stream
    **/
    std::shared_ptr<IOStream> createMemoryStream();

    /**
    *      Create decoder in cuda, decode image has in cuda device memory
    **/
    std::shared_ptr<DecoderInCuda> createDecoderInCuda(const Source& source, const CudaNorm& norm = CudaNorm::none(), int gpuID = 0, CUStream stream = nullptr);

    /**
     *     Create decoder, use cuenc, decode image to host memory
     **/
    std::shared_ptr<Decoder> createDecoder(const Source& source);

    /**
     *     Set cuda device id, used before createMemoryStream
     **/
    bool setDevice(int device_id);


    /**
     *     Set ffmpeg log level,  @reference: libavutil/log.h
     * 
    * Print no output.
    #define AV_LOG_QUIET    -8

    * Something went really wrong and we will crash now.
    #define AV_LOG_PANIC     0

    * Something went wrong and recovery is not possible.
    * For example, no header was found for a format which depends
    * on headers or an illegal combination of parameters is used.
    #define AV_LOG_FATAL     8

    * Something went wrong and cannot losslessly be recovered.
    * However, not all future data is affected.
    #define AV_LOG_ERROR    16

    * Something somehow does not look correct. This may or may not
    * lead to problems. An example would be the use of '-vstrict -2'.
    #define AV_LOG_WARNING  24

    * Standard information.
    #define AV_LOG_INFO     32

    * Detailed information.
    #define AV_LOG_VERBOSE  40

    * Stuff which is only useful for libav* developers.
    #define AV_LOG_DEBUG    48

    * Extremely verbose debugging, useful for libav* development.
    #define AV_LOG_TRACE    56
    **/
    void setLogLevel(int level = -8);
};

#endif //H264_CODEC_HPP