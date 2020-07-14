

#include "h264_codec.hpp"
#include <cuda_runtime.h>
#include <condition_variable>
#include <mutex>
#include <vector>

extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/opt.h>
};



// change mov.c line: 6876 ->  
//  if(c->found_moov){
//     if (!(pb->seekable & AVIO_SEEKABLE_NORMAL) || c->fc->flags & AVFMT_FLAG_IGNIDX || c->frag_index.complete)
//         c->next_root_atom = start_pos + a.size;
//     c->atom_depth --;
//     return 0;
//  }
namespace H264Codec{

    using namespace std;
    using namespace cv;

    class MemoryStream : public IOStream{
    public:
        MemoryStream(){
            is_ref_datas_ = false;
            block_if_no_data_ = false;

            datas_.reset(new vector<uchar>());
            lck_.reset(new mutex());
            eof_.reset(new bool(false));
            cv_.reset(new condition_variable());
        }

        MemoryStream(IOStream* ref, bool block_if_no_data=true){
            is_ref_datas_ = true;
            block_if_no_data_ = block_if_no_data;

            auto mem = static_cast<MemoryStream*>(ref);
            datas_ = mem->datas_;
            lck_ = mem->lck_;
            eof_ = mem->eof_;
            cv_ = mem->cv_;
            cursor_ = 0;
        }

        virtual ~MemoryStream(){

            std::unique_lock<mutex> l(*lck_);
            *eof_ = true;
            cv_->notify_all();
        }

        virtual void write(const void* data, size_t size) override{

            if(is_ref_datas_){
                // is ref read
                fprintf(stderr, "write stream is ref on read only.\n");
                return;
            }

            std::unique_lock<mutex> l(*lck_);

            if(cursor_ == datas_->size())
                // if append as end
                datas_->insert(datas_->end(), (uchar*)data, (uchar*)data + size);
            else{
                size_t total = cursor_ + size;
                if(total > datas_->size())
                    datas_->insert(datas_->end(), total - datas_->size(), 0);
                
                memcpy(datas_->data() + cursor_, data, size);
            }

            cursor_ += size;
            cv_->notify_all();
        }

        virtual void send() override{

            if(is_ref_datas_){
                // is ref read
                fprintf(stderr, "write stream is ref on read only.\n");
                return;
            }

            std::unique_lock<mutex> l(*lck_);
            *eof_ = true;
            cv_->notify_all();
        }

        // 0 if status is success, else status = 1
        // SEEK_SET				set position = offset,          
        // SEEK_CUR				set position = offset + cursor		
        // SEEK_END				no support
        virtual int64_t seek(int64_t offset, int whence) override{

            std::unique_lock<mutex> l(*lck_);
            const static int64_t ret_success = 0;
            const static int64_t ret_failure = 1;

            int64_t position = 0;
            if(whence == SEEK_SET){
                position = offset;
            }else if(whence == SEEK_CUR){
                position = offset + cursor_;
            }else if(whence == SEEK_END){
                if(eof_){
                    position = (int64_t)datas_->size() + offset;
                }else{
                    // important, return failure if not eof  
                    return ret_failure;
                }
            }else{
                return ret_failure;
            }

            if(position > datas_->size()){
                position = datas_->size();
            }else if(position < 0){
                position = 0;
            }
            
            cursor_ = position;
            return ret_success;
        }

        const void* data() const override{
            return this->datas_->data();
        }

        size_t size() const override{
            return this->datas_->size();
        }

        // block if data not ready
        virtual size_t read(void* data, size_t size) override{

            std::unique_lock<mutex> l(*lck_);
            cv_->wait(l, [&]{return cursor_ < datas_->size() || *eof_ || !block_if_no_data_;});

            int64_t readlen = min((int64_t)size, (int64_t)datas_->size() - cursor_);
            if(readlen <= 0)
                return 0;

            memcpy(data, datas_->data() + cursor_, readlen);
            cursor_ += readlen;
            return readlen;
        }

        virtual bool eof() override{

            std::unique_lock<mutex> l((mutex&)*lck_);
            if(cursor_ < datas_->size())
                return false;

            return *eof_;
        }

        virtual int64_t tell() override{
            return cursor_;
        }

    private:
        int64_t cursor_ = 0;
        shared_ptr<bool> eof_;
        shared_ptr<vector<uchar>> datas_;
        shared_ptr<condition_variable> cv_;
        shared_ptr<mutex> lck_;
        bool is_ref_datas_ = false;
        bool block_if_no_data_ = true;
    };



    static void free_encode_context(AVCodecContext* codecCtx){

        if(!codecCtx) return;
        avcodec_free_context(&codecCtx);
    }

    static void close_codec(AVCodecContext* codec){
        if(!codec) return;
        avcodec_close(codec);
    }

    static void encoder_io_free_format_context(AVFormatContext* fmtctx){
        
        if(!fmtctx) return;

        if(fmtctx->pb){
            av_write_trailer(fmtctx);
        }

        //clean context,   auto free fmtctx->pb
        avformat_close_input(&fmtctx);
    }

    static void encoder_stream_free_format_context(AVFormatContext* fmtctx){
        
        if(!fmtctx) return;

        if(fmtctx->pb){
            av_write_trailer(fmtctx);

            av_freep(&fmtctx->pb->buffer);
            avio_context_free(&fmtctx->pb);
        }

        //clean context,   custom io, avformat_close_input do not free that
        avformat_free_context(fmtctx);
    }

    static void decoder_stream_free_format_context(AVFormatContext* fmtctx){

        if(!fmtctx) return;

        if(fmtctx->pb){
            av_freep(&fmtctx->pb->buffer);
            avio_context_free(&fmtctx->pb);
        }
        avformat_free_context(fmtctx);
    }

    static void decoder_io_free_format_context(AVFormatContext* fmtctx){

        if(!fmtctx) return;
        avformat_close_input(&fmtctx);
    }

    static void free_frame(AVFrame* frame){

        if(!frame) return;
        av_frame_free(&frame);
    }

    static void free_packet(AVPacket* packet){
        if(!packet) return;

        av_free_packet(packet);
        delete packet;
    }

    static void free_cuda_image(CudaImage* image){
        if(!image) return;

        if(image->ptr){
            cudaFree(image->ptr);
            image->ptr = nullptr;
        }
        delete image;
    }

    static void free_avio_context(AVIOContext* ptr){
        if(!ptr) return;

        /* note: the internal buffer could have changed, and be != avio_ctx_buffer */
        //if(ptr->buffer) av_freep(&ptr->buffer);
        //avio_context_free(&ptr);
        avio_close(ptr);
    }

    static void init_codec(){

        static volatile bool inited = false;

        if(!inited){
            inited = true;
            av_register_all();
            avformat_network_init();
            avcodec_register_all(); 
        }
    }

    static int local_read_packet_ios(void *opaque, uint8_t *buf, int buf_size){

	    IOStream* ios = (IOStream*)opaque;
        if(!ios->eof()){
            int len = ios->read(buf, buf_size);
            if(len == 0 && ios->eof())
                return AVERROR_EOF;
            return len;
        }
        return AVERROR_EOF;
    }

    static int local_write_packet_ios(void *opaque, uint8_t *buf, int buf_size){

	    IOStream* ios = (IOStream*)opaque;
        ios->write(buf, buf_size);
        return 0;
    }

    static int64_t local_seek_ios(void *opaque, int64_t offset, int whence){

        IOStream* ios = (IOStream*)opaque;
        if(ios->seek(offset, whence) == 0)
            return ios->tell();
        
        return -1;
    }

    static int local_open(struct AVFormatContext *s, AVIOContext **pb, const char *url, int flags, AVDictionary **options){
        
        size_t buffer_size = 4096;
        unsigned char* stream_read_buffer = (unsigned char*)av_malloc(buffer_size);

        // reference for s->pb->opaque stream
        MemoryStream* stream_ptr = new MemoryStream((IOStream*)s->pb->opaque, false);
        *pb = avio_alloc_context(stream_read_buffer, buffer_size, 0, 
            stream_ptr, &local_read_packet_ios, nullptr, &local_seek_ios);
        return 0;
    }

    static void local_close(struct AVFormatContext *s, AVIOContext *pb){

        if(!pb) return;

        MemoryStream* stream_ptr = (MemoryStream*)pb->opaque;
        if(stream_ptr)
            delete stream_ptr;

        pb->opaque = nullptr;
        av_freep(&pb->buffer);
        avio_context_free(&pb);
    }

    static bool string_begin_with(const string& str, const string& with){

        if(str.size() < with.size()) return false;
        if(with.empty()) return true;

        return memcmp(str.c_str(), with.c_str(), with.size()) == 0;
    }


    class EncoderImpl : public Encoder{

    public:
        bool open(const Source& source, int width, int height, int fps){

            close();

            pkt_.reset(new AVPacket(), free_packet);
            av_init_packet(pkt_.get());

            this->source_ = source;
            this->width_ = width;
            this->height_ = height;
            this->fps_ = fps;
            this->has_nvenc_ = true;

            encoder_ = avcodec_find_encoder_by_name("h264_nvenc");

            if(!encoder_){
                this->has_nvenc_ = false;
                encoder_ = avcodec_find_encoder(AV_CODEC_ID_H264);
            }

            if (!encoder_){
                fprintf(stderr, "avcodec_find_encoder h264 failed!\n");
                return false;
            }

            //get encoder contex
            encodeCtx_.reset(avcodec_alloc_context3(encoder_), free_encode_context);
            if (!encodeCtx_){
                fprintf(stderr, "avcodec_alloc_context3 for encoder contx failed!\n");
                return false;
            }

            //set encoder params
            //bit rate
            encodeCtx_->bit_rate = 1280 * 1000;  //1280k
            
            encodeCtx_->width = width_;
            encodeCtx_->height = height_;
            encodeCtx_->time_base = {1, fps_};
            
            //set gop size, in another way I frame gap
            //mEncCtx->gop_size = 50;
            encodeCtx_->gop_size = 1000;
            
            encodeCtx_->max_b_frames = 0;
            encodeCtx_->pix_fmt = AV_PIX_FMT_YUV420P;
            encodeCtx_->codec_id = AV_CODEC_ID_H264;
            encodeCtx_->thread_count = 4;
            encodeCtx_->qmin = 10;
            encodeCtx_->qmax = 51;
            encodeCtx_->qcompress  = 0.6;

            if(!this->has_nvenc_){
                // speed up
                av_opt_set(encodeCtx_->priv_data, "preset", "veryfast", 0);
            }

            av_opt_set(encodeCtx_->priv_data, "tune", "zerolatency", 0);
            
            //global header info
            encodeCtx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            
            //open encoder
            int ret = avcodec_open2(encodeCtx_.get(), encoder_, nullptr);
            if (ret < 0){
                fprintf(stderr, "avcodec_open2 encoder failed!\n");
                return false;
            }
            
            //2 create out context
            AVFormatContext* format_context = nullptr;
            if(source_.type() == SourceType::File){
                avformat_alloc_output_context2(&format_context, 0, 0, source_.file().c_str());
                formatCtx_.reset(format_context, encoder_io_free_format_context);
            }else if(source_.type() == SourceType::Stream){
                avformat_alloc_output_context2(&format_context, 0, 0, ".mp4");
                formatCtx_.reset(format_context, encoder_stream_free_format_context);
            }else{
                fprintf(stderr, "unsupport dest: %d\n", source_.type());
                return false;
            }
            
            //3 add video stream
            outstream_ = avformat_new_stream(formatCtx_.get(), nullptr);
            outstream_->id = 0;
            outstream_->codecpar->codec_tag = 0;
            avcodec_parameters_from_context(outstream_->codecpar, encodeCtx_.get());
            
            if(source_.type() == SourceType::File){
                ret = avio_open(&formatCtx_->pb, source_.file().c_str(), AVIO_FLAG_WRITE);
                if (ret < 0){
                    fprintf(stderr, "avio_open  failed!\n");
                    return false;
                }

            }else if(source_.type() == SourceType::Stream){

                size_t buffer_size = 4096;
                unsigned char* stream_read_buffer = (unsigned char*)av_malloc(buffer_size);
                auto pb = avio_alloc_context(stream_read_buffer, buffer_size, 1, 
                    source_.stream().get(), nullptr, &local_write_packet_ios, &local_seek_ios);

                formatCtx_->pb = pb;        // alloc with close input
                formatCtx_->flags |= AVFMT_FLAG_CUSTOM_IO;
                formatCtx_->io_open = local_open;
                formatCtx_->io_close = local_close;
            }

            // set movflags options
            AVDictionary* options = nullptr;
            av_dict_set(&options, "movflags", "faststart", 0);

            ret = avformat_write_header(formatCtx_.get(), &options);
            av_dict_free(&options);

            if (ret < 0){
                fprintf(stderr, "avformat_write_header failed!\n");
                return false;
            }

            success_ = true;
            return true;
        }

        void close(){

            if(success_)
                flush_remain();

            success_ = false;
            width_ = height_ = fps_ = pts_ = 0;
            encoder_ = nullptr;
            outstream_ = nullptr;

            formatCtx_.reset();
            encodeCtx_.reset();
            yuvframe_.reset();
            pkt_.reset();

            if(source_.type() == SourceType::Stream){
                source_.stream()->send();
                source_.stream()->seek(0, SEEK_SET);
            }
            source_.clear();
        }

        void flush_remain(){

            int ret = 0;
            while(ret >= 0){

                //flush out rest frames, send nullptr frame data
                avcodec_send_frame(encodeCtx_.get(), nullptr);
                //receive frame from encoder
                ret = avcodec_receive_packet(encodeCtx_.get(), pkt_.get());
                if (ret != 0)
                    break;

                //wirte encoded frame to file
                av_interleaved_write_frame(formatCtx_.get(), pkt_.get());
                av_packet_unref(pkt_.get());
            }
        }

        bool write_420p(const uint8_t* y, size_t ysize, const uint8_t* u, const uint8_t* v, size_t uvsize, int64_t pts = -1){

            if(!yuvframe_){
                //alloc output yuv frame
                yuvframe_.reset(av_frame_alloc(), free_frame);
                yuvframe_->format = AV_PIX_FMT_YUV420P;
                yuvframe_->width = width_;
                yuvframe_->height = height_;

                //alloc frame buffer
                int ret = av_frame_get_buffer(yuvframe_.get(), 32);
                if (ret < 0){
                    fprintf(stderr, "av_frame_get_buffer failed!\n");
                    return false;
                }
            }

            memcpy(yuvframe_->data[0], y, ysize);
            memcpy(yuvframe_->data[1], u, uvsize);
            memcpy(yuvframe_->data[2], v, uvsize);

            if(pts == -1){
                pts = pts_;

                const int RFC_HZ = 90000;
                pts_ += RFC_HZ / fps_;
                //pts_ ++;
            }

            yuvframe_->pts = pts;
            
            //send to encoder
            int ret = avcodec_send_frame(encodeCtx_.get(), yuvframe_.get());
            if (ret != 0){
                //fprintf(stderr, "encoder send frame %d\n", yuvframe_->pts);
                return true;
            }

            ret = avcodec_receive_packet(encodeCtx_.get(), pkt_.get());
            if (ret != 0){
                //fprintf(stderr, "encoder recieve frame %d err\n", yuvframe_->pts);
                return true;
            }

            //write encoded frame to file
            av_interleaved_write_frame(formatCtx_.get(), pkt_.get());
            av_packet_unref(pkt_.get());
            return true;
        }

        virtual bool write(const cv::Mat& image) override{

            if(image.empty())
                return false;

            cvtColor(image, yuv_i420_, cv::COLOR_BGR2YUV_I420);

            int area = image.size().area();
            return write_420p(
                yuv_i420_.ptr<uchar>(0), area, 
                yuv_i420_.ptr<uchar>(image.rows),
                yuv_i420_.ptr<uchar>(image.rows * 1.25), 
                area / 4);
        }

        virtual ~EncoderImpl(){
            close();
        }

    private:
        Source source_;
        int width_, height_;
        int fps_ = 0;
        int64_t pts_ = 0;
        AVCodec* encoder_ = nullptr;
        shared_ptr<AVCodecContext> encodeCtx_;
        shared_ptr<AVFormatContext> formatCtx_;
        AVStream* outstream_ = nullptr;
        shared_ptr<AVFrame> yuvframe_;
        Mat yuv_i420_;
        shared_ptr<AVPacket> pkt_;
        bool success_ = false;
        bool has_nvenc_ = false;
    };

    std::shared_ptr<Encoder> createEncoder(const Source& file, int width, int height, int fps){

        init_codec();
        shared_ptr<EncoderImpl> impl(new EncoderImpl());

        if(!impl->open(file, width, height, fps))
            impl.reset();
        return impl;
    }








    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    CudaNorm CudaNorm::meanStd(float mean[3], float std[3], float scale){
        CudaNorm norm;
        norm.type = CudaNormType::MeanSTD;
        memcpy(norm.mean, mean, sizeof(float) * 3);
        memcpy(norm.std, std, sizeof(float) * 3);
        norm.scale = scale;
        return norm;
    }

    CudaNorm CudaNorm::scaleAdd(float scale, float add){
        CudaNorm norm;
        norm.type = CudaNormType::ScaleAdd;
        norm.scale = scale;
        norm.add = add;
        return norm;
    } 
    
    CudaNorm CudaNorm::none(){
        CudaNorm norm;
        norm.type = CudaNormType::NoneOfNormType;
        return norm;
    }

    size_t CudaImage::area() const{
        return width * height;
    }

    float* CudaImage::plane(int index) const{
        if(index < 0 || index >= 3) return nullptr;
        return ptr + width * height * index;
    }

    size_t CudaImage::bytes() const{
        return width * height * 3 * sizeof(float);
    }

    std::shared_ptr<CudaImage> createCudaImage(int width, int height){
        std::shared_ptr<CudaImage> image(new CudaImage(), free_cuda_image);
        image->width = width;
        image->height = height;

        auto status = cudaMalloc(&image->ptr, image->bytes());
        if(status != cudaSuccess){
            fprintf(stderr, "cudaMalloc %d bytes fail: %s\n", image->bytes(), cudaGetErrorString(status));
            image.reset();
        }
        return image;
    }

    enum DecodeTo : int{
        Cuda = 0,
        OpenCV = 1
    };

    void convert_nv12_to_bgr_float(const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, const CudaNorm& norm, float* dst, cudaStream_t stream);

    class CudaDecoderImpl : public DecoderInCuda, public Decoder{

    public:
        CudaDecoderImpl(){
        }

        virtual ~CudaDecoderImpl(){
            close();
        }

        bool open(const Source& source, DecodeTo to){
            
            close();

            this->to_ = to;
            this->source_ = source;
            
            auto format_ctx_ptr = avformat_alloc_context();
            if(source_.type() == SourceType::File){

                AVDictionary* options = nullptr;
                if (string_begin_with(source_.file(), "rtsp")) 
                    av_dict_set(&options, "rtsp_transport", "tcp", 0);

                int ret = avformat_open_input(&format_ctx_ptr, source_.file().c_str(), nullptr, &options);
                format_ctx_.reset(format_ctx_ptr, decoder_io_free_format_context);

                if (ret != 0) {
                    fprintf(stderr, "open %s fail.\n", source_.file().c_str());
                    return false;
                }
            }else if(source_.type() == SourceType::Stream){

                // reference other stream for read
                source_ = Source(make_shared<MemoryStream>(source_.stream().get()));

                size_t buffer_size = 4096;
                unsigned char* stream_read_buffer = (unsigned char*)av_malloc(buffer_size);
                auto pb = avio_alloc_context(stream_read_buffer, buffer_size, 0, 
                    source_.stream().get(), &local_read_packet_ios, nullptr, &local_seek_ios);

                format_ctx_ptr->pb = pb;
                format_ctx_ptr->flags |= AVFMT_FLAG_CUSTOM_IO;
                format_ctx_.reset(format_ctx_ptr, decoder_stream_free_format_context);

                if (avformat_open_input(&format_ctx_ptr, nullptr, nullptr, nullptr) != 0) {
                    fprintf(stderr, "open stream fail.\n");
                    return false;
                }
            }else{
                fprintf(stderr, "unsupport source: %d\n", source_.type());
                return false;
            }
        
            if (avformat_find_stream_info(format_ctx_.get(), nullptr) < 0) {
                fprintf(stderr, "Could't find stream infomation\n");
                return false;
            }
        
            for (int i = 0; i < format_ctx_->nb_streams; i++) {
                if (format_ctx_->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) 
                    video_stream_ = i;
            }
        
            if (video_stream_ == -1) {
                fprintf(stderr, "no find vedio_stream\n");
                return false;
            }
        
            codec_ctx_.reset(format_ctx_->streams[video_stream_]->codec, close_codec);

            //codec_ = avcodec_find_decoder(codec_ctx_->codec_id);
            codec_ = avcodec_find_decoder_by_name("h264_cuvid");
            if (codec_ == nullptr) {
                fprintf(stderr, "not found decodec.\n");
                return false;
            }

            AVDictionary *opts = nullptr;
            if(this->to_ == DecodeTo::Cuda){

                char number_buffer[64];
                sprintf(number_buffer, "%d", gpuid_);
                av_dict_set(&opts, "gpu", number_buffer, 0);
            }

            if (avcodec_open2(codec_ctx_.get(), codec_, &opts) < 0) {
                fprintf(stderr, "Could not open decodec.\n");
                return false;
            }

            frame_.reset(av_frame_alloc(), free_frame);
            int plane_size = codec_ctx_->width * codec_ctx_->height;
            
            pkt_.reset(new AVPacket(), free_packet);
            //av_new_packet(pkt_.get(), plane_size);
            av_init_packet(pkt_.get());

            //av_dump_format(format_ctx_.get(), 0, file.c_str(), 0);
            //fps_ = format_ctx_->streams[video_stream_]->avg_frame_rate.num / format_ctx_->streams[video_stream_]->avg_frame_rate.den;
            int num = format_ctx_->streams[video_stream_]->avg_frame_rate.num;
            int den = format_ctx_->streams[video_stream_]->avg_frame_rate.den;

            if(den != 0)
                fps_ = num / den;
            else
                fps_ = 0;

            if(this->to_ == DecodeTo::Cuda){
                cuda_image_ = createCudaImage(codec_ctx_->width, codec_ctx_->height);
                if(!cuda_image_)
                    return false;
            }
            return true;
        }

        // 0  error,  1  finish,  2  empty 
        int decode_host_image(AVPacket* pkt, cv::Mat& image){

            if(skipped_frame_mode_){
                pkt->data = nullptr;
                pkt->size = 0;
            }

            int got_picture = 0;
            int ret = avcodec_decode_video2(codec_ctx_.get(), frame_.get(), &got_picture, pkt);

            if (ret < 0) 
                return 0;

            bool finish = false;
            if (got_picture) {
                Size s(codec_ctx_->width, codec_ctx_->height);
                int line_size = frame_->linesize[0];

                if(yuv_nv12_cache_.rows != s.height * 1.5 || yuv_nv12_cache_.cols != line_size)
                    yuv_nv12_cache_ = Mat(s.height * 1.5, line_size, CV_8U, line_size);

                memcpy(yuv_nv12_cache_.ptr<uchar>(0), frame_->data[0], s.height * line_size);
                memcpy(yuv_nv12_cache_.ptr<uchar>(s.height), frame_->data[1], s.height / 2 * line_size);

                // cropped it if line_size != s.width.
                cv::cvtColor(yuv_nv12_cache_(cv::Rect(0, 0, s.width, yuv_nv12_cache_.rows)), image, cv::COLOR_YUV2BGR_NV12);
                finish = true;
            }else{
                if(!skipped_frame_mode_)
                    count_skipped_frame_++;
            }
            av_free_packet(pkt);
            return finish ? 1 : 2;
        }


        virtual bool read(cv::Mat& image) override {

            if(this->to_ != DecodeTo::OpenCV){
                fprintf(stderr, "decode to is not cuda.\n");
                return false;
            }
            
            if(!skipped_frame_mode_){

                int index = 0;
                while (av_read_frame(format_ctx_.get(), pkt_.get()) >=0){
                    if (pkt_->stream_index == video_stream_){

                        // 0  error,  1  finish,  2  empty 
                        int code = decode_host_image(pkt_.get(), image);
                        if(code == 0)
                            return false;

                        if(code == 1)
                            return true;
                    }
                    av_packet_unref(pkt_.get());
                }
                skipped_frame_mode_ = true;
            }

            if(skipped_frame_mode_){
                while(count_skipped_frame_-- > 0){
                    int code = decode_host_image(pkt_.get(), image);
                    if(code == 0)
                        return false;

                    if(code == 1)
                        return true;
                }
            }
            return false;
        }


        // 0  error,  1  finish,  2  empty 
        int decode_cuda_image(AVPacket* pkt, std::shared_ptr<CudaImage>& cuda_image, cv::Mat& raw_image, bool keep_opencv_image){

            codec_ctx_->pix_fmt = AV_PIX_FMT_CUDA;

            if(skipped_frame_mode_){
                pkt->data = nullptr;
                pkt->size = 0;
            }

            int got_picture = 0;
            int ret = avcodec_decode_video2(codec_ctx_.get(), frame_.get(), &got_picture, pkt);

            if (ret < 0) 
                return 0;

            bool finish = false;
            if (got_picture) {
                Size s(codec_ctx_->width, codec_ctx_->height);
                int line_size = frame_->linesize[0];

                if(keep_opencv_image){
                    if(yuv_nv12_cache_.rows != s.height * 1.5 || yuv_nv12_cache_.cols != line_size)
                        yuv_nv12_cache_ = Mat(s.height * 1.5, line_size, CV_8U, line_size);

                    cudaMemcpy(yuv_nv12_cache_.ptr<uchar>(0), frame_->data[0], s.height * line_size, cudaMemcpyDeviceToHost);
                    cudaMemcpy(yuv_nv12_cache_.ptr<uchar>(s.height), frame_->data[1], s.height / 2 * line_size, cudaMemcpyDeviceToHost);

                    // cropped it if line_size != s.width.
                    cv::cvtColor(yuv_nv12_cache_(cv::Rect(0, 0, s.width, yuv_nv12_cache_.rows)), raw_image, cv::COLOR_YUV2BGR_NV12);
                }
    
                if(cuda_image == nullptr || 
                    cuda_image->width * cuda_image->height != 
                    cuda_image_->width * cuda_image_->height)
                {
                    cuda_image = cuda_image_;
                }
                
                convert_nv12_to_bgr_float(
                    frame_->data[0], frame_->data[1], codec_ctx_->width, 
                    codec_ctx_->height, line_size, norm_, cuda_image->ptr, stream_);
                
                finish = true;
            }else{
                if(!skipped_frame_mode_)
                    count_skipped_frame_++;
            }
            av_packet_unref(pkt);
            return finish ? 1 : 2;
        }

        virtual bool read(std::shared_ptr<CudaImage>& cuda_image, cv::Mat& raw_image, bool keep_opencv_image) override {

            if(this->to_ != DecodeTo::Cuda){
                fprintf(stderr, "decode to is not cuda.\n");
                return false;
            }
            
            if(!skipped_frame_mode_){

                int index = 0;
                while (av_read_frame(format_ctx_.get(), pkt_.get()) >=0){
                    if (pkt_->stream_index == video_stream_){

                        // 0  error,  1  finish,  2  empty 
                        int code = decode_cuda_image(pkt_.get(), cuda_image, raw_image, keep_opencv_image);
                        if(code == 0)
                            return false;

                        if(code == 1)
                            return true;
                    }
                    av_packet_unref(pkt_.get());
                }
                skipped_frame_mode_ = true;
            }

            if(skipped_frame_mode_){
                while(count_skipped_frame_-- > 0){
                    int code = decode_cuda_image(pkt_.get(), cuda_image, raw_image, keep_opencv_image);
                    if(code == 0)
                        return false;

                    if(code == 1)
                        return true;
                }
            }
            return false;
        }

        void setGPUID(int gpuid){
            gpuid_ = gpuid;
        }

        void setNorm(const CudaNorm& norm){
            norm_ = norm;
        }

        void setStream(CUStream stream){
            stream_ = stream;
        }

        void close(){
            
            source_.clear();
            codec_ = nullptr;
            skipped_frame_mode_ = false;
            count_skipped_frame_ = 0;
            video_stream_ = -1;
            fps_ = 0;

            pkt_.reset();
            cuda_image_.reset();
            frame_.reset();
            codec_ctx_.reset();
            format_ctx_.reset();
        }

    private:
        Source source_;
        shared_ptr<AVFormatContext> format_ctx_;
        shared_ptr<AVCodecContext> codec_ctx_;
        shared_ptr<AVFrame> frame_;
        shared_ptr<AVPacket> pkt_;
        shared_ptr<CudaImage> cuda_image_;
        AVCodec* codec_ = nullptr;
        int video_stream_ = -1;
        int fps_ = 0;
        DecodeTo to_;
        CudaNorm norm_ = CudaNorm::none();
        Mat yuv_nv12_cache_;
        CUStream stream_ = nullptr;
        int gpuid_ = 0;
        bool skipped_frame_mode_ = false;
        int count_skipped_frame_ = 0;
    };


    std::shared_ptr<DecoderInCuda> createDecoderInCuda(const Source& file, const CudaNorm& norm, int gpuID, CUStream stream){

        std::shared_ptr<CudaDecoderImpl> impl(new CudaDecoderImpl());
        impl->setNorm(norm);
        impl->setStream(stream);
        impl->setGPUID(gpuID);

        if(!impl->open(file, DecodeTo::Cuda))
            impl.reset();
        return impl;
    }

    std::shared_ptr<Decoder> createDecoder(const Source& file){
        std::shared_ptr<CudaDecoderImpl> impl(new CudaDecoderImpl());
        if(!impl->open(file, DecodeTo::OpenCV))
            impl.reset();
        return impl;
    };

    Source::Source(){
        this->type_ = SourceType::NoneOfSourceType;
    }

    Source::Source(const std::string& file){
        this->file_ = file;
        this->type_ = SourceType::File;
    }

    Source::Source(const std::shared_ptr<IOStream>& stream){
        this->stream_ = stream;
        this->type_ = SourceType::Stream;
    }

    void Source::clear(){
        this->file_ = "";
        this->stream_.reset();
        this->type_ = SourceType::NoneOfSourceType;
    }

    const std::string& Source::file(){
        return this->file_;
    }

    const std::shared_ptr<IOStream>& Source::stream(){
        return this->stream_;
    }

    const SourceType& Source::type(){
        return this->type_;
    }

    std::shared_ptr<IOStream> createMemoryStream(){
        return std::shared_ptr<IOStream>(new MemoryStream());
    }

    bool setDevice(int device_id){
        return cudaSetDevice(device_id) == cudaSuccess;
    }

    void setLogLevel(int level){
        av_log_set_level(level);
    }
};
