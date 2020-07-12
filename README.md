# ffmpeg-hardware-decode
Use ffmpeg and NVCodec to hardware decode video or video stream.   in DNN inference system



##### Decoder

​		video.mp4   ->   cuvid(GPU Decoder)   ->   GPU Memory  ->   Normialize Cuda Kernel   ->   float*  Channel  based   ->   DNN Inference



Input Format:  			  Supported File/USB Camera/RTSP/RTMP/CustomStream etc.

Normialize Kernel:   	Subtract mean and divide std, convert YUV to BGR or RGB



##### Encoder

   	Camera/Other Source   ->   nvenc(GPU Encoder)   ->   Custom Memory Stream/File



Output Format:   		Only supported mp4



## Environment

* ffmpeg 4.2
* nasm 2.14.02
* nv-codec-headers
* opencv 4.2.0 [optional]

* nvcodec-VideoCodecSDK 10.0.26
* cuda 10.2
* x264-snapshot-20190704-2245-stable
* GPU 2080Ti [optional]

Or download all lean code [ffmpeg.hw.lean.code.tar.gz](http://zifuture.com:1000/fs/16.std/ffmpeg.hw.lean.code.tar.gz) 





## Startup

* note:   /datav/newbb/lean       					is root directory
  * ​     /datav/newbb/lean/build         	   lean code directory
  * ​     /datav/newbb/lean/lean                 lean build result binary directory

Change the directory in *_build file, to your path

```bash
> mkdir build && mkdir lean
> cd build
> tar -zxvf ../ffmpeg.hw.lean.code.tar.gz
> mv Video_Codec_SDK_10.0.26 ../lean/
```



* **Change  /datav/newbb/lean  to your path**, after to run this code

```bash
> bash nasm_build
> bash nvcodec_build
> bash x264_build
> bash ffmpeg_build
> bash opencv_build
```



### Run examples

```bash
make run -j8
```

