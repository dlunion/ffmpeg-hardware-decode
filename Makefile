
CC := @g++
CUCC := nvcc
ECHO := @echo
SRCDIR := src
OBJDIR := objs
BINDIR := ./workspace
LEAN := /datav/newbb/lean

# -gencode=arch=compute_60,code=sm_60
OUTNAME := ffmpeg_hw_decode
CFLAGS := -std=c++11 -m64 -fPIC -g -fopenmp -w -O3 -pthread
CUFLAGS := -std=c++11 -m64 -Xcompiler -fPIC -g -w -O3 -gencode=arch=compute_75,code=sm_75
INC_OPENCV := $(LEAN)/opencv4.2.0/include/opencv4 $(LEAN)/opencv4.2.0/include/opencv4/opencv $(LEAN)/opencv4.2.0/include/opencv4/opencv2
INC_FFMPEG := $(LEAN)/ffmpeg4.2/include
INC_CUDA := $(LEAN)/cuda10.2/include 
INCS := $(INC_OPENCV) $(INC_FFMPEG) $(INC_CUDA)
INCS := $(foreach inc, $(INCS), -I$(inc))

LIB_CUDA := $(LEAN)/cuda10.2/lib
LIB_FFMPEG := $(LEAN)/ffmpeg4.2/lib
LIB_OPENCV := $(LEAN)/opencv4.2.0/lib
LIBS := $(LIB_OPENCV) $(LIB_FFMPEG) $(LIB_CUDA)
RPATH := $(foreach lib, $(LIBS),-Wl,-rpath=$(lib))
LIBS := $(foreach lib, $(LIBS),-L$(lib))

LD_OPENCV := opencv_core opencv_highgui opencv_imgproc opencv_video opencv_videoio opencv_imgcodecs
LD_CUDA := cuda cudart curand
LD_FFMPEG := avcodec avformat avresample swscale avutil
LD_SYS := dl stdc++ pthread
LDS := $(LD_OPENCV) $(LD_FFMPEG) $(LD_CUDA) $(LD_SYS)
LDS := $(foreach lib, $(LDS), -l$(lib))

SRCS := $(shell cd $(SRCDIR) && find -name "*.cpp")
OBJS := $(patsubst %.cpp,%.o,$(SRCS))
OBJS := $(foreach item,$(OBJS),$(OBJDIR)/$(item))
CUS := $(shell cd $(SRCDIR) && find -name "*.cu")
CUOBJS := $(patsubst %.cu,%.o,$(CUS))
CUOBJS := $(foreach item,$(CUOBJS),$(OBJDIR)/$(item))
CS := $(shell cd $(SRCDIR) && find -name "*.c")
COBJS := $(patsubst %.c,%.o,$(CS))
COBJS := $(foreach item,$(COBJS),$(OBJDIR)/$(item))
OBJS := $(subst /./,/,$(OBJS))
CUOBJS := $(subst /./,/,$(CUOBJS))
COBJS := $(subst /./,/,$(COBJS))

all : $(BINDIR)/$(OUTNAME)
	$(ECHO) Compile done.

run: all
	@cd $(BINDIR) && ./$(OUTNAME);

$(BINDIR)/$(OUTNAME): $(OBJS) $(CUOBJS) $(COBJS)
	$(ECHO) Linking: $@
	@if [ ! -d $(BINDIR) ]; then mkdir $(BINDIR); fi
	@$(CC) $(CFLAGS) $(LIBS) -o $@ $^ $(LDS) $(RPATH)

$(CUOBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cu
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CUCC) $(CUFLAGS) $(INCS) -c -o $@ $<

$(OBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

$(COBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.c
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)/$(OUTNAME)
