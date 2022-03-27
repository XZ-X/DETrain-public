SRCS = real.cpp \
       libinstrumenter.cpp

INCS = real.hh 

DEPS = $(SRCS) $(INCS)

CXX = clang++ 

CFLAGS = -O2 -Wall --std=c++11 -g -fno-omit-frame-pointer 

TARGETS = libinstrumenter.so

all: $(TARGETS)

$(TARGETS): $(DEPS)
	$(CXX) $(CFLAGS) $(INCLUDE_DIRS) -shared -fPIC $(SRCS) -o $(TARGETS) -ldl -lpthread 

clean:
	rm -f $(TARGETS)
