
#include <stdio.h>
#include <dlfcn.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include "real.hh"

__attribute__((constructor)) void initializer() {
  Real::initializer();
}

int rand_fd = -1;

FILE* fopen(const char* filename, const char* modes) {
  FILE* file = Real::fopen(filename, modes);
  if(strcmp (filename, "/dev/urandom") == 0) {
    //fprintf(stderr, "fopen*****random device*****\n");
    rand_fd = fileno(file);
  }
  return file; 
}

int open64(const char *path, int oflag, ... ) {
  int mode;
  if(oflag & O_CREAT) {
    va_list arg;
    va_start(arg, oflag);
    mode = va_arg(arg, mode_t);
    va_end(arg);
  } else {
    mode = 0;
  }
  return Real::open64(path, oflag, mode);  
}

ssize_t read(int fd, void* buf, size_t count) {
  ssize_t ret = 0;
  ret = Real::read(fd, buf, count);
  if(fd == rand_fd) {
    char* a = (char*) buf;
    for(int i=0; i<ret; i++) {
      //fprintf(stderr, "%d: [%d]\n", i, a[i]);
      a[i] = 41;
    }
  }
  return ret;
}

int fclose(FILE* stream){
  int fd = fileno(stream);
  if(fd == rand_fd) {
    rand_fd = -1;
  }
  int ret = Real::fclose(stream);
  return ret;
}

