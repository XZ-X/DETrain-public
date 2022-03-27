#ifndef __REAL_HH_
#define __REAL_HH_

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <sys/random.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define DECLARE_WRAPPER(name) extern decltype(::name) * name;

namespace Real {
	void initializer();
	DECLARE_WRAPPER(fopen);
	DECLARE_WRAPPER(fclose);
	DECLARE_WRAPPER(read);
	DECLARE_WRAPPER(open64);
};

#endif
