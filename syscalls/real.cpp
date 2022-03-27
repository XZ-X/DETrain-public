#include <dlfcn.h>
#include <stdlib.h>

#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "real.hh"

#define DEFINE_WRAPPER(name) decltype(::name) * name;
#define INIT_WRAPPER(name, handle) name = (decltype(::name)*)dlsym(handle, #name);

namespace Real {
	DEFINE_WRAPPER(fopen);
	DEFINE_WRAPPER(fclose);
	DEFINE_WRAPPER(read);
	DEFINE_WRAPPER(open64);

	void initializer() {
		INIT_WRAPPER(fopen, RTLD_NEXT);
		INIT_WRAPPER(fclose, RTLD_NEXT);
		INIT_WRAPPER(read, RTLD_NEXT);
		INIT_WRAPPER(open64, RTLD_NEXT);
	}
}
