include sources.mk
.SUFFIXES: .c .cpp .o .h
.cpp.o:
	g++-7 -c -O3 -std=c++11 -I$(SOURCE_PATH) -w -o $@ $<
.c.o:
	gcc-7 -c -O3 -I$(SOURCE_PATH) -w -o $@ $<
.PHONY: all
all: libRicVolume.a
libRicVolume.a:  $(C_OBJECTS) $(CXX_OBJECTS) $(HEADERS)
	libtool -static -o $@ $(C_OBJECTS) $(CXX_OBJECTS)
install:
	mkdir -p /Users/brian/solar_versions/solar-eclipse-8.5.1/bin_src/solarmain-8.5.1/include
	mkdir -p /Users/brian/solar_versions/solar-eclipse-8.5.1/bin_src/solarmain-8.5.1/lib
	cp $(HEADERS) /Users/brian/solar_versions/solar-eclipse-8.5.1/bin_src/solarmain-8.5.1/include
	cp libRicVolume.a /Users/brian/solar_versions/solar-eclipse-8.5.1/bin_src/solarmain-8.5.1/lib
clean:
	rm -f $(C_OBJECTS) $(CXX_OBJECTS) libRicVolume.a
