include sources.mk
.SUFFIXES: .c .o .h
.c.o:
	gcc-7 -c -O3 -I$(INCLUDE_PATH) -o $@ $<
.PHONY: all
all: libplinkio.a
libplinkio.a :  $(C_OBJECTS) $(HEADERS)
	libtool -static -o $@ $(C_OBJECTS) 
install:
	mkdir -p /Users/brian/solar_versions/solar-eclipse-8.5.1/bin_src/solarmain-8.5.1/include
	mkdir -p /Users/brian/solar_versions/solar-eclipse-8.5.1/bin_src/solarmain-8.5.1/lib
	cp -R /Users/brian/solar_versions/solar-eclipse-8.5.1/lib_src/libplinkio/include/* /Users/brian/solar_versions/solar-eclipse-8.5.1/bin_src/solarmain-8.5.1/include
	cp libplinkio.a /Users/brian/solar_versions/solar-eclipse-8.5.1/bin_src/solarmain-8.5.1/lib
clean:
	rm -f $(C_OBJECTS) libplinkio.a
