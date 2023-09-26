SOURCE_PATH=/Users/brian/solar_versions/solar-eclipse-8.5.1/lib_src/libplinkio/src
INCLUDE_PATH=/Users/brian/solar_versions/solar-eclipse-8.5.1/lib_src/libplinkio/include
C_OBJECTS= \
$(SOURCE_PATH)/plinkio.o \
$(SOURCE_PATH)/libcsv.o \
$(SOURCE_PATH)/file.o \
$(SOURCE_PATH)/fam_parse.o \
$(SOURCE_PATH)/fam.o \
$(SOURCE_PATH)/bim_parse.o \
$(SOURCE_PATH)/bim.o \
$(SOURCE_PATH)/bed_header.o \
$(SOURCE_PATH)/bed.o
HEADERS= \
$(INCLUDE_PATH)/csv.h \
$(INCLUDE_PATH)/plinkio.h \
$(INCLUDE_PATH)/plinkio/bed.h \
$(INCLUDE_PATH)/plinkio/bed_header.h \
$(INCLUDE_PATH)/plinkio/bim.h \
$(INCLUDE_PATH)/plinkio/bim_parse.h \
$(INCLUDE_PATH)/plinkio/fam.h \
$(INCLUDE_PATH)/plinkio/fam_parse.h \
$(INCLUDE_PATH)/plinkio/file.h \
$(INCLUDE_PATH)/plinkio/snp_lookup.h \
$(INCLUDE_PATH)/plinkio/snp_lookup_big.h \
$(INCLUDE_PATH)/plinkio/snp_lookup_little.h \
$(INCLUDE_PATH)/plinkio/status.h \
$(INCLUDE_PATH)/plinkio/utarray.h
