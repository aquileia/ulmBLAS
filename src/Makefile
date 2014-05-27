CC       = gcc-4.8
CFLAGS  += -Wall -I.. -O3

DIRS             = level1 level3 auxiliary
SOURCE_FILES     = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.c))
OBJECT_FILES     = $(SOURCE_FILES:.c=.o)
ATL_SOURCE_FILES = $(foreach file, $(SOURCE_FILES), $(dir $(file))atl_$(notdir $(file)))
ATL_OBJECT_FILES = $(ATL_SOURCE_FILES:.c=.o)

ULMBLAS      = ../libulmblas.a
ATLULMBLAS   = ../libatlulmblas.a

all : $(ULMBLAS) $(ATLULMBLAS)

$(ULMBLAS) : $(OBJECT_FILES)
	ar cru $(ULMBLAS) $(OBJECT_FILES)
	ranlib $(ULMBLAS)

$(ATLULMBLAS) : $(ATL_OBJECT_FILES)
	ar cru $(ATLULMBLAS) $(ATL_OBJECT_FILES)
	ranlib $(ATLULMBLAS)

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

atl_%.o : %.c
	$(CC) $(CFLAGS) -DFAKE_ATLAS -c -o $@ $<

clean :
	rm -f $(OBJECT_FILES)
	rm -f $(ATL_OBJECT_FILES)
	rm -f $(ULMBLAS)
	rm -f $(ATLULMBLAS)

echo:
	@echo $(SOURCE_FILES)
	@echo $(OBJECT_FILES)
	@echo $(ATL_SOURCE_FILES)
	@echo $(ATL_OBJECT_FILES)

