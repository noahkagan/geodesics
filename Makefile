-include Makefile.config

# External commands
CMAKE ?= cmake
CLANG_FORMAT ?= clang-format
FIND ?= find
MKDIR ?= mkdir -p
PYTHON ?= python
RM_R ?= rm -fr
XARGS ?= xargs

## Path and build flags
FLAVOR ?= optimize
BUILD = build/$(FLAVOR)
#prefix ?= /usr/local
#libdir ?= lib

## Temporary staging directory
# DESTDIR =
## Specified by `git make-pkg` when building .pkg files
# mac_pkg =

ifdef prefix
    CMAKE_ARGS += -DCMAKE_INSTALL_PREFIX=$(prefix)
endif
ifdef libdir
    CMAKE_ARGS += -DCMAKE_INSTALL_LIBDIR=$(libdir)
endif
ifdef FLAVOR
    CMAKE_ARGS += -DFLAVOR=$(FLAVOR)
endif

CMAKE_FILES += Makefile
CMAKE_FILES += $(wildcard CMakeLists.txt */*/CMakeLists.txt)

export CXX
export DESTDIR
export prefix

all: $(BUILD)
	$(MAKE) -C $(BUILD) $@

install: all
	$(MAKE) -C $(BUILD) $@

test: all
	$(MAKE) -C $(BUILD) $@

clean:
	$(RM_R) $(BUILD) Linux-* Darwin-*

$(BUILD): $(CMAKE_FILES)
	mkdir -p $(BUILD)
	cd $(BUILD) && $(CMAKE) $(CMAKE_ARGS) $(EXTRA_CMAKE_ARGS) ../..
	touch $@

format:
	find . -name '*.cpp' -o -name '*.h' | $(XARGS) $(CLANG_FORMAT) -i
