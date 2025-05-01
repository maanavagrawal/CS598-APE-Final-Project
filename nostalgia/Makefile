.PHONY: all clean build

all: clean build

clean:
	rm -f *.so
	python3 setup.py clean --all

build:
	python3 setup.py build_ext --inplace

test_all:
	python3 test_fingerprint_pybind.py
test_cpp:
	python3 test_fingerprint_cpp_version.py