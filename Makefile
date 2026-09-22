CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -fPIC
PYTHON ?= ./.venv/bin/python

# Project directories
ROOT_DIR := $(shell pwd)
CPP_SRC_DIR := $(ROOT_DIR)/GoTorch/backend/cppsrc
TEST_SRC_DIR := $(ROOT_DIR)/tests/backend/cppsrc
BUILD_DIR := $(ROOT_DIR)/build
BIN_DIR := $(BUILD_DIR)/bin
GTEST_DIR := $(BUILD_DIR)/gtest

# Compiler flags
INCLUDES := -I$(CPP_SRC_DIR) -isystem $(GTEST_DIR)/src/googletest/include
LDFLAGS := -pthread

.PHONY: all build test-cpp test-specific clean gtest help

all: build

# 1. Build the Python C++ native extension
build:
	@echo "==> Building GoTorch native_backend extension..."
	PYTHON=$(PYTHON) $(CPP_SRC_DIR)/build.sh

# 2. Setup GoogleTest dependency
$(GTEST_DIR)/libgtest.a:
	@echo "==> Setting up GoogleTest in $(GTEST_DIR)..."
	@mkdir -p $(GTEST_DIR)
	@if [ ! -d "$(GTEST_DIR)/src" ]; then \
		git clone --depth=1 https://github.com/google/googletest.git $(GTEST_DIR)/src; \
	fi
	@$(CXX) -std=c++17 -isystem $(GTEST_DIR)/src/googletest/include -I$(GTEST_DIR)/src/googletest -pthread -c $(GTEST_DIR)/src/googletest/src/gtest-all.cc -o $(GTEST_DIR)/gtest-all.o
	@$(CXX) -std=c++17 -isystem $(GTEST_DIR)/src/googletest/include -I$(GTEST_DIR)/src/googletest -pthread -c $(GTEST_DIR)/src/googletest/src/gtest_main.cc -o $(GTEST_DIR)/gtest_main.o
	@ar -rv $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/gtest-all.o $(GTEST_DIR)/gtest_main.o

gtest: $(GTEST_DIR)/libgtest.a

# 3. Build C++ test executable
$(BIN_DIR)/test_ndarray: $(GTEST_DIR)/libgtest.a $(CPP_SRC_DIR)/ndarray.cpp $(TEST_SRC_DIR)/test_ndarray.cpp
	@mkdir -p $(BIN_DIR)
	@echo "==> Compiling test_ndarray..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) \
		$(CPP_SRC_DIR)/ndarray.cpp \
		$(TEST_SRC_DIR)/test_ndarray.cpp \
		$(GTEST_DIR)/libgtest.a \
		$(LDFLAGS) \
		-o $(BIN_DIR)/test_ndarray

# 4. Run all C++ tests (or specific tests via TEST=... or FILTER=...)
test-cpp: $(BIN_DIR)/test_ndarray
	@if [ -n "$(TEST)" ]; then \
		echo "==> Running specific C++ test matching *$(TEST)*..."; \
		$(BIN_DIR)/test_ndarray --gtest_filter="*$(TEST)*"; \
	elif [ -n "$(FILTER)" ]; then \
		echo "==> Running C++ tests with filter $(FILTER)..."; \
		$(BIN_DIR)/test_ndarray --gtest_filter="$(FILTER)"; \
	else \
		echo "==> Running all C++ tests..."; \
		$(BIN_DIR)/test_ndarray; \
	fi

# 5. Run a specific test (alias: make test-specific TEST=...)
test-specific: $(BIN_DIR)/test_ndarray
	@if [ -z "$(TEST)" ] && [ -z "$(FILTER)" ]; then \
		echo "Error: Please specify TEST=<name> or FILTER=<pattern>, e.g.:"; \
		echo "  make test-specific TEST=ShouldProperlyAddTwoMatrices"; \
		exit 1; \
	fi
	@if [ -n "$(TEST)" ]; then \
		$(BIN_DIR)/test_ndarray --gtest_filter="*$(TEST)*"; \
	else \
		$(BIN_DIR)/test_ndarray --gtest_filter="$(FILTER)"; \
	fi

# 6. Clean build artifacts
clean:
	@echo "==> Cleaning build artifacts..."
	@rm -rf $(BIN_DIR) $(BUILD_DIR)/*.o $(BUILD_DIR)/test_ndarray
	@rm -f $(ROOT_DIR)/GoTorch/backend/native_backend*.so
	@echo "Done."

# 7. Help
help:
	@echo "Available commands:"
	@echo "  make build                     Build native C++ backend extension"
	@echo "  make test-cpp                  Run all C++ tests"
	@echo "  make test-cpp TEST=<name>      Run specific C++ test matching pattern"
	@echo "  make test-specific TEST=<name> Run specific C++ test"
	@echo "  make clean                     Remove build artifacts"
