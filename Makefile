APP_NAME = test.app
BUILD_DIR = build
BIN_DIR = bin
CXX = g++
SHADERSCOMPILER = glslangValidator

CXXFLAGS = -std=c++17 -fopenmp -O0 -g -Wall -Warray-bounds -Wdiv-by-zero -fno-omit-frame-pointer
CXXFLAGS += -DDEBUG
CXXFLAGS += -fsanitize=address -fsanitize=undefined -fsanitize=bounds -fsanitize=bounds-strict

LDFLAGS = -lgomp -lvulkan

VPATH = Networks 
VPATH += Learning 
VPATH += Functions 
VPATH += DataProviders
VPATH += libs/Math
VPATH += libs/Vulkan

SOURCE = main.cpp
#SOURCE += vtests.cpp
#SOURCE += cputest.cpp
SOURCE += lstmtest.cpp
SOURCE += $(wildcard Networks/*.cpp)
SOURCE += $(wildcard Learning/*.cpp)
SOURCE += $(wildcard Functions/*.cpp)
SOURCE += $(wildcard DataProviders/*.cpp)
SOURCE += $(wildcard libs/Math/*.cpp)
SOURCE += $(wildcard libs/Vulkan/*.cpp)

OBJECTS = $(notdir $(SOURCE:.cpp=.o))

all: prepere $(BIN_DIR)/$(APP_NAME) shaders

$(BIN_DIR)/$(APP_NAME): $(addprefix $(BUILD_DIR)/,$(OBJECTS))
	$(CXX) -o $(BIN_DIR)/$(APP_NAME) $(addprefix $(BUILD_DIR)/,$(OBJECTS)) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

.PHONY: prepere run clean dbg shaders

prepere:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)

shaders:
	$(SHADERSCOMPILER) -V -o bin/comp.spv Shaders/test.comp

run: all $(BIN_DIR)/$(APP_NAME)
	./$(BIN_DIR)/$(APP_NAME)

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)

dbg: all $(BIN_DIR)/$(APP_NAME)
	gdb ./$(BIN_DIR)/$(APP_NAME)
