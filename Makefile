APP_NAME = test.app
BUILD_DIR = build
CXX = '${OFFLOAD_GCC_PATH}/g++'

CXXFLAGS = -std=c++17 -fopenmp -foffload=-lm -O0 -g -Wall -Warray-bounds -Wdiv-by-zero
CXXFLAGS += -DDEBUG
LDFLAGS = -lgomp

VPATH = Networks Learning
SOURCE = main.cpp $(wildcard Networks/*.cpp) $(wildcard Learning/*.cpp)

OBJECTS = $(notdir $(SOURCE:.cpp=.o))

all: prepere $(APP_NAME)

$(APP_NAME): $(addprefix $(BUILD_DIR)/,$(OBJECTS))
	$(CXX) -o $(APP_NAME) $(addprefix $(BUILD_DIR)/,$(OBJECTS)) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

.PHONY: prepere run clean dbg shaders

prepere:
	mkdir -p $(BUILD_DIR)

shaders:
	

run: all $(APP_NAME)
	./$(APP_NAME)

clean:
	rm -r $(BUILD_DIR)
	rm -f $(APP_NAME)

dbg: all $(APP_NAME)
	gdb ./$(APP_NAME)
