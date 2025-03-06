##########################################################
## CC compiler options

CC=g++
CC_FLAGS= -Wall -pedantic -O3 -fomit-frame-pointer -fstrict-aliasing -ffast-math -std=c++11 -mavx
CC_LIBS=-lpthread

ifeq ($(DBG),true)
     CC_FLAGS += -g -ggdb
endif

##########################################################
## Sources

SRC_DIR = .
OBJ_DIR = .
INC_DIR = .
CC_SRCS = main.cpp pgn.cpp ../magics.cpp ../bitboards.cpp ../zobrist.cpp ../position.cpp


EXE = tune.exe

CC_OBJS = $(CC_SRCS:%.cpp=%.o)
OBJS = $(CC_OBJS)
LINK = $(CC_LIBS)

$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(LINK)

$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

clean:
	find . -name "*.o" | xargs rm -vf
	find . -name "*.ii" | xargs rm -vf
	find . -name "*.s" | xargs rm -vf
	rm -vf *~ *#
