
FLAGS = -o
TARGET = test
OBJECTS = read.o

$(TARGET) : $(OBJECTS)
				$(CXX) $(FLAGS) $(TARGET) $(OBJECTS) `pkg-config --libs opencv`

clean :
				rm *.o $(TARGET)

new :
				$(MAKE) clean
				$(MAKE)
