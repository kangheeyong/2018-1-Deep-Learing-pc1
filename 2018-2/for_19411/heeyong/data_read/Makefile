
FLAGS = -o
TARGET = test
OBJECTS = main.o

$(TARGET) : $(OBJECTS)
				$(CXX) $(FLAGS) $(TARGET) $(OBJECTS) `pkg-config --libs opencv`

clean :
				rm *.o $(TARGET)

new :
				$(MAKE) clean
				$(MAKE)
