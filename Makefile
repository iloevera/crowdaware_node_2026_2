CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2

TARGET = test_thermal_image_processor

SRCS = thermal_image_processor.c test_thermal_image_processor.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
