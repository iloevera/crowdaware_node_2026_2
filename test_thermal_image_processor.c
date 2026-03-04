#include "thermal_image_processor.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

// Helper to fill an image with a constant value
static void fill_image(uint8_t img[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t val) {
    for (int y = 0; y < IMAGE_HEIGHT; y++)
        for (int x = 0; x < IMAGE_WIDTH; x++)
            img[y][x] = val;
}

// Compare two images, return 1 if equal
static int images_equal(const uint8_t a[IMAGE_HEIGHT][IMAGE_WIDTH], const uint8_t b[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    for (int y = 0; y < IMAGE_HEIGHT; y++)
        for (int x = 0; x < IMAGE_WIDTH; x++)
            if (a[y][x] != b[y][x]) return 0;
    return 1;
}

int main(void) {
    ThermalProcessor proc;
    uint8_t frame[IMAGE_HEIGHT][IMAGE_WIDTH];
    uint8_t out[IMAGE_HEIGHT][IMAGE_WIDTH];
    DetectedPerson people[MAX_PEOPLE];

    // test init
    thermal_processor_init(&proc);
    for (int y = 0; y < IMAGE_HEIGHT; y++)
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            assert(proc.background[y][x] == 0);
            assert(proc.work_buffer[y][x] == 0);
            assert(proc.labeled_image[y][x] == 0);
            assert(proc.distance_map[y][x] == 0);
        }
    assert(proc.background_update_counter == 0);

    // test background update and subtraction
    fill_image(frame, 50);
    update_background(&proc, frame, 255);
    // background should now equal frame
    assert(images_equal(proc.background, frame));
    // subtract
    memset(out, 0xff, sizeof(out));
    subtract_frames(&proc, frame, out);
    // difference should be zero
    for (int y = 0; y < IMAGE_HEIGHT; y++)
        for (int x = 0; x < IMAGE_WIDTH; x++)
            assert(out[y][x] == 0);

    // erosion/dilation on simple pattern
    uint8_t pattern[IMAGE_HEIGHT][IMAGE_WIDTH];
    fill_image(pattern, 0);
    // set a cross in the center
    pattern[IMAGE_HEIGHT/2][IMAGE_WIDTH/2] = 255;
    pattern[IMAGE_HEIGHT/2 - 1][IMAGE_WIDTH/2] = 255;
    pattern[IMAGE_HEIGHT/2 + 1][IMAGE_WIDTH/2] = 255;
    erode_3x3(pattern, out);
    // erosion should remove isolated pixels -> entire output should be 0
    fill_image(frame,0);
    assert(images_equal(out, frame));
    
    // dilation: start with single pixel
    fill_image(pattern, 0);
    pattern[IMAGE_HEIGHT/2][IMAGE_WIDTH/2] = 100;
    dilate_3x3(pattern, out);
    // dilation should propagate the value to neighbors
    assert(out[IMAGE_HEIGHT/2][IMAGE_WIDTH/2] == 100);
    assert(out[IMAGE_HEIGHT/2-1][IMAGE_WIDTH/2] == 100);
    assert(out[IMAGE_HEIGHT/2+1][IMAGE_WIDTH/2] == 100);
    assert(out[IMAGE_HEIGHT/2][IMAGE_WIDTH/2-1] == 100);
    assert(out[IMAGE_HEIGHT/2][IMAGE_WIDTH/2+1] == 100);

    // gaussian blur: uniform image should remain same
    fill_image(pattern, 123);
    gaussian_blur_3x3(pattern, out);
    assert(images_equal(pattern, out));

    // distance transform: binary image with single object pixel
    fill_image(pattern, 0);
    pattern[0][0] = 255;
    distance_transform(pattern, out);
    // object at (0,0) distance should be >0 and neighbors 1
    assert(out[0][0] > 0);
    assert(out[0][1] == 1 || out[1][0] == 1);

    // watershed: create simple distance map with two peaks
    uint8_t dist[IMAGE_HEIGHT][IMAGE_WIDTH];
    fill_image(dist, 0);
    dist[5][5] = 5;
    dist[5][6] = 4;
    dist[10][10] = 8;
    
    thermal_processor_init(&proc);
    uint8_t num = watershed(dist, &proc, people, MAX_PEOPLE);
    // expect at least 1 detection (two peaks but areas maybe too small)
    assert(num >= 1);

    // process pipeline: feed uniform frame
    fill_image(frame, 20);
    thermal_processor_init(&proc);
    num = process_thermal_frame(&proc, frame, people, MAX_PEOPLE);
    // no people since uniform background subtraction yields zero
    assert(num == 0);

    printf("All tests passed!\n");
    return 0;
}
