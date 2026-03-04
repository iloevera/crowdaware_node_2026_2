#include "thermal_image_processor.h"
#include <string.h>

/**
 * @brief Helper: Clamp pixel coordinates to image bounds
 */
static inline uint8_t pixel_clamp(const uint8_t img[IMAGE_HEIGHT][IMAGE_WIDTH], int y, int x) {
    if (y < 0) y = 0;
    else if (y >= IMAGE_HEIGHT) y = IMAGE_HEIGHT - 1;
    if (x < 0) x = 0;
    else if (x >= IMAGE_WIDTH) x = IMAGE_WIDTH - 1;
    return img[y][x];
}

/**
 * @brief Helper: Clamp value to 0-255
 */
static inline uint8_t clamp_u8(int32_t value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (uint8_t)value;
}

// ============================================================================
// Public API Implementation
// ============================================================================

void thermal_processor_init(ThermalProcessor* processor) {
    if (!processor) return;
    
    memset(processor->background, 0, sizeof(processor->background));
    memset(processor->work_buffer, 0, sizeof(processor->work_buffer));
    memset(processor->labeled_image, 0, sizeof(processor->labeled_image));
    memset(processor->distance_map, 0, sizeof(processor->distance_map));
    processor->background_update_counter = 0;
}

void update_background(ThermalProcessor* processor, const uint8_t current_frame[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t alpha) {
    if (!processor || !current_frame) return;
    
    // Exponential moving average: background = (1 - alpha/255) * background + (alpha/255) * current
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            uint32_t bg = processor->background[y][x];
            uint32_t cur = current_frame[y][x];
            
            // Weighted average
            uint32_t new_bg = ((255 - alpha) * bg + alpha * cur) / 255;
            processor->background[y][x] = (uint8_t)new_bg;
        }
    }
    
    processor->background_update_counter++;
}

void subtract_frames(const ThermalProcessor* processor, const uint8_t current_frame[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    if (!processor || !current_frame || !output) return;
    
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int32_t diff = (int32_t)current_frame[y][x] - (int32_t)processor->background[y][x];
            output[y][x] = clamp_u8(diff);
        }
    }
}

void erode_3x3(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    if (!input || !output) return;
    
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            uint8_t min_val = 255;
            
            // 3x3 neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uint8_t val = pixel_clamp(input, y + dy, x + dx);
                    if (val < min_val) {
                        min_val = val;
                    }
                }
            }
            
            output[y][x] = min_val;
        }
    }
}

void dilate_3x3(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    if (!input || !output) return;
    
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            uint8_t max_val = 0;
            
            // 3x3 neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uint8_t val = pixel_clamp(input, y + dy, x + dx);
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
            
            output[y][x] = max_val;
        }
    }
}

void gaussian_blur_3x3(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    if (!input || !output) return;
    
    /*
     * Gaussian kernel (3x3):
     * [ 1  2  1 ]
     * [ 2  4  2 ] / 16
     * [ 1  2  1 ]
     */
    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            uint32_t sum = 0;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uint8_t val = pixel_clamp(input, y + dy, x + dx);
                    sum += val * kernel[dy + 1][dx + 1];
                }
            }
            
            output[y][x] = (uint8_t)(sum / 16);
        }
    }
}

void distance_transform(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    if (!input || !output) return;
    
    // Initialize: 0 for background, max for foreground
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            output[y][x] = (input[y][x] > 128) ? DT_MAX_DISTANCE : 0;
        }
    }
    
    // Forward pass: propagate distances from top-left
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            if (output[y][x] == 0) continue;
            
            uint8_t min_dist = DT_MAX_DISTANCE;
            
            // Check top and left neighbors
            if (y > 0 && output[y - 1][x] > 0) {
                min_dist = (output[y - 1][x] > 1) ? output[y - 1][x] - 1 : 1;
            }
            if (x > 0 && output[y][x - 1] > 0) {
                uint8_t left_dist = (output[y][x - 1] > 1) ? output[y][x - 1] - 1 : 1;
                if (left_dist < min_dist) min_dist = left_dist;
            }
            
            if (min_dist < DT_MAX_DISTANCE) {
                output[y][x] = min_dist;
            }
        }
    }
    
    // Backward pass: propagate distances from bottom-right
    for (int y = IMAGE_HEIGHT - 1; y >= 0; y--) {
        for (int x = IMAGE_WIDTH - 1; x >= 0; x--) {
            if (output[y][x] == 0) continue;
            
            uint8_t min_dist = output[y][x];
            
            // Check bottom and right neighbors
            if (y < IMAGE_HEIGHT - 1 && output[y + 1][x] > 0) {
                uint8_t bottom_dist = (output[y + 1][x] > 1) ? output[y + 1][x] - 1 : 1;
                if (bottom_dist < min_dist) min_dist = bottom_dist;
            }
            if (x < IMAGE_WIDTH - 1 && output[y][x + 1] > 0) {
                uint8_t right_dist = (output[y][x + 1] > 1) ? output[y][x + 1] - 1 : 1;
                if (right_dist < min_dist) min_dist = right_dist;
            }
            
            if (min_dist < output[y][x]) {
                output[y][x] = min_dist;
            }
        }
    }
}

/**
 * @brief Helper: Flood fill to label connected components (watershed)
 */
static void flood_fill_label(uint8_t labeled[IMAGE_HEIGHT][IMAGE_WIDTH], 
                            const uint8_t distance_map[IMAGE_HEIGHT][IMAGE_WIDTH],
                            int start_y, int start_x, uint8_t label,
                            uint16_t* area, int* centroid_x, int* centroid_y) {
    if (label == 0 || labeled[start_y][start_x] != 0) return;
    
    int stack_y[IMAGE_SIZE];
    int stack_x[IMAGE_SIZE];
    int stack_ptr = 0;
    
    // Push initial pixel
    stack_y[stack_ptr] = start_y;
    stack_x[stack_ptr] = start_x;
    stack_ptr++;
    
    *area = 0;
    *centroid_x = 0;
    *centroid_y = 0;
    
    // Flood fill with stack-based approach
    while (stack_ptr > 0) {
        stack_ptr--;
        int y = stack_y[stack_ptr];
        int x = stack_x[stack_ptr];
        
        if (x < 0 || x >= IMAGE_WIDTH || y < 0 || y >= IMAGE_HEIGHT) continue;
        if (labeled[y][x] != 0) continue;
        if (distance_map[y][x] == 0) continue;
        
        labeled[y][x] = label;
        (*area)++;
        *centroid_x += x;
        *centroid_y += y;
        
        // Push 4-connected neighbors
        if (stack_ptr + 4 <= IMAGE_SIZE) {
            stack_y[stack_ptr] = y - 1; stack_x[stack_ptr] = x; stack_ptr++;
            stack_y[stack_ptr] = y + 1; stack_x[stack_ptr] = x; stack_ptr++;
            stack_y[stack_ptr] = y;     stack_x[stack_ptr] = x - 1; stack_ptr++;
            stack_y[stack_ptr] = y;     stack_x[stack_ptr] = x + 1; stack_ptr++;
        }
    }
}

uint8_t watershed(const uint8_t distance_map[IMAGE_HEIGHT][IMAGE_WIDTH], 
                  ThermalProcessor* processor,
                  DetectedPerson detected_people[],
                  uint8_t max_people) {
    if (!distance_map || !processor || !detected_people || max_people == 0) return 0;
    
    // Clear labeled image
    memset(processor->labeled_image, 0, sizeof(processor->labeled_image));
    
    uint8_t num_people = 0;
    
    // Find local maxima and label from them (watershed source points)
    for (int y = 1; y < IMAGE_HEIGHT - 1 && num_people < max_people; y++) {
        for (int x = 1; x < IMAGE_WIDTH - 1 && num_people < max_people; x++) {
            if (processor->labeled_image[y][x] != 0) continue;
            if (distance_map[y][x] == 0) continue;
            
            // Check if local maximum
            uint8_t center = distance_map[y][x];
            int is_maximum = 1;
            
            for (int dy = -1; dy <= 1 && is_maximum; dy++) {
                for (int dx = -1; dx <= 1 && is_maximum; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (distance_map[y + dy][x + dx] > center) {
                        is_maximum = 0;
                    }
                }
            }
            
            if (is_maximum && center > 0) {
                // Found a local maximum - start flood fill
                uint8_t label = num_people + 1;
                uint16_t area = 0;
                int centroid_x = 0, centroid_y = 0;
                
                flood_fill_label(processor->labeled_image, distance_map, y, x, label, &area, &centroid_x, &centroid_y);
                
                // Check if area is within valid range
                if (area >= MIN_PERSON_AREA && area <= MAX_PERSON_AREA) {
                    detected_people[num_people].x = centroid_x / area;
                    detected_people[num_people].y = centroid_y / area;
                    detected_people[num_people].area = area;
                    detected_people[num_people].max_distance = distance_map[y][x];
                    num_people++;
                } else {
                    // Remove this label if area is invalid
                    for (int yy = 0; yy < IMAGE_HEIGHT; yy++) {
                        for (int xx = 0; xx < IMAGE_WIDTH; xx++) {
                            if (processor->labeled_image[yy][xx] == label) {
                                processor->labeled_image[yy][xx] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return num_people;
}

uint8_t process_thermal_frame(ThermalProcessor* processor,
                              const uint8_t current_frame[IMAGE_HEIGHT][IMAGE_WIDTH],
                              DetectedPerson detected_people[],
                              uint8_t max_people) {
    if (!processor || !current_frame || !detected_people || max_people == 0) return 0;
    
    uint8_t num_people = 0;
    
    // Step 1: Background subtraction
    subtract_frames(processor, current_frame, processor->work_buffer);
    
    // Step 2: Morphological opening (erosion followed by dilation)
    uint8_t eroded[IMAGE_HEIGHT][IMAGE_WIDTH];
    erode_3x3(processor->work_buffer, eroded);
    dilate_3x3(eroded, processor->work_buffer);
    
    // Step 3: Gaussian blur for preprocessing
    uint8_t blurred[IMAGE_HEIGHT][IMAGE_WIDTH];
    gaussian_blur_3x3(processor->work_buffer, blurred);
    
    // Step 4: Distance transform
    distance_transform(blurred, processor->distance_map);
    
    // Step 5: Watershed segmentation
    num_people = watershed(processor->distance_map, processor, detected_people, max_people);
    
    return num_people;
}
