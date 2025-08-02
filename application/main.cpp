#include "dac.h"
#include "stm32l4xx_hal.h"
#include <memory>

int main() {
    // Initialize HAL and system clocks
    HAL_Init();
    MX_DAC1_Init();
    
    // Use C++ abstractions
    // auto dac = std::make_unique<DAC>();
    // dac->setValue(2048); // Example usage
    
    while(1) {
        // Main loop
    }
    
    return 0;
}