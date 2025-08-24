#include "dac.h"
#include "gpio.h"
#include "stm32l4xx_hal.h"
#include <memory>

int main() {
    // Initialize HAL and system clocks
    HAL_Init();
    MX_DAC1_Init();
    MX_GPIO_Init();

    // Use C++ abstractions
    // auto dac = std::make_unique<DAC>();
    // dac->setValue(2048); // Example usage
    
    while(1) {
        // Main loop
        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_3); // Toggle a GPIO pin
        HAL_Delay(250);
    }
    
    return 0;
}