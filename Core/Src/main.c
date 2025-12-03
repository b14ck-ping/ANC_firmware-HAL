/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "gpio.h"
#include "sai.h"
#include "tim.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#ifdef USING_USB
#include "usbd_cdc_if.h"
#endif
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* ADC settings */
#define VREF 3.3f
#define ADC_MAX_CODE 4095.0f
#define V_BIAS 1.25f	   // MAX9814 output DC bias (V)
#define MIC_GAIN_DB 40.0f  // software-set gain of MAX9814, dB

/* Microphone capsule sensitivity (without preamp) in dBV (V per Pa =
10^(dBV/20)).
   Typical electret values: -44 dBV (~6.31 mV/Pa), -36 dBV (~15.8 mV/Pa).
   MUST be set according to the actual microphone capsule used.
*/
#define MIC_SENSITIVITY_DBV (-44.0f)

#define SAMPLE_RATE_HZ 16000u
#define FRAMES_PER_HALF 256u  // фреймов в половине буфера (~5.3мс)
#define SLOTS 2u			  // L,R

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
#ifdef USING_USB
#pragma pack(push, 1)
typedef struct {
	uint8_t pre0;
	uint8_t pre1;
	uint32_t cnt;
	uint16_t err;
	uint16_t ref;
} usb_pkt_t;

#pragma pack(pop)
#endif
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* ----------------- HAL handles (from MX_..._Init) ----------------- */
extern ADC_HandleTypeDef hadc1;
extern TIM_HandleTypeDef htim6;
#ifdef USING_USB
extern USBD_HandleTypeDef hUsbDeviceFS;
#endif
extern SAI_HandleTypeDef hsai_BlockA1;
/* ----------------- Globals ----------------- */
/* ADC pair ring buffer (ref, err) - power of two recommended */
/* ADC DMA Buffer */
#define ADC_BUF_LEN 2U					 // two channels scanned
volatile uint16_t adc_buf[ADC_BUF_LEN];	 // filled by DMA (make volatile)

#ifdef USING_USB
#define ADC_RING_SIZE 4096U	 // must be power of two
_Static_assert((ADC_RING_SIZE & (ADC_RING_SIZE - 1)) == 0, "ADC_RING_SIZE must be power of two");
typedef struct {
	uint16_t ref;
	uint16_t err;
} adc_pair_t;
static volatile uint32_t adc_ring_head = 0, adc_ring_tail = 0;
// static uint16_t adc_ring[ADC_RING_SIZE];
static usb_pkt_t adc_ring[ADC_RING_SIZE];
static volatile uint32_t sample_cnt = 0;
static volatile uint32_t adc_dropped = 0;
extern volatile bool usb_busy;
#else
static volatile uint16_t last_err = 0;
static volatile uint16_t last_ref = 0;
#endif

/* Transmission batch */
#define TX_BATCH_LEN 32U

/* precomputed constants */
static float mic_sens_v_per_pa;	 // S (V/Pa)
static float mic_gain_linear;	 // G

// SAI DMA buffer

static int16_t i2s_tx_buf[2];

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
#ifdef USING_USB
static inline bool adc_ring_push_isr(const adc_pair_t *p);
static void adc_ring_advance_tail(uint32_t count);
static uint32_t adc_ring_count(void);

static void usb_try_send_batch_peek(void);
#endif

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */

	/* MCU
	 * Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the
	 * Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_DMA_Init();
	MX_ADC1_Init();
	MX_TIM6_Init();
	MX_SAI1_Init();
	/* USER CODE BEGIN 2 */
	mic_sens_v_per_pa = powf(10.0f, MIC_SENSITIVITY_DBV / 20.0f);  // V/Pa at gain=0dB
	mic_gain_linear = powf(10.0f, MIC_GAIN_DB / 20.0f);

	/* Start ADC in interrupt mode if necessary; if ADC triggered by TIM,
	 * HAL_ADC_Start_IT could be used */
	HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED);	// калибровка АЦП
	if (HAL_TIM_Base_Start(&htim6) != HAL_OK) {
		Error_Handler();
	}

	if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_buf, ADC_BUF_LEN) != HAL_OK) {
		Error_Handler();
	}

	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
		/* 2) Если USB готов — отправляем по USB. Используем
		 * peek-advance подход
		 */
		// if (hUsbDeviceFS.dev_state == USBD_STATE_CONFIGURED &&
		// !usb_busy) {
		//     usb_try_send_batch_peek();
		// }
		if (HAL_SAI_Transmit_DMA(&hsai_BlockA1, (uint8_t *)i2s_tx_buf, sizeof(i2s_tx_buf) / sizeof(uint16_t)) !=
			HAL_OK) {
			Error_Handler();
		}
	}

	/* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
	RCC_OscInitTypeDef RCC_OscInitStruct = {0};
	RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

	/** Configure the main internal regulator output voltage
	 */
	if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK) {
		Error_Handler();
	}

	/** Initializes the RCC Oscillators according to the specified
	 * parameters in the RCC_OscInitTypeDef structure.
	 */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
	RCC_OscInitStruct.HSIState = RCC_HSI_ON;
	RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
	RCC_OscInitStruct.PLL.PLLM = 1;
	RCC_OscInitStruct.PLL.PLLN = 10;
	RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
	RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
	RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
		Error_Handler();
	}

	/** Initializes the CPU, AHB and APB buses clocks
	 */
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
		Error_Handler();
	}
}

/* USER CODE BEGIN 4 */
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc) { (void)hadc; }

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
	(void)hadc;
	/* fast: read both ADC values and push into adc_ring */
	/*
	RANK_1 = PA6 = CH11 = err = green
	RANK_2 = PA7 = CH12 = ref = yellow
  */
#ifdef USING_USB
	adc_pair_t p;
	p.ref = adc_buf[0];	 // rank1
	p.err = adc_buf[1];	 // rank2
	// ISR-safe push
	(void)adc_ring_push_isr(&p);
	HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_3);
#else
	last_err = adc_buf[0];
#endif
}

#ifdef USING_USB
/* ----------------- processing ----------------- */
/* ----------------- Ring helpers ----------------- */
/* ADC ring push (ISR-safe, fast) */
static inline bool adc_ring_push_isr(const adc_pair_t *p) {
	uint32_t head = adc_ring_head;
	uint32_t next = (head + 1) & (ADC_RING_SIZE - 1);
	uint32_t tail = adc_ring_tail;	// tail может меняться в main, но чтение
									// здесь "best-effort"
	++sample_cnt;
	if (next == tail) {
		// кольцо полно — отбрасываем пакет (не перезаписываем)
		adc_dropped++;
		return false;
	}
	adc_ring[head].pre0 = 0xAA;
	adc_ring[head].pre1 = 0x55;
	adc_ring[head].cnt = sample_cnt;
	adc_ring[head].err = p->err;
	adc_ring[head].ref = p->ref;
	// publish
	__asm__ volatile("" ::: "memory");	// небольшая баррикада
	adc_ring_head = next;
	return true;
}

/* ADC ring pop (main) */
static inline bool adc_ring_peek_at(uint32_t offset, usb_pkt_t *out) {
	if (offset >= adc_ring_count())
		return false;
	uint32_t pos = (adc_ring_tail + offset) & (ADC_RING_SIZE - 1);
	*out = adc_ring[pos];
	return true;
}

static inline void adc_ring_advance_tail(uint32_t count) {
	__disable_irq();
	adc_ring_tail = (adc_ring_tail + count) & (ADC_RING_SIZE - 1);
	__enable_irq();
}

static inline uint32_t adc_ring_count(void) {
	uint32_t tail = 0, head = 0;
	__disable_irq();
	tail = adc_ring_tail;
	head = adc_ring_head;
	__enable_irq();
	return (head - tail) & (ADC_RING_SIZE - 1);
}

static void usb_try_send_batch_peek(void) {
	uint32_t avail = adc_ring_count();
	if (hUsbDeviceFS.dev_state != USBD_STATE_CONFIGURED || usb_busy || avail < TX_BATCH_LEN || avail == 0)
		return;

	uint32_t to_send = (avail > TX_BATCH_LEN) ? TX_BATCH_LEN : avail;

	static usb_pkt_t sendbuf[TX_BATCH_LEN];

	/* peek items into sendbuf without modifying tail */
	for (uint32_t i = 0; i < to_send; ++i) {
		if (!adc_ring_peek_at(i, &sendbuf[i])) {
			to_send = i;
			break;
		}
	}
	if (to_send == 0)
		return;

	uint8_t ret = CDC_Transmit_FS((uint8_t *)sendbuf, (uint16_t)(to_send * sizeof(usb_pkt_t)));
	if (ret == USBD_OK) {
		adc_ring_advance_tail(to_send);
		usb_busy = true;
	} else if (ret == USBD_BUSY) {
		/* НЕ трогаем кольцо — попробуем позже. */
	} else {
		/* Ошибка — не теряем пакеты: просто попробуем позже. */
	}
}
#endif
/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return
	 * state
	 */
	__disable_irq();
	while (1) {
	}
	/* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line) {
	/* USER CODE BEGIN 6 */
	/* User can add his own implementation to report the file name and line
	   number, ex: printf("Wrong parameters value: file %s on line %d\r\n",
	   file, line) */
	/* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
