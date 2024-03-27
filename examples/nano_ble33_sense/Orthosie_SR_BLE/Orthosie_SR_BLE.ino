/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <Light-Dark_inferencing.h>

/************* BLE **************/
#include <ArduinoBLE.h>
#define RELAY_LED D2
#define RELAY_POWER D4
#define HEARTBEAT_LED D5

BLEService smartplugService("280A"); // BLE LED Service

// BLE LED Switch Characteristic - custom 128-bit UUID, read and writable by central
BLEByteCharacteristic smartplugCharacteristic("2A57", BLERead | BLEWrite);
/**
GPIO_25 P0.04 ADC - A0 - ADC
GPIO_39 P1.11 D2  - D2 - IO13 - RELAY_LED
GPIO_40 P1.15 D4  - D4 - IO5 - RELAY_POWER
GPIO_41 P1.14 D6  - D6 - IO12
GPIO_47 P0.23 D7  - D7 - TX0
GPIO_48 P0.21 D8  - D8 - RX0
GPIO_22 P1.13 D5  - D5 - IO14 - HEARTBEAT_LED
GPIO_21 P1.12 D3  - D3 - IO4

GPIO_42 P0.26 PDMCLK
GPIO_51 P0.17 MIC_PWR
GPIO_07 P0.25 PDMDIN
RESET_N P0.18
*/

bool plug = LOW;
/************* BLE **************/

// Variables will change:
int marvinCall = LOW;  // ledState used to set the LED

// Generally, you should use "unsigned long" for variables that hold time
// The value will quickly become too large for an int to store
unsigned long previousMillis = 0;  // will store last time LED was updated



// constants won't change:
const long interval = 5000;  // interval at which to blink (milliseconds)

/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    //while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");

/********** BLE *************/
    // set LED's pin to output mode
    pinMode(RELAY_LED, OUTPUT);
    pinMode(RELAY_POWER, OUTPUT);
    pinMode(HEARTBEAT_LED, OUTPUT);
    
    
    digitalWrite(RELAY_LED, plug);
    digitalWrite(RELAY_POWER, plug);

    // begin initialization
    if (!BLE.begin()) {
      Serial.println("starting Bluetooth® Low Energy failed!");

      while (1);
    }

    // set advertised local name and service UUID:
    BLE.setLocalName("Orthosie");
    BLE.setAdvertisedService(smartplugService);

    // add the characteristic to the service
    smartplugService.addCharacteristic(smartplugCharacteristic);

    // add service
    BLE.addService(smartplugService);

    // set the initial value for the characteristic:
    smartplugCharacteristic.writeValue(0);

    // start advertising
    BLE.advertise();

    Serial.println("BLE Smart Plug");
/********** BLE *************/

    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) /
                                            sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    unsigned long currentMillis = millis();

/************** BLE **************/
    // listen for Bluetooth® Low Energy peripherals to connect:
    BLEDevice central = BLE.central();
    //Serial.println("In Loop");
    // if a central is connected to peripheral:
    if (central) {
      Serial.print("Connected to central: ");
      // print the central's MAC address:
      Serial.println(central.address());
      digitalWrite(HEARTBEAT_LED, HIGH);            // turn on the LED to indicate the connection

      // while the central is still connected to peripheral:
      while (central.connected()) {
        // if the remote device wrote to the characteristic,
        // use the value to control the LED:
        if (smartplugCharacteristic.written()) {
          switch (smartplugCharacteristic.value()) {   // any value other than 0
            case 00:
              Serial.println("Smart Plug OFF");
              turnPlugOff();
              break;
            case 01:
              Serial.println("Smart Plug ON");
              turnPlugOn();
              break;
            default:
              Serial.println(F("Smart Plug OFF"));
              turnPlugOff();
              break;
          }
        }
      }

      // when the central disconnects, print it out:
      Serial.print(F("Disconnected from central: "));
      Serial.println(central.address());
      digitalWrite(HEARTBEAT_LED, LOW);            // turn off the LED to indicate the diconnect
    }
/************** BLE **************/


    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        // print the predictions
        /*ei_printf("Predictions ");
        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        ei_printf(": \n");*/
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            /*ei_printf("    %s: %.5f\n", result.classification[ix].label,
                      result.classification[ix].value);*/
            if ((result.classification[ix].label == "marvin") && result.classification[ix].value > 0.9) {
              Serial.println("Somebody activated Marvin!");
              marvinCall = HIGH;
              //Serial.print("currentMillis marvin: ");Serial.println(currentMillis);
              previousMillis = currentMillis;
              //Serial.print("currentMillis marvin time reset: ");Serial.println(currentMillis);
            } else if (marvinCall && (result.classification[ix].label == "visual") && result.classification[ix].value > 0.8) {
              turnPlugOn();
              Serial.println("Smart plug on");
            } else if (marvinCall && (result.classification[ix].label == "zero") && result.classification[ix].value > 0.8) {
              turnPlugOff();
              Serial.println("Smart plug off");
            }

        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

        print_results = 0;
    }
/**************************************************/
  

  if (currentMillis - previousMillis >= interval) {
    //Serial.print("currentMillis check passed: ");Serial.println(currentMillis);
    // save the last time you blinked the LED
    previousMillis = currentMillis;

    // if the LED is off turn it on and vice-versa:
    if (marvinCall == HIGH) {
      Serial.println("Marvin called off");
      marvinCall = LOW;
    }
  }
/**************************************************/
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i<bytesRead>> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

    record_ready = true;

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;

    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif


/**
 * @brief      Turn the Relay On
 */
void turnPlugOn() {
    plug = HIGH;
    digitalWrite(RELAY_LED, plug);
    digitalWrite(RELAY_POWER, plug);

}

/**
 * @brief      Turn the Relay Off
 */
void turnPlugOff() {
    plug = LOW;
    digitalWrite(RELAY_LED, plug);
    digitalWrite(RELAY_POWER, plug);

}