#import RPi.GPIO as GPIO
#import spidev
from machine import Pin, SPI
import controller
from rp2 import PIO, StateMachine, asm_pio

# Configura los pines utilizados en el programa
sck_pin = Pin(18, Pin.OUT)
mosi_pin = Pin(19, Pin.OUT)
miso_pin = Pin(16, Pin.IN)
cs_pin = Pin(8, Pin.OUT)


#GPIO.setmode(GPIO.BCM)

#try:
#    GPIO.cleanup()
#except Exception as e:
#    print(e)
    


class Controller(controller.Controller):
    
    # BOARD config
    ON_BOARD_LED_PIN_NO = 'LED' # RPi's on-board LED
    ON_BOARD_LED_HIGH_IS_ON = True
    #GPIO_PINS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,)

    
    # LoRa config
    PIN_ID_FOR_LORA_RESET = None

    PIN_ID_FOR_LORA_SS = 17
    PIN_ID_SCK = 18
    PIN_ID_MOSI = 19
    PIN_ID_MISO = 20

    PIN_ID_FOR_LORA_DIO0 = 4
    PIN_ID_FOR_LORA_DIO1 = None 
    PIN_ID_FOR_LORA_DIO2 = None 
    PIN_ID_FOR_LORA_DIO3 = None
    PIN_ID_FOR_LORA_DIO4 = None
    PIN_ID_FOR_LORA_DIO5 = None 
    

    def __init__(self, 
                 pin_id_led = ON_BOARD_LED_PIN_NO, 
                 on_board_led_high_is_on = ON_BOARD_LED_HIGH_IS_ON,
                 pin_id_reset = PIN_ID_FOR_LORA_RESET,
                 blink_on_start = (2, 0.5, 0.5)):
                
        super().__init__(pin_id_led, on_board_led_high_is_on, pin_id_reset, blink_on_start)

         
    def prepare_pin(self, pin_id, in_out=Pin.OUT):
        pin = Pin(pin_id, mode=in_out)
        return pin

            

    def prepare_irq_pin(self, pin_id):
        pin = Pin(pin_id, Pin.IN)
        if pin:
            irq_handler = lambda pin: print("Interrupt occurred on pin", pin.id())
            pin.irq(trigger=Pin.IRQ_RISING, handler=irq_handler)
            return pin
            
    
    def get_spi(self):
        MOSI = 19
        SCK = 18
        MISO = 20
        spi = SPI(0, baudrate=10000000, sck=Pin(SCK), mosi=Pin(MOSI), miso=Pin(MISO))
        spi.init()
        return spi
            
    # https://www.raspberrypi.org/documentation/hardware/raspberrypi/spi/README.md
    # https://www.raspberrypi.org/forums/viewtopic.php?f=44&t=19489
    def prepare_spi(self, spi):
        class CustomSPI:
            def __init__(self, spi):
                self.spi = spi

            def transfer(self, pin_ss, address, value=0x00):
                response = bytearray(1)

                pin_ss.low()
                self.spi.write(bytes([address]))
                self.spi.readinto(response)
                pin_ss.high()

                return response

        if spi:
            new_spi = CustomSPI(spi)
            return new_spi


        
    def __exit__(self): 
        #Pin.GPIO.cleanup()
        self.spi.deinit()

        
