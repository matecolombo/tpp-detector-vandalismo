from time import sleep
from machine import Pin, SPI
            
class Controller:

    #class Mock:
      #  pass        

    ON_BOARD_LED_PIN_NO = 'LED'
    ON_BOARD_LED_HIGH_IS_ON = True
    GPIO_PINS = []
                 
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
        
        self.pin_led = self.prepare_pin(pin_id_led)
        self.on_board_led_high_is_on = on_board_led_high_is_on
        #self.pin_reset = self.prepare_pin(pin_id_reset)        
        #self.reset_pin(self.pin_reset)
        self.spi = self.prepare_spi(self.get_spi())        
        self.transceivers = {}
        self.blink_led(*blink_on_start) 
        

    def add_transceiver(self, 
                        transceiver, 
                        pin_id_ss = PIN_ID_FOR_LORA_SS,
                        pin_id_RxDone = PIN_ID_FOR_LORA_DIO0):
                        #pin_id_RxTimeout = PIN_ID_FOR_LORA_DIO1,
                        #pin_id_ValidHeader = PIN_ID_FOR_LORA_DIO2,
                        #pin_id_CadDone = PIN_ID_FOR_LORA_DIO3,     
                        #pin_id_CadDetected = PIN_ID_FOR_LORA_DIO4,
                        #pin_id_PayloadCrcError = PIN_ID_FOR_LORA_DIO5):
        
        transceiver.transfer = self.spi.transfer
        transceiver.blink_led = self.blink_led
        
        transceiver.pin_ss = self.prepare_pin(pin_id_ss)
        transceiver.pin_RxDone = self.prepare_irq_pin(pin_id_RxDone)
        #transceiver.pin_RxTimeout = self.prepare_irq_pin(pin_id_RxTimeout)
        #transceiver.pin_ValidHeader = self.prepare_irq_pin(pin_id_ValidHeader)
        #transceiver.pin_CadDone = self.prepare_irq_pin(pin_id_CadDone)
        #transceiver.pin_CadDetected = self.prepare_irq_pin(pin_id_CadDetected)
        #transceiver.pin_PayloadCrcError = self.prepare_irq_pin(pin_id_PayloadCrcError)
        
        transceiver.init()        
        self.transceivers[transceiver.name] = transceiver 
        return transceiver
        
                 
    def prepare_pin(self, pin_id, in_out=Pin.OUT):
        pin = Pin(pin_id, mode=in_out)
        return new_pin

    def prepare_irq_pin(self, pin_id):
        pin = self.prepare_pin(pin_id, Pin.IN)
        if pin:
            pin.set_handler_for_irq_on_rising_edge = lambda handler: pin.irq(trigger=Pin.IRQ_RISING, handler=handler)
            pin.detach_irq = lambda: pin.irq(trigger=0)
            return pin
        
        
    def get_spi(self):
        MOSI = 19
        SCK = 18
        MISO = 20
        spi = SPI(0, baudrate=10000000, sck=Pin(SCK), mosi=Pin(MOSI), miso=Pin(MISO))
        spi.init()
        print(spi)
        return spi
    
        
    def prepare_spi(self, spi):
        
        if spi:
            new_spi = spi

            def transfer(pin_ss, address, value=0x00):
                response = bytearray(1)

                pin_ss.low()
                spi.write(bytes([address]))
                spi.readinto(response)
                pin_ss.high()

                return response

            new_spi.transfer = transfer
            return new_spi


    def led_on(self, on = True):
        self.pin_led.high() if self.on_board_led_high_is_on == on else self.pin_led.low()
            

    def blink_led(self, times = 1, on_seconds = 0.1, off_seconds = 0.1):
        for i in range(times):
            self.led_on(True)
            sleep(on_seconds)
            self.led_on(False)
            sleep(off_seconds) 
            

    def reset_pin(self, pin, duration_low = 0.05, duration_high = 0.05):
        pin.low()
        sleep(duration_low)
        pin.high()
        sleep(duration_high)
        
        
    def __exit__(self): 
        self.spi.close()        
