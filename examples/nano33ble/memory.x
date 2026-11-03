/* Arduino Nano 33 BLE (nRF52840): the Arduino bootloader occupies the first 64 KiB of flash and
   starts the application at 0x10000. */
MEMORY
{
  FLASH : ORIGIN = 0x00010000, LENGTH = 1024K - 64K
  RAM   : ORIGIN = 0x20000000, LENGTH = 256K
}
