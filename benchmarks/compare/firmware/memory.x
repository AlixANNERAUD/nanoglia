/* nRF52840 (Arduino Nano 33 BLE), without bootloader: 1 MiB of flash, 256 KiB of RAM.
   cortex-m-rt places the stack at the end of RAM, growing down towards .bss. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 1024K
  RAM   : ORIGIN = 0x20000000, LENGTH = 256K
}
