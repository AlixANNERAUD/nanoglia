/* Arduino Nano 33 BLE (nRF52840): the Arduino bootloader occupies the first 64 KiB of flash and
   starts the application at 0x10000. The linker refuses an image that does not fit in the 960 KiB
   that follow; nothing in the firmware writes to flash. */
MEMORY
{
  FLASH : ORIGIN = 0x00010000, LENGTH = 1024K - 64K
  RAM   : ORIGIN = 0x20000000, LENGTH = 256K
}

/* The stack: the last 24 KiB of the RAM. The RAM store, which receives the inputs of a case (and
   its model, when it is not linked in) and holds the arena, lies between the static data and this
   limit. */
_runner_stack_limit = ORIGIN(RAM) + LENGTH(RAM) - 24K;
