/* Arduino Nano 33 BLE (nRF52840): the Arduino bootloader occupies the first 64 KiB of flash and
   starts the application at 0x10000. The linker refuses an image that does not fit in the 960 KiB
   that follow.

   The last 256 bytes of the RAM hold the message of a crash across the reset (runner/src/crash.rs):
   the bootloader, which runs at every reset, clears and uses the RAM from 0x20000000 to 0x20009e14
   (its bss, noinit and data sections), far below. */
MEMORY
{
  FLASH : ORIGIN = 0x00010000, LENGTH = 1024K - 64K
  RAM   : ORIGIN = 0x20000000, LENGTH = 256K - 256
  CRASH : ORIGIN = 0x20000000 + 256K - 256, LENGTH = 256
}

SECTIONS
{
  .nanoglia_crash (NOLOAD) :
  {
    KEEP(*(.nanoglia_crash));
  } > CRASH
}

/* The stack: the 24 KiB below the crash record. The RAM store, which receives the inputs of a
   case (and its model, when it is not linked in) and holds the arena, lies between the static
   data and this limit. */
_runner_stack_limit = ORIGIN(RAM) + LENGTH(RAM) - 24K;
