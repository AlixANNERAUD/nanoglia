/* Emulated Cortex-M memory, for cortex-m-rt's link.x. The RAM is laid out as the Arduino Nano 33
   BLE's (nano33ble/memory.x), so that a case that fits here fits there; the flash is larger, to
   hold every model at once. tools/ maps the regions from the `_emulator_*` symbols. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   : ORIGIN = 0x20000000, LENGTH = 256K - 256
  /* The message of a crash, kept across a reset (runner/src/crash.rs). */
  CRASH : ORIGIN = 0x20000000 + 256K - 256, LENGTH = 256
}

SECTIONS
{
  .nanoglia_crash (NOLOAD) :
  {
    KEEP(*(.nanoglia_crash));
  } > CRASH
}

_emulator_flash_start = ORIGIN(FLASH);
_emulator_flash_end = ORIGIN(FLASH) + LENGTH(FLASH);
_emulator_ram_start = ORIGIN(RAM);
_emulator_ram_end = ORIGIN(CRASH) + LENGTH(CRASH);

/* The stack: the 24 KiB below the crash record, as on the board. The RAM store lies between the
   static data and this limit. */
_runner_stack_limit = ORIGIN(RAM) + LENGTH(RAM) - 24K;
