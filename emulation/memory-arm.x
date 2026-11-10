/* Emulated Cortex-M memory, for cortex-m-rt's link.x. No real chip: the sizes only have to hold
   the firmware, the largest golden case and its arena. tools/run.py maps every region from the
   `_emulator_*`, `_arena_*` and `_case_*` symbols, so this file is the only description of it. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  /* Statics and the stack, which grows down from the end. */
  RAM   : ORIGIN = 0x20000000, LENGTH = 1M
}

_emulator_flash_start = ORIGIN(FLASH);
_emulator_flash_end = ORIGIN(FLASH) + LENGTH(FLASH);
_emulator_ram_start = ORIGIN(RAM);
_emulator_ram_end = ORIGIN(RAM) + LENGTH(RAM);

/* The arena of the graph and its execution context: not zeroed by the start-up code. */
_arena_start = 0x20100000;
_arena_end = 0x20500000;

/* The case written by tools/run.py before the reset: header, model and inputs. */
_case_start = 0x60000000;
_case_end = 0x60200000;
