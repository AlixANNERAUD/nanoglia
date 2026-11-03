# Keyword spotting on the Arduino Nano 33 BLE

Runs the MLPerf Tiny keyword-spotting model (DS-CNN, 13 operators) with nanoglia on the nRF52840
of an Arduino Nano 33 BLE, from a static 32 KiB arena (no heap), and prints over RTT the memory
used, the kernel chosen for each operator (`DEFMT_LOG=debug`) and the inference time.

```sh
mise run example-nano33ble        # build and print the firmware size
cd examples/nano33ble
cargo run --release               # flash and run through a debug probe (probe-rs) on the SWD pads
```

The application is linked at `0x10000`, after the Arduino bootloader, which is left intact.
