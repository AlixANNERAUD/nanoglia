//! The runner of the golden cases on an Arduino Nano 33 BLE (nRF52840, Cortex-M4F), behind its USB
//! serial port: everything the runner does not know about the board.
//!
//! | What | How |
//! | --- | --- |
//! | Link | USB CDC-ACM (embassy-usb on embassy-nrf's driver), polled by hand: no executor, no interrupt handler of ours on the path of the data. |
//! | Clock | HFCLK from the 32 MHz crystal (HFXO), which USB needs; the core runs at 64 MHz. |
//! | Cycle counter | `DWT.CYCCNT`, at the core clock. |
//! | Watchdog | TIMER1 at 1 MHz: its compare interrupt, the only one at the highest priority, fires when nothing fed it for [`WATCHDOG_US`], counts the crash and resets. Not the WDT of the chip, which cannot be stopped and would go on counting in the bootloader, where nothing feeds it. |
//! | Bootloader | `GPREGRET = 0xB0` then a system reset, what the Arduino core does when the port is opened at 1200 baud and closed (`_ontouch1200bps_` in `variants/ARDUINO_NANO33BLE/variant.cpp` of ArduinoCore-mbed); the firmware does it on the `RESET` command and on the same 1200 baud touch. |
//! | Silence | A firmware that gets no request from a host for [`NO_CONTACT_US`] after it started resets into the bootloader: one whose USB port does not work does not stay out of reach. |
//! | Crashes | The panic, fault and unexpected-interrupt handlers and the watchdog leave a message in the last bytes of the RAM, count the crash in `GPREGRET2` and reset: into the bootloader at the third in a row without a request. The count is made in the handler, so a crash in the first lines of `main` counts. |
//! | Start | The bootloader jumps here without resetting the peripherals: `main` first turns off what it left on (the PPI channels, the USB device, RTC1, UARTE0, the interrupts of CLOCK and POWER, then the two clocks, waiting until they have stopped), so that `embassy_nrf::init` starts from the state after a reset. |
//! | Flash | The firmware has no code that erases or writes a page of flash. `embassy_nrf::init` holds a routine that programs words of the UICR; as configured here (`reset-pin-as-gpio`, NFC pins and debug port left alone, no regulator voltage) every call either finds the word as wanted or gives up, and none writes. |
//!
//! Nothing here has a test on the host: it first runs on the board. The runner, which does, is in
//! `../runner`.

#![no_std]
#![no_main]

use core::fmt;
use core::future::Future;
use core::pin::{Pin, pin};
use core::task::{Context, Poll, Waker};
use cortex_m_rt::entry;
use embassy_nrf::interrupt::{self, InterruptExt, Priority};
use embassy_nrf::usb::vbus_detect::{self, HardwareVbusDetect};
use embassy_nrf::usb::{self, Driver};
use embassy_nrf::{bind_interrupts, pac, peripherals};
use embassy_usb::class::cdc_acm::{CdcAcmClass, Receiver, Sender, State};
use nanoglia_hardware_runner::boot::{self, Crash};
use nanoglia_hardware_runner::{
    Board, BoardInfo, EmbeddedModel, Fault, LinkError, Reset, Runner, crash, stack,
};

bind_interrupts!(struct Irqs {
    USBD => usb::InterruptHandler<peripherals::USBD>;
    CLOCK_POWER => vbus_detect::InterruptHandler;
});

/// `DFU_MAGIC_SERIAL_ONLY_RESET` of ArduinoCore-mbed (`variants/ARDUINO_NANO33BLE/pins_arduino.h`):
/// in `GPREGRET` at a reset, it keeps the bootloader waiting for a firmware.
const BOOTLOADER_MAGIC: u32 = 0xB0;
/// The core clock, in hertz.
const CPU_HZ: u32 = 64_000_000;
/// Time without [`feed`] after which the watchdog resets the board, in microseconds. Far above the
/// longest operator of a model that fits the board; the host waits longer than this for a run.
const WATCHDOG_US: u32 = 20_000_000;
/// Time after the start without a request from a host after which the firmware resets into the
/// bootloader, in microseconds: a firmware whose port does not work on this board gives the hand
/// back by itself. The host tool talks to the firmware within seconds of flashing it, and waits
/// longer than this before giving the board up (`NO_CONTACT_S` in tools/).
const NO_CONTACT_US: u32 = 300_000_000;
/// Time the host has to read a packet, in microseconds.
const WRITE_TIMEOUT_US: u32 = 2_000_000;
/// Time left to the host to read the last answer before a reset, in microseconds.
const RESET_DELAY_US: u32 = 100_000;
/// Time a clock has to stop once asked, in microseconds: far more than it takes.
const CLOCK_STOP_US: u32 = 100_000;
/// Size of a USB packet of the port, in bytes.
const PACKET: usize = 64;

/// The MLPerf Tiny models linked into this image: they run in place from flash, as in an
/// application.
static MODELS: &[EmbeddedModel] = &[
    #[cfg(feature = "model-kws")]
    EmbeddedModel {
        name: "kws_ref_model.tflite",
        bytes: nanoglia_hardware_runner::include_model!(
            "../../../testdata/mlperf-tiny/kws_ref_model.tflite"
        ),
    },
    #[cfg(feature = "model-resnet")]
    EmbeddedModel {
        name: "pretrainedResnet_quant.tflite",
        bytes: nanoglia_hardware_runner::include_model!(
            "../../../testdata/mlperf-tiny/pretrainedResnet_quant.tflite"
        ),
    },
    #[cfg(feature = "model-str-ww")]
    EmbeddedModel {
        name: "str_ww_ref_model.tflite",
        bytes: nanoglia_hardware_runner::include_model!(
            "../../../testdata/mlperf-tiny/str_ww_ref_model.tflite"
        ),
    },
    #[cfg(feature = "model-ad")]
    EmbeddedModel {
        name: "ad01_int8.tflite",
        bytes: nanoglia_hardware_runner::include_model!(
            "../../../testdata/mlperf-tiny/ad01_int8.tflite"
        ),
    },
    #[cfg(feature = "model-vww")]
    EmbeddedModel {
        name: "vww_96_int8.tflite",
        bytes: nanoglia_hardware_runner::include_model!(
            "../../../testdata/mlperf-tiny/vww_96_int8.tflite"
        ),
    },
];

/// Turns off the peripherals the bootloader leaves on. It jumps to the application without
/// resetting them (it only disables and clears the interrupt lines); the Arduino core does the
/// same clean-up when a sketch starts (`initVariant` in `variant.cpp`). After a reset this
/// changes nothing. Before [`timer_start`]: a PPI channel left enabled could clear or capture
/// the timer of the watchdog behind its back.
fn quiesce() {
    let ppi = pac::PPI;
    ppi.chenclr().write(|w| w.0 = !0);
    for group in 0..6 {
        ppi.tasks_chg(group).dis().write_value(1);
        ppi.chg(group).write(|w| w.0 = 0);
    }
    // The USB device of the bootloader: embassy-nrf enables it again from scratch, with the
    // workarounds of its errata, when it starts the bus.
    pac::USBD.intenclr().write(|w| w.0 = !0);
    pac::USBD.enable().write(|w| w.0 = 0);
    // CLOCK and POWER are one peripheral, with one set of interrupt enables.
    pac::CLOCK.intenclr().write(|w| w.0 = !0);
    pac::RTC1.tasks_stop().write_value(1);
    pac::RTC1.intenclr().write(|w| w.0 = !0);
    pac::RTC1.evtenclr().write(|w| w.0 = !0);
    pac::UARTE0.intenclr().write(|w| w.0 = !0);
    pac::UARTE0.enable().write(|w| w.0 = 0);
}

/// Stops both clocks and clears their events: `embassy_nrf::init` selects the source of each,
/// starts it and waits for its event. The source of the LFCLK can only be written once the clock
/// has stopped, and a clock that already runs may never give its event again. After
/// [`timer_start`], which bounds the wait (and whose timer runs on the internal oscillator).
fn stop_clocks() {
    let clock = pac::CLOCK;
    clock.tasks_lfclkstop().write_value(1);
    clock.tasks_hfclkstop().write_value(1);
    let start = now_us();
    // LFCLKSTAT.STATE (bit 16): the LFCLK runs. HFCLKSTAT.SRC (bit 0): the HFCLK is the crystal.
    while (clock.lfclkstat().read().0 & (1 << 16) != 0 || clock.hfclkstat().read().0 & 1 != 0)
        && now_us().wrapping_sub(start) < CLOCK_STOP_US
    {}
    clock.events_lfclkstarted().write_value(0);
    clock.events_hfclkstarted().write_value(0);
}

/// Starts TIMER1 as a free-running 1 MHz counter whose compare 0 is the watchdog, from whatever
/// state it is in: nothing is assumed to have its reset value.
fn timer_start() {
    let timer = pac::TIMER1;
    timer.tasks_stop().write_value(1);
    timer.shorts().write(|w| w.0 = 0);
    timer.intenclr().write(|w| w.0 = !0);
    // Timer mode, 32 bits, 16 MHz / 2^4.
    timer.mode().write(|w| w.0 = 0);
    timer.bitmode().write(|w| w.0 = 3);
    timer.prescaler().write(|w| w.0 = 4);
    timer.tasks_clear().write_value(1);
    for compare in 0..4 {
        timer.cc(compare).write_value(0);
        timer.events_compare(compare).write_value(0);
    }
    timer.cc(0).write_value(WATCHDOG_US);
    // Interrupt on compare 0 only, above every other interrupt of the firmware.
    timer.intenset().write(|w| w.0 = 1 << 16);
    interrupt::TIMER1.set_priority(Priority::P0);
    timer.tasks_start().write_value(1);
    // SAFETY: the handler below is the only user of this interrupt and shares no data.
    unsafe { cortex_m::peripheral::NVIC::unmask(pac::Interrupt::TIMER1) };
}

/// The time since [`timer_start`], in microseconds, which wraps around after 71 minutes.
fn now_us() -> u32 {
    let timer = pac::TIMER1;
    timer.tasks_capture(1).write_value(1);
    timer.cc(1).read()
}

/// Pushes the watchdog [`WATCHDOG_US`] away.
fn feed() {
    pac::TIMER1
        .cc(0)
        .write_value(now_us().wrapping_add(WATCHDOG_US));
}

/// The watchdog: nothing fed it for [`WATCHDOG_US`].
#[unsafe(no_mangle)]
unsafe extern "C" fn TIMER1() {
    pac::TIMER1.events_compare(0).write_value(0);
    die(
        Crash::Watchdog,
        format_args!("watchdog: nothing moved for {} s", WATCHDOG_US / 1_000_000),
    )
}

/// Resets the chip, into the firmware or into the bootloader.
fn reset(to: Reset) -> ! {
    cortex_m::interrupt::disable();
    let magic = match to {
        Reset::Application => 0,
        Reset::Bootloader => BOOTLOADER_MAGIC,
    };
    pac::POWER.gpregret().write(|w| w.0 = magic);
    cortex_m::peripheral::SCB::sys_reset()
}

/// Counts the crash, leaves its message for the next run and resets: into the bootloader after
/// [`boot::MAX_FAILED_BOOTS`] crashes in a row without a request of the host.
fn die(kind: Crash, message: fmt::Arguments<'_>) -> ! {
    cortex_m::interrupt::disable();
    crash::store(message);
    let verdict = boot::crash(pac::POWER.gpregret2().read().0 as u8, kind);
    pac::POWER
        .gpregret2()
        .write(|w| w.0 = u32::from(verdict.retained));
    reset(verdict.to)
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    die(Crash::Panic, format_args!("{info}"))
}

#[cortex_m_rt::exception]
unsafe fn HardFault(frame: &cortex_m_rt::ExceptionFrame) -> ! {
    die(
        Crash::Fault,
        format_args!(
            "hard fault at pc=0x{:08x}, lr=0x{:08x}",
            frame.pc(),
            frame.lr()
        ),
    )
}

/// An interrupt nothing handles (one the bootloader left pending, or a bug): a crash like the
/// others, not the endless loop of the default handler.
#[cortex_m_rt::exception]
unsafe fn DefaultHandler(irqn: i16) -> ! {
    die(Crash::Fault, format_args!("unexpected interrupt {irqn}"))
}

/// Polls a future once. The futures of embassy check their condition at each poll: nothing here
/// waits for a waker.
fn poll<F: Future + ?Sized>(future: Pin<&mut F>) -> Poll<F::Output> {
    future.poll(&mut Context::from_waker(Waker::noop()))
}

type UsbDriver<'d> = Driver<'d, HardwareVbusDetect>;

/// The board: its USB port and its counters.
struct Nano<'f, 'd, F: Future> {
    /// The future of `UsbDevice::run`, which serves the control requests of the host.
    usb: Pin<&'f mut F>,
    sender: Sender<'d, UsbDriver<'d>>,
    receiver: Receiver<'d, UsbDriver<'d>>,
    /// The last packet received, of which `packet_len` bytes, read up to `packet_at`.
    packet: [u8; PACKET],
    packet_len: usize,
    packet_at: usize,
    reset_reason: u32,
    /// Whether a host sent a request since the firmware started: the runner says so.
    contacted: bool,
    /// What decides the times besides the clock: the state of the instruction cache.
    settings: &'static str,
    /// Whether `DWT.CYCCNT` counts.
    cycle_counter: bool,
}

/// Serves the USB device, and resets into the bootloader on the 1200 baud touch of Arduino: the
/// port set to 1200 baud, then closed.
fn serve_usb<F: Future>(usb: &mut Pin<&mut F>, baud: u32, dtr: bool) {
    let _ = poll(usb.as_mut());
    if baud == 1200 && !dtr {
        reset(Reset::Bootloader);
    }
}

impl<F: Future> Nano<'_, '_, F> {
    /// Serves the USB device for `duration_us`.
    fn serve_for(&mut self, duration_us: u32) {
        let start = now_us();
        while now_us().wrapping_sub(start) < duration_us {
            serve_usb(
                &mut self.usb,
                self.sender.line_coding().data_rate(),
                self.sender.dtr(),
            );
        }
    }
}

impl<F: Future> Board for Nano<'_, '_, F> {
    fn info(&self) -> BoardInfo {
        BoardInfo {
            board: "Arduino Nano 33 BLE (nRF52840, Cortex-M4F)",
            firmware: concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION")),
            clock: if self.cycle_counter {
                "DWT cycle counter, core clock of 64 MHz from the 32 MHz crystal (HFXO)"
            } else {
                "DWT cycle counter not running: no timing"
            },
            counter_hz: if self.cycle_counter { CPU_HZ } else { 0 },
            reset_reason: self.reset_reason,
            settings: self.settings,
        }
    }

    fn read(&mut self, buffer: &mut [u8], timeout_ms: Option<u32>) -> Result<usize, LinkError> {
        let start = now_us();
        while self.packet_at == self.packet_len {
            let received = {
                let mut read = pin!(self.receiver.read_packet(&mut self.packet));
                loop {
                    // Waiting for the host is no hang.
                    feed();
                    serve_usb(
                        &mut self.usb,
                        self.sender.line_coding().data_rate(),
                        self.sender.dtr(),
                    );
                    // Checked at every turn, before the port is asked: while the host has not
                    // configured the device, the read fails at once, and this loop turns on it.
                    let now = now_us();
                    if !self.contacted && now >= NO_CONTACT_US {
                        reset(Reset::Bootloader);
                    }
                    let waited = now.wrapping_sub(start);
                    if timeout_ms.is_some_and(|timeout| waited >= timeout.saturating_mul(1000)) {
                        return Err(LinkError::Timeout);
                    }
                    if let Poll::Ready(received) = poll(read.as_mut()) {
                        break received;
                    }
                }
            };
            // An error is a port that is not configured yet, or was closed: nothing received.
            self.packet_len = received.unwrap_or(0);
            self.packet_at = 0;
        }
        let available = &self.packet[self.packet_at..self.packet_len];
        let count = available.len().min(buffer.len());
        buffer[..count].copy_from_slice(&available[..count]);
        self.packet_at += count;
        Ok(count)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), LinkError> {
        // A transfer ends with a short packet: an empty one after a last full packet.
        let end: &[u8] = &[];
        let last = bytes.len().is_multiple_of(PACKET).then_some(end);
        for chunk in bytes.chunks(PACKET).chain(last) {
            let start = now_us();
            let mut write = pin!(self.sender.write_packet(chunk));
            loop {
                feed();
                serve_usb(
                    &mut self.usb,
                    self.receiver.line_coding().data_rate(),
                    self.receiver.dtr(),
                );
                match poll(write.as_mut()) {
                    Poll::Ready(Ok(())) => break,
                    Poll::Ready(Err(_)) => return Err(LinkError::Timeout),
                    Poll::Pending if now_us().wrapping_sub(start) >= WRITE_TIMEOUT_US => {
                        return Err(LinkError::Timeout);
                    }
                    Poll::Pending => {}
                }
            }
        }
        Ok(())
    }

    fn cycles(&mut self) -> u32 {
        cortex_m::peripheral::DWT::cycle_count()
    }

    fn idle(&mut self) {
        feed();
        serve_usb(
            &mut self.usb,
            self.sender.line_coding().data_rate(),
            self.sender.dtr(),
        );
    }

    fn reset(&mut self, to: Reset) {
        // The answer is with the host once its packet is acknowledged; leave it the time to read
        // it before the port goes away.
        self.serve_for(RESET_DELAY_US);
        reset(to)
    }

    fn fault(&mut self, fault: Fault) {
        self.serve_for(RESET_DELAY_US);
        match fault {
            Fault::Panic => panic!("fault requested by the host"),
            Fault::HardFault => cortex_m::asm::udf(),
            // The watchdog ends it.
            Fault::Hang => loop {
                cortex_m::asm::nop();
            },
        }
    }

    fn contact(&mut self) {
        self.contacted = true;
    }

    fn retained(&mut self) -> u8 {
        pac::POWER.gpregret2().read().0 as u8
    }

    fn set_retained(&mut self, value: u8) {
        pac::POWER.gpregret2().write(|w| w.0 = u32::from(value));
    }

    fn stack_paint(&mut self) {
        stack::paint();
    }

    fn stack_peak(&mut self) -> Option<u32> {
        Some(stack::peak())
    }
}

unsafe extern "C" {
    static mut __sheap: u8;
    static _runner_stack_limit: u8;
}

#[entry]
fn main() -> ! {
    // The state a reset leaves, whatever the bootloader left, then the watchdog, before anything
    // that can wait: a start-up that hangs resets, and at the third reset ends in the bootloader.
    quiesce();
    timer_start();
    stop_clocks();

    // Why the chip reset; the bits are cleared by writing them.
    let reset_reason = pac::POWER.resetreas().read().0;
    pac::POWER.resetreas().write(|w| w.0 = reset_reason);

    let mut config = embassy_nrf::config::Config::default();
    // USB needs the crystal; `init` waits for it.
    config.hfclk_source = embassy_nrf::config::HfclkSource::ExternalXtal;
    // Leave the debug port as it is: the default would write `UICR.APPROTECT`.
    config.debug = embassy_nrf::config::Debug::NotConfigured;
    let p = embassy_nrf::init(config);

    // The instruction cache of the flash: off after a reset of the chip, but the Arduino bootloader
    // turns it on before it starts the application (`nordicsemi_nrf52_init` of its Zephyr image),
    // and nothing here turns it off: without the feature `icache`, the firmware runs with the state
    // it inherits, on after that bootloader. The state read here is the one reported.
    if cfg!(feature = "icache") {
        pac::NVMC.icachecnf().write(|w| w.0 = 1);
    }
    let settings = if pac::NVMC.icachecnf().read().0 & 1 == 1 {
        "instruction cache of the flash on (NVMC ICACHECNF)"
    } else {
        "instruction cache of the flash off (NVMC ICACHECNF, as after a reset)"
    };

    // SAFETY: embassy-nrf does not use the DCB nor the DWT, and nothing else does.
    let mut core = unsafe { cortex_m::Peripherals::steal() };
    core.DCB.enable_trace();
    cortex_m::peripheral::DWT::unlock();
    core.DWT.enable_cycle_counter();
    let before = cortex_m::peripheral::DWT::cycle_count();
    cortex_m::asm::delay(64);
    let cycle_counter = cortex_m::peripheral::DWT::cycle_count() != before;

    // The serial number of the port: the identifier of the chip.
    let mut serial = [0u8; 16];
    let id =
        (u64::from(pac::FICR.deviceid(1).read()) << 32) | u64::from(pac::FICR.deviceid(0).read());
    for (index, digit) in serial.iter_mut().enumerate() {
        *digit = b"0123456789ABCDEF"[((id >> (60 - 4 * index)) & 0xF) as usize];
    }

    // Below the watchdog, which must be able to interrupt them.
    interrupt::USBD.set_priority(Priority::P2);
    interrupt::CLOCK_POWER.set_priority(Priority::P2);
    let driver = Driver::new(p.USBD, Irqs, HardwareVbusDetect::new(Irqs));
    // The test identifiers of pid.codes: this firmware is a test bench, not a product.
    let mut config = embassy_usb::Config::new(0x1209, 0x0001);
    config.manufacturer = Some("nanoglia");
    config.product = Some("nanoglia hardware runner");
    config.serial_number = core::str::from_utf8(&serial).ok();
    config.max_power = 100;
    config.max_packet_size_0 = 64;
    let mut config_descriptor = [0; 256];
    let mut bos_descriptor = [0; 256];
    let mut msos_descriptor = [0; 256];
    let mut control = [0; 64];
    let mut state = State::new();
    let mut builder = embassy_usb::Builder::new(
        driver,
        config,
        &mut config_descriptor,
        &mut bos_descriptor,
        &mut msos_descriptor,
        &mut control,
    );
    let (sender, receiver) = CdcAcmClass::new(&mut builder, &mut state, PACKET as u16).split();
    let mut usb = builder.build();
    let usb = pin!(usb.run());

    // The RAM store: from the end of the static data, on a 16-byte boundary, to the stack.
    let start = (&raw mut __sheap).addr().next_multiple_of(16);
    let len = (&raw const _runner_stack_limit).addr() - start;
    // SAFETY: the linker script leaves this memory to no one else, and `main` runs once.
    let ram = unsafe {
        let start = (&raw mut __sheap).with_addr(start);
        start.write_bytes(0, len);
        core::slice::from_raw_parts_mut(start, len)
    };

    let board = Nano {
        usb,
        sender,
        receiver,
        packet: [0; PACKET],
        packet_len: 0,
        packet_at: 0,
        reset_reason,
        contacted: false,
        settings,
        cycle_counter,
    };
    let mut runner = Runner::new(board, ram, MODELS);
    runner.serve();
    // No board comes here.
    reset(Reset::Application)
}
