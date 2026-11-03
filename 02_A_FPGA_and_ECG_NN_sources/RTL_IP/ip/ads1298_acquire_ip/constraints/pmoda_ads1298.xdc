## ADS1298 front-end H1 plugs directly into PYNQ-Z2 PmodA.
## Physical mating swaps the H1 odd/even columns relative to the PmodA
## drawing: H1-2 is on JA1P. H1 pins 9/10 = GND; 11/12 = PmodA 3.3 V.
## Use these constraints only after removing the existing PmodA IOP
## and exposing these eight ecg_* signals as top-level PL ports.
## Do not apply alongside the pmoda_rpi_gpio_tri_io[7:0] PmodA constraints.

## H1-2  ECG_MOSI  -> JA1P
set_property -dict {PACKAGE_PIN Y18 IOSTANDARD LVCMOS33} [get_ports {ecg_mosi}]
## H1-4  ECG_RESET -> JA1N
set_property -dict {PACKAGE_PIN Y19 IOSTANDARD LVCMOS33} [get_ports {ecg_reset_n}]
## H1-6  ECG_CS    -> JA2P
set_property -dict {PACKAGE_PIN Y16 IOSTANDARD LVCMOS33} [get_ports {ecg_cs_n}]
## H1-8  ECG_MISO  -> JA2N
set_property -dict {PACKAGE_PIN Y17 IOSTANDARD LVCMOS33} [get_ports {ecg_miso}]
## H1-1  ECG_PWDN  -> JA3P
set_property -dict {PACKAGE_PIN U18 IOSTANDARD LVCMOS33} [get_ports {ecg_pwdn_n}]
## H1-3  ECG_START -> JA3N
set_property -dict {PACKAGE_PIN U19 IOSTANDARD LVCMOS33} [get_ports {ecg_start}]
## H1-5  ECG_SCLK  -> JA4P
set_property -dict {PACKAGE_PIN W18 IOSTANDARD LVCMOS33} [get_ports {ecg_sclk}]
## H1-7  ECG_READY -> JA4N; ADS1298 DRDY# is active low
set_property -dict {PACKAGE_PIN W19 IOSTANDARD LVCMOS33} [get_ports {ecg_ready_n}]
