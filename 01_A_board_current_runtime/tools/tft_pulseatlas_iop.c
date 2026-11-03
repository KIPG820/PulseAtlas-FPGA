/*
 * Landscape ST7789 + GT911 service for the PYNQ-Z2 Arduino IOP.
 *
 * The IOP accepts RGB565 frame chunks through its shared BRAM mailbox, sends
 * them to the panel over the isolated AXI Quad SPI trial path, then polls the
 * GT911 over I2C on the same D4/D5 wires. GT911 is held in reset during an
 * LCD transfer, then restarted before touch polling resumes.
 */
#include <stdint.h>
#include "xil_io.h"
#include "xparameters.h"
#include "xio_switch.h"

#define GPIO_DATA_ADDR       0x40020000u
#define GPIO_TRI_ADDR        0x40020004u
#define TFT_SPI_BASE         0x44A10000u
#define TFT_SPI_RESET        (TFT_SPI_BASE + 0x40u)
#define TFT_SPI_CR           (TFT_SPI_BASE + 0x60u)
#define TFT_SPI_SR           (TFT_SPI_BASE + 0x64u)
#define TFT_SPI_DTR          (TFT_SPI_BASE + 0x68u)
#define TFT_SPI_DRR          (TFT_SPI_BASE + 0x6Cu)
#define TFT_SPI_SSR          (TFT_SPI_BASE + 0x70u)
#define TFT_SPI_ENABLE      0x002u
#define TFT_SPI_MASTER      0x004u
#define TFT_SPI_TXRST       0x020u
#define TFT_SPI_RXRST       0x040u
#define TFT_SPI_MANUAL_SS   0x080u
#define TFT_SPI_INHIBIT     0x100u
#define TFT_SPI_RX_EMPTY    0x001u
#define TFT_SPI_TX_EMPTY    0x004u
#define TFT_SPI_TX_FULL     0x008u
#define MB_COMMAND_ADDR      0x0000F000u
#define MB_STATUS_ADDR       0x0000F004u
#define MB_PAYLOAD_ADDR      0x0000F008u

#define PIN_LCD_SCK          (1u << 5)   /* T15 / Arduino D5 */
#define PIN_LCD_MOSI         (1u << 4)   /* V15 / Arduino D4 */
#define PIN_LCD_CS           (1u << 17)  /* V11 / Arduino A3 */
#define PIN_TOUCH_RST        (1u << 1)   /* U12 / Arduino D1 */
#define PIN_TOUCH_INT        (1u << 2)   /* U13 / Arduino D2 */
#define PIN_LCD_RST          (1u << 14)  /* Y11 / Arduino A0 */
#define PIN_LCD_DC           (1u << 16)  /* W11 / Arduino A2 */
#define PIN_LCD_BL           (1u << 19)  /* U10 / Arduino A5 */
#define PIN_I2C_SDA          PIN_LCD_MOSI
#define PIN_I2C_SCL          PIN_LCD_SCK

#define CMD_FRAME_BEGIN      0x46524247u /* FRBG */
#define CMD_FRAME_CHUNK      0x43484E4Bu /* CHNK */
#define CMD_FRAME_RECT       0x52454354u /* RECT */
#define CMD_FRAME_END        0x4652454Eu /* FREN */
#define STATE_READY          0x52454144u /* READ */
#define STATE_ACK_BEGIN      0x41434B31u /* ACK1 */
#define STATE_ACK_CHUNK      0x41434B32u /* ACK2 */
#define STATE_ACK_RECT       0x41434B34u /* ACK4 */
#define STATE_ACK_END        0x41434B33u /* ACK3 */
#define STATE_TOUCH_EVENT    0x544F5543u /* TOUC */

#define GT911_ADDRESS        0x5Du
#define GT911_STATUS_REG     0x814Eu
#define GT911_POINT_REG      0x814Fu

static uint32_t gpio_data;
static uint32_t gpio_tri;
static int touch_latched;
static uint8_t gt911_address = GT911_ADDRESS;
static unsigned consecutive_touch_read_failures;
static unsigned spi_bytes_sent;
static int spi_routed;
static int spi_fault;
static volatile uint32_t *const mb_command =
    (volatile uint32_t *)MB_COMMAND_ADDR;
static volatile uint32_t *const mb_status =
    (volatile uint32_t *)MB_STATUS_ADDR;
static volatile uint32_t *const mb_payload =
    (volatile uint32_t *)MB_PAYLOAD_ADDR;

static void delay_cycles(volatile uint32_t n)
{
    while (n--) __asm__ volatile ("nop");
}

static void set_gpio(uint32_t mask, int high)
{
    gpio_data = high ? (gpio_data | mask) : (gpio_data & ~mask);
    Xil_Out32(GPIO_DATA_ADDR, gpio_data);
}

static void set_direction(uint32_t mask, int output)
{
    gpio_tri = output ? (gpio_tri & ~mask) : (gpio_tri | mask);
    Xil_Out32(GPIO_TRI_ADDR, gpio_tri);
}

static int read_gpio(uint32_t mask)
{
    return (Xil_In32(GPIO_DATA_ADDR) & mask) != 0;
}

static void drive_low(uint32_t mask)
{
    set_gpio(mask, 0);
    set_direction(mask, 1);
}

static void release_line(uint32_t mask)
{
    set_gpio(mask, 1);
    set_direction(mask, 0);
}

static void spi_byte_gpio(uint8_t value)
{
    /* Two GPIO writes per SPI bit. Keep the output latch coherent for the
     * other pins that share this GPIO register. */
    uint32_t base = gpio_data & ~(PIN_LCD_MOSI | PIN_LCD_SCK);
    uint32_t last = base;
    for (int bit = 7; bit >= 0; --bit) {
        last = base | (((value >> bit) & 1u) ? PIN_LCD_MOSI : 0u);
        Xil_Out32(GPIO_DATA_ADDR, last);
        Xil_Out32(GPIO_DATA_ADDR, last | PIN_LCD_SCK);
    }
    gpio_data = last;
    Xil_Out32(GPIO_DATA_ADDR, last);
}

static void init_tft_spi(void)
{
    Xil_Out32(TFT_SPI_RESET, 0x0Au);
    Xil_Out32(TFT_SPI_CR, TFT_SPI_ENABLE | TFT_SPI_MASTER |
              TFT_SPI_MANUAL_SS | TFT_SPI_INHIBIT |
              TFT_SPI_TXRST | TFT_SPI_RXRST);
    Xil_Out32(TFT_SPI_SSR, 0xFFFFFFFEu);
    spi_bytes_sent = 0;
    spi_routed = 0;
    spi_fault = 0;
}

static void spi_enter(void)
{
    if (spi_routed || spi_fault) return;
    /* GT911 is held in reset for the whole frame. Only LCD SCK/MOSI switch;
     * CS/DC and touch reset remain controlled by the verified GPIO path. */
    set_gpio(PIN_LCD_SCK, 0);
    set_gpio(PIN_LCD_MOSI, 1);
    set_pin(4, MOSI0);
    set_pin(5, SPICLK0);
    Xil_Out32(TFT_SPI_CR, TFT_SPI_ENABLE | TFT_SPI_MASTER |
              TFT_SPI_MANUAL_SS);
    spi_routed = 1;
}

static void spi_drain_rx(void)
{
    unsigned limit = 512;
    while (limit-- && !(Xil_In32(TFT_SPI_SR) & TFT_SPI_RX_EMPTY))
        (void)Xil_In32(TFT_SPI_DRR);
}

static void spi_fallback(void)
{
    /* Leave the shared wires in the proven GPIO mode on a stalled core. */
    Xil_Out32(TFT_SPI_CR, TFT_SPI_ENABLE | TFT_SPI_MASTER |
              TFT_SPI_MANUAL_SS | TFT_SPI_INHIBIT);
    set_gpio(PIN_LCD_SCK, 0);
    set_gpio(PIN_LCD_MOSI, 1);
    set_pin(4, GPIO);
    set_pin(5, GPIO);
    spi_routed = 0;
    spi_fault = 1;
}

static void spi_byte_hw(uint8_t value)
{
    if (spi_fault) {
        spi_byte_gpio(value);
        return;
    }
    unsigned limit = 1000000;
    while (Xil_In32(TFT_SPI_SR) & TFT_SPI_TX_FULL) {
        if (limit-- == 0) {
            spi_fallback();
            spi_byte_gpio(value);
            return;
        }
        spi_drain_rx();
    }
    Xil_Out32(TFT_SPI_DTR, value);
    if ((++spi_bytes_sent & 63u) == 0u)
        spi_drain_rx();
}

static void spi_leave(void)
{
    if (!spi_routed) return;
    unsigned limit = 1000000;
    while (!(Xil_In32(TFT_SPI_SR) & TFT_SPI_TX_EMPTY)) {
        if (limit-- == 0) {
            spi_fallback();
            return;
        }
        spi_drain_rx();
    }
    /* TX_EMPTY can precede the final byte leaving the shift register. */
    delay_cycles(1000);
    Xil_Out32(TFT_SPI_CR, TFT_SPI_ENABLE | TFT_SPI_MASTER |
              TFT_SPI_MANUAL_SS | TFT_SPI_INHIBIT);
    set_gpio(PIN_LCD_SCK, 0);
    set_gpio(PIN_LCD_MOSI, 1);
    set_pin(4, GPIO);
    set_pin(5, GPIO);
    spi_drain_rx();
    spi_routed = 0;
}

static void command(uint8_t cmd, const uint8_t *data, unsigned length)
{
    set_gpio(PIN_LCD_CS, 0);
    set_gpio(PIN_LCD_DC, 0);
    spi_byte_gpio(cmd);
    if (length) {
        set_gpio(PIN_LCD_DC, 1);
        for (unsigned i = 0; i < length; ++i) spi_byte_gpio(data[i]);
    }
    set_gpio(PIN_LCD_CS, 1);
}

static void set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1)
{
    uint8_t col[4] = {x0 >> 8, x0, x1 >> 8, x1};
    uint8_t row[4] = {y0 >> 8, y0, y1 >> 8, y1};
    command(0x2A, col, 4);
    command(0x2B, row, 4);
}

static void init_panel_landscape(void)
{
    uint8_t p;
    set_gpio(PIN_LCD_RST, 0);
    delay_cycles(5000000);
    set_gpio(PIN_LCD_RST, 1);
    delay_cycles(12000000);
    command(0x11, 0, 0);             /* SLPOUT */
    delay_cycles(12000000);
    p = 0x55; command(0x3A, &p, 1);  /* RGB565 */
    p = 0x60; command(0x36, &p, 1);  /* MX + MV: 320x240 landscape */
    command(0x29, 0, 0);             /* DISPON */
    delay_cycles(1000000);
}

static void fill_rect(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1,
                      uint16_t color)
{
    unsigned pixels = (unsigned)(x1 - x0 + 1) * (unsigned)(y1 - y0 + 1);
    set_window(x0, y0, x1, y1);
    set_gpio(PIN_LCD_CS, 0);
    set_gpio(PIN_LCD_DC, 0);
    spi_byte_gpio(0x2C);
    set_gpio(PIN_LCD_DC, 1);
    while (pixels--) {
        spi_byte_gpio((uint8_t)(color >> 8));
        spi_byte_gpio((uint8_t)color);
    }
    set_gpio(PIN_LCD_CS, 1);
}

static int begin_rectangle(void)
{
    volatile uint32_t *rect = (volatile uint32_t *)MB_PAYLOAD_ADDR;
    uint32_t x0 = rect[0], y0 = rect[1], x1 = rect[2], y1 = rect[3];
    if (x0 > x1 || x1 >= 320u || y0 > y1 || y1 >= 240u) return 0;
    set_gpio(PIN_LCD_CS, 1);
    set_window((uint16_t)x0, (uint16_t)y0, (uint16_t)x1, (uint16_t)y1);
    set_gpio(PIN_LCD_CS, 0);
    set_gpio(PIN_LCD_DC, 0);
    spi_byte_gpio(0x2C);
    set_gpio(PIN_LCD_DC, 1);
    return 1;
}

static void touch_select_5d(void)
{
    drive_low(PIN_TOUCH_INT);
    drive_low(PIN_TOUCH_RST);
    delay_cycles(100000);  /* >5 ms at the measured IOP loop rate */
    release_line(PIN_TOUCH_RST);
    delay_cycles(100000);  /* Keep INT strapped during reset release */
    release_line(PIN_TOUCH_INT);
    delay_cycles(500000);  /* >50 ms before first I2C transaction */
}

static void i2c_delay(void)
{
    delay_cycles(120);
}

static void i2c_start(void)
{
    release_line(PIN_I2C_SDA);
    release_line(PIN_I2C_SCL);
    i2c_delay();
    drive_low(PIN_I2C_SDA);
    i2c_delay();
    drive_low(PIN_I2C_SCL);
    i2c_delay();
}

static void i2c_stop(void)
{
    drive_low(PIN_I2C_SDA);
    i2c_delay();
    release_line(PIN_I2C_SCL);
    i2c_delay();
    release_line(PIN_I2C_SDA);
    i2c_delay();
}

static int i2c_write_byte(uint8_t value)
{
    for (int bit = 7; bit >= 0; --bit) {
        if ((value >> bit) & 1) release_line(PIN_I2C_SDA);
        else drive_low(PIN_I2C_SDA);
        i2c_delay();
        release_line(PIN_I2C_SCL);
        i2c_delay();
        drive_low(PIN_I2C_SCL);
        i2c_delay();
    }
    release_line(PIN_I2C_SDA);
    i2c_delay();
    release_line(PIN_I2C_SCL);
    i2c_delay();
    int ack = !read_gpio(PIN_I2C_SDA);
    drive_low(PIN_I2C_SCL);
    i2c_delay();
    return ack;
}

static uint8_t i2c_read_byte(int ack)
{
    uint8_t value = 0;
    release_line(PIN_I2C_SDA);
    for (unsigned bit = 0; bit < 8; ++bit) {
        release_line(PIN_I2C_SCL);
        i2c_delay();
        value = (uint8_t)((value << 1) | read_gpio(PIN_I2C_SDA));
        drive_low(PIN_I2C_SCL);
        i2c_delay();
    }
    if (ack) drive_low(PIN_I2C_SDA);
    else release_line(PIN_I2C_SDA);
    i2c_delay();
    release_line(PIN_I2C_SCL);
    i2c_delay();
    drive_low(PIN_I2C_SCL);
    i2c_delay();
    release_line(PIN_I2C_SDA);
    return value;
}

static int gt911_read(uint16_t reg, uint8_t *data, unsigned length)
{
    i2c_start();
    if (!i2c_write_byte((uint8_t)(gt911_address << 1)) ||
        !i2c_write_byte((uint8_t)(reg >> 8)) ||
        !i2c_write_byte((uint8_t)reg)) {
        i2c_stop();
        return 0;
    }
    i2c_start();
    if (!i2c_write_byte((uint8_t)((gt911_address << 1) | 1))) {
        i2c_stop();
        return 0;
    }
    for (unsigned i = 0; i < length; ++i)
        data[i] = i2c_read_byte(i + 1 < length);
    i2c_stop();
    return 1;
}

static int gt911_write(uint16_t reg, uint8_t value)
{
    i2c_start();
    int ok = i2c_write_byte((uint8_t)(gt911_address << 1)) &&
             i2c_write_byte((uint8_t)(reg >> 8)) &&
             i2c_write_byte((uint8_t)reg) &&
             i2c_write_byte(value);
    i2c_stop();
    return ok;
}

static void poll_touch(void)
{
    uint8_t status;
    if (*mb_status != 0) return;
    if (!gt911_read(GT911_STATUS_REG, &status, 1)) {
        uint8_t previous = gt911_address;
        gt911_address = previous == 0x5D ? 0x14 : 0x5D;
        if (!gt911_read(GT911_STATUS_REG, &status, 1)) {
            gt911_address = previous;
            if (++consecutive_touch_read_failures >= 10u) {
                release_line(PIN_I2C_SDA);
                release_line(PIN_I2C_SCL);
                touch_select_5d();
                gt911_address = GT911_ADDRESS;
                consecutive_touch_read_failures = 0;
            }
            return;
        }
    }
    consecutive_touch_read_failures = 0;
    if (!(status & 0x80)) return;

    unsigned count = status & 0x0F;
    if (count > 5) count = 5;
    if (count && !touch_latched) {
        uint8_t point[8];
        if (gt911_read(GT911_POINT_REG, point, sizeof(point))) {
            uint16_t x = (uint16_t)point[1] | ((uint16_t)point[2] << 8);
            uint16_t y = (uint16_t)point[3] | ((uint16_t)point[4] << 8);
            if (x < 240u && y < 320u) {
                *mb_payload = point[0];
                *(volatile uint32_t *)(MB_PAYLOAD_ADDR + 4) = x;
                *(volatile uint32_t *)(MB_PAYLOAD_ADDR + 8) = y;
                *mb_status = STATE_TOUCH_EVENT;
            }
        }
    }
    touch_latched = count != 0;
    gt911_write(GT911_STATUS_REG, 0x00);
}

static void acknowledge(uint32_t value)
{
    *mb_command = 0;
    *mb_status = value;
}

int main(void)
{
    init_io_switch();
    /* The panel's first SPI bit needs SCK low before its first rising edge. */
    gpio_data = PIN_LCD_CS | PIN_LCD_DC;
    gpio_tri = ~(PIN_LCD_SCK | PIN_LCD_MOSI | PIN_LCD_CS | PIN_TOUCH_RST |
                 PIN_TOUCH_INT | PIN_LCD_RST | PIN_LCD_DC | PIN_LCD_BL);
    Xil_Out32(GPIO_DATA_ADDR, gpio_data);
    Xil_Out32(GPIO_TRI_ADDR, gpio_tri);

    drive_low(PIN_TOUCH_INT);
    drive_low(PIN_TOUCH_RST);
    init_panel_landscape();
    fill_rect(0, 0, 319, 239, 0x0000);
    set_gpio(PIN_LCD_BL, 1);
    touch_select_5d();
    init_tft_spi();
    *mb_status = 0;
    *mb_command = STATE_READY;

    int frame_active = 0;
    for (;;) {
        uint32_t cmd = *mb_command;
        if (cmd == CMD_FRAME_BEGIN) {
            /* The display and touch chip share clock/data. Isolate GT911
             * throughout SPI, including ST7789 window commands. */
            drive_low(PIN_TOUCH_INT);
            drive_low(PIN_TOUCH_RST);
            delay_cycles(100000);
            set_gpio(PIN_LCD_MOSI, 1);
            set_gpio(PIN_LCD_SCK, 0);
            set_direction(PIN_LCD_MOSI | PIN_LCD_SCK, 1);
            frame_active = begin_rectangle();
            acknowledge(STATE_ACK_BEGIN);
        } else if (cmd == CMD_FRAME_RECT && frame_active) {
            spi_leave();
            begin_rectangle();
            acknowledge(STATE_ACK_RECT);
        } else if (cmd == CMD_FRAME_CHUNK && frame_active) {
            uint32_t pixels = *mb_status;
            if (pixels > 2040) pixels = 2040;
            spi_enter();
            volatile uint16_t *source = (volatile uint16_t *)MB_PAYLOAD_ADDR;
            for (uint32_t i = 0; i < pixels; ++i) {
                uint16_t color = source[i];
                spi_byte_hw((uint8_t)(color >> 8));
                spi_byte_hw((uint8_t)color);
            }
            spi_leave();
            acknowledge(STATE_ACK_CHUNK);
        } else if (cmd == CMD_FRAME_END && frame_active) {
            spi_leave();
            set_gpio(PIN_LCD_CS, 1);
            set_gpio(PIN_LCD_DC, 0);
            /* Hand the wires back to I2C before GT911 leaves reset. */
            release_line(PIN_I2C_SDA);
            release_line(PIN_I2C_SCL);
            touch_select_5d();
            gt911_address = GT911_ADDRESS;
            consecutive_touch_read_failures = 0;
            frame_active = 0;
            acknowledge(STATE_ACK_END);
        } else if (!frame_active && cmd == 0 && *mb_status == 0) {
            poll_touch();
            delay_cycles(200000);
        }
    }
    return 0;
}
