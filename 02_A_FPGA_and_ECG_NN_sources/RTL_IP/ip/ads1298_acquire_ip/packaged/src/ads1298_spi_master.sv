`timescale 1ns/1ps
// ADS1298 SPI mode 1 (CPOL=0, CPHA=1). One transaction per start pulse.
module ads1298_spi_master #(
    parameter integer CS_HOLD_CYCLES = 400
) (
    input  wire         clk,
    input  wire         aresetn,
    input  wire         start,
    input  wire [7:0]   bit_count,
    input  wire [23:0]  tx_prefix,
    input  wire [7:0]   half_period,
    input  wire         miso,
    output reg          mosi,
    output reg          sclk,
    output reg          cs_n,
    output reg          busy,
    output reg          done,
    output reg  [215:0] rx_bits
);
    localparam [1:0] IDLE=0, SETUP=1, SHIFT=2, HOLD=3;
    reg [1:0] state;
    reg [7:0] bit_index;
    reg [7:0] divider;
    reg [31:0] hold_count;

    always @(posedge clk) begin
        if (!aresetn) begin
            state <= IDLE;
            bit_index <= 0;
            divider <= 0;
            hold_count <= 0;
            mosi <= 0;
            sclk <= 0;
            cs_n <= 1;
            busy <= 0;
            done <= 0;
            rx_bits <= 0;
        end else begin
            done <= 0;
            case (state)
                IDLE: if (start && bit_count != 0) begin
                    cs_n <= 0;
                    busy <= 1;
                    mosi <= tx_prefix[23];
                    sclk <= 0;
                    rx_bits <= 0;
                    bit_index <= 0;
                    divider <= half_period;
                    state <= SETUP;
                end
                SETUP: if (divider == 0) begin
                    divider <= half_period;
                    state <= SHIFT;
                end else divider <= divider - 1'b1;
                SHIFT: if (divider != 0) begin
                    // Change DIN well after the previous falling edge, not on
                    // the same edge at which ADS1298 samples it.
                    if (!sclk && bit_index != 0 && divider == half_period - 1'b1)
                        mosi <= bit_index < 24 ? tx_prefix[23-bit_index] : 1'b0;
                    divider <= divider - 1'b1;
                end else if (!sclk) begin
                    // DOUT changes after this rising edge; DIN is already stable.
                    sclk <= 1;
                    divider <= half_period;
                end else begin
                    // ADS1298 presents DOUT before the falling edge and latches DIN here.
                    sclk <= 0;
                    rx_bits <= {rx_bits[214:0], miso};
                    divider <= half_period;
                    if (bit_index == bit_count - 1'b1) begin
                        hold_count <= CS_HOLD_CYCLES;
                        state <= HOLD;
                    end else begin
                        bit_index <= bit_index + 1'b1;
                    end
                end
                HOLD: if (hold_count != 0) begin
                    hold_count <= hold_count - 1'b1;
                end else begin
                    cs_n <= 1;
                    mosi <= 0;
                    busy <= 0;
                    done <= 1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
