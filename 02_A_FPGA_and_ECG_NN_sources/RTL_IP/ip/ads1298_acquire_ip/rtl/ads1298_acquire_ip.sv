`timescale 1ns/1ps
// PYNQ-Z2 ADS1298 acquisition IP. All digital logic uses one PL clock domain.
// Normal data format: 24-bit status + 8 x 24-bit two's-complement samples.
module ads1298_acquire_ip #(
    parameter integer POWER_WAIT_CYCLES  = 20000000, // 200 ms at 100 MHz
    parameter integer PRE_RESET_CYCLES   = 20000000, // 200 ms after PWDN release for oscillator/POR
    parameter integer RESET_PULSE_CYCLES = 1000,     // 10 us
    parameter integer POST_RESET_CYCLES  = 100000,   // 1 ms
    parameter integer REF_WAIT_CYCLES    = 16000000, // 160 ms
    parameter integer CS_HOLD_CYCLES     = 400,      // 4 us
    parameter integer ID_POLL_CYCLES      = 100000000, // 1 s at 100 MHz
    parameter integer RETRY_WAIT_CYCLES   = 25000000,  // 250 ms at 100 MHz
    parameter integer DRDY_TIMEOUT_CYCLES = 50000000  // 500 ms at 100 MHz
) (
    input  wire         aclk,
    input  wire         aresetn,
    input  wire         ecg_miso,
    input  wire         ecg_ready_n,
    output wire         ecg_mosi,
    output wire         ecg_sclk,
    output wire         ecg_cs_n,
    output reg          ecg_start,
    output reg          ecg_pwdn_n,
    output reg          ecg_reset_n,

    output wire [255:0] m_axis_ecg8_tdata,
    output wire [127:0] m_axis_ecg8_tuser,
    output wire         m_axis_ecg8_tlast,
    output wire         m_axis_ecg8_tvalid,
    input  wire         m_axis_ecg8_tready,

    input  wire [7:0]   s_axi_awaddr,
    input  wire         s_axi_awvalid,
    output wire         s_axi_awready,
    input  wire [31:0]  s_axi_wdata,
    input  wire [3:0]   s_axi_wstrb,
    input  wire         s_axi_wvalid,
    output wire         s_axi_wready,
    output reg  [1:0]   s_axi_bresp,
    output reg          s_axi_bvalid,
    input  wire         s_axi_bready,
    input  wire [7:0]   s_axi_araddr,
    input  wire         s_axi_arvalid,
    output wire         s_axi_arready,
    output reg  [31:0]  s_axi_rdata,
    output reg  [1:0]   s_axi_rresp,
    output reg          s_axi_rvalid,
    input  wire         s_axi_rready
);
    localparam [4:0]
        S_DISABLED=0, S_POWER_WAIT=1, S_RELEASE_PRE=2,
        S_RESET_PULSE=3, S_RESET_WAIT=4,
        S_SD_SEND=5, S_SD_WAIT=6, S_ID_SEND=7, S_ID_WAIT=8,
        S_CFG_WRITE_SEND=9, S_CFG_WRITE_WAIT=10,
        S_CFG_READ_SEND=11, S_CFG_READ_WAIT=12,
        S_REF_WAIT=13, S_RDATAC_SEND=14, S_RDATAC_WAIT=15,
        S_RUN=16, S_FRAME_WAIT=17, S_ERROR=18,
        S_IDLE=19, S_ID_POLL_SEND=20, S_ID_POLL_WAIT=21,
        S_STOP_SEND=22, S_STOP_WAIT=23;
    reg [4:0] state;
    reg [31:0] wait_count;
    reg [4:0] cfg_index;

    reg ctrl_enable, ctrl_test, ctrl_1000, ctrl_lead_off, ctrl_stream_enable;
    reg [31:0] drdy_timeout_count;
    reg [7:0] spi_half_period, cfg_gain, cfg_loff, cfg_loff_p, cfg_loff_n;
    reg reinit_pulse, clear_stats_pulse;
    reg [7:0] last_id;
    reg config_error, frame_error, overflow;
    reg [31:0] frame_seq, active_seq, frame_count, drop_count;
    reg [31:0] config_error_count, frame_error_count;

    reg [7:0] aw_addr_hold;
    reg [31:0] w_data_hold;
    reg [3:0] w_strb_hold;
    reg aw_hold, w_hold;
    assign s_axi_awready = !aw_hold && !s_axi_bvalid;
    assign s_axi_wready  = !w_hold && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;

    always @(posedge aclk) begin
        if (!aresetn) begin
            ctrl_enable <= 1;
            ctrl_test <= 0;
            ctrl_1000 <= 0;
            ctrl_lead_off <= 0;
            ctrl_stream_enable <= 0;
            spi_half_period <= 24; // 100 MHz / (2 * 25) = 2 MHz
            cfg_gain <= 0;         // all channels: gain 6, normal electrode input
            cfg_loff <= 0;
            cfg_loff_p <= 0;
            cfg_loff_n <= 0;
            reinit_pulse <= 0;
            clear_stats_pulse <= 0;
            aw_hold <= 0;
            w_hold <= 0;
            aw_addr_hold <= 0;
            w_data_hold <= 0;
            w_strb_hold <= 0;
            s_axi_bresp <= 0;
            s_axi_bvalid <= 0;
            s_axi_rdata <= 0;
            s_axi_rresp <= 0;
            s_axi_rvalid <= 0;
        end else begin
            reinit_pulse <= 0;
            clear_stats_pulse <= 0;
            if (s_axi_awvalid && s_axi_awready) begin
                aw_addr_hold <= s_axi_awaddr;
                aw_hold <= 1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_data_hold <= s_axi_wdata;
                w_strb_hold <= s_axi_wstrb;
                w_hold <= 1;
            end
            if (aw_hold && w_hold && !s_axi_bvalid) begin
                case (aw_addr_hold[7:2])
                    6'h00: if (w_strb_hold[0]) begin
                        ctrl_enable <= w_data_hold[0];
                        reinit_pulse <= w_data_hold[1];
                        ctrl_test <= w_data_hold[2];
                        ctrl_1000 <= w_data_hold[3];
                        ctrl_lead_off <= w_data_hold[4];
                        ctrl_stream_enable <= w_data_hold[5];
                    end
                    6'h01: if (w_strb_hold[0] && w_data_hold[0])
                        clear_stats_pulse <= 1;
                    6'h07: if (w_strb_hold[0] && w_data_hold[7:0] >= 4)
                        spi_half_period <= w_data_hold[7:0];
                    6'h08: if (w_strb_hold[0]) cfg_gain <= {1'b0, w_data_hold[6:4], 4'b0};
                    6'h09: if (w_strb_hold[0]) cfg_loff <= w_data_hold[7:0];
                    6'h0a: if (w_strb_hold[0]) cfg_loff_p <= w_data_hold[7:0];
                    6'h0b: if (w_strb_hold[0]) cfg_loff_n <= w_data_hold[7:0];
                    default: ;
                endcase
                aw_hold <= 0;
                w_hold <= 0;
                s_axi_bresp <= 0;
                s_axi_bvalid <= 1;
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 0;
            if (s_axi_arvalid && s_axi_arready) begin
                case (s_axi_araddr[7:2])
                    6'h00: s_axi_rdata <= {26'b0, ctrl_stream_enable,
                        ctrl_lead_off, ctrl_1000, ctrl_test, 1'b0, ctrl_enable};
                    6'h01: s_axi_rdata <= {24'b0, last_id == 8'h92,
                        state == S_FRAME_WAIT, state == S_RUN,
                        overflow, frame_error, config_error, state == S_ERROR, ctrl_enable};
                    6'h02: s_axi_rdata <= frame_seq;
                    6'h03: s_axi_rdata <= frame_count;
                    6'h04: s_axi_rdata <= drop_count;
                    6'h05: s_axi_rdata <= frame_error_count;
                    6'h06: s_axi_rdata <= config_error_count;
                    6'h07: s_axi_rdata <= {24'b0, spi_half_period};
                    6'h08: s_axi_rdata <= {24'b0, cfg_gain};
                    6'h09: s_axi_rdata <= {24'b0, cfg_loff};
                    6'h0a: s_axi_rdata <= {24'b0, cfg_loff_p};
                    6'h0b: s_axi_rdata <= {24'b0, cfg_loff_n};
                    6'h0c: s_axi_rdata <= {24'b0, last_id};
                    6'h0d: s_axi_rdata <= 32'h4c4f4646; // "LOFF" capability
                    default: s_axi_rdata <= 0;
                endcase
                s_axi_rresp <= 0;
                s_axi_rvalid <= 1;
            end
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 0;
        end
    end

    function automatic [7:0] config_address(input [4:0] index);
        begin
            if (index < 12)
                config_address = 8'h01 + index;
            else if (index < 16)
                config_address = 8'h0d + (index - 12);
            else if (index < 18)
                config_address = 8'h18 + (index - 16);
            else
                config_address = 8'h17; // CONFIG4: lead-off comparator power
        end
    endfunction

    function automatic [7:0] config_value(input [4:0] index);
        begin
            case (index)
                0: config_value = ctrl_1000 ? 8'h85 : 8'h86; // HR, 1000/500 SPS
                1: config_value = ctrl_test ? 8'h11 : 8'h00; // enable internal test source
                2: config_value = 8'hcc; // internal 2.4 V reference and RLD
                3: config_value = ctrl_lead_off ? cfg_loff : 8'h00;
                4,5,6,7,8,9,10,11: config_value = cfg_gain | (ctrl_test ? 8'h05 : 8'h00);
                12: config_value = 8'h01; // RLD: LL (CH1P)
                13: config_value = 8'h03; // RLD: RA (CH1N), LA (CH2N)
                14: config_value = ctrl_lead_off ? cfg_loff_p : 8'h00;
                15: config_value = ctrl_lead_off ? cfg_loff_n : 8'h00;
                16: config_value = 8'h09; // WCTA = RA (CH1N)
                17: config_value = 8'hd8; // WCTB = LA (CH2N), WCTC = LL (CH1P)
                18: config_value = (ctrl_lead_off && !ctrl_test) ? 8'h02 : 8'h00;
                default: config_value = 0;
            endcase
        end
    endfunction

    function automatic [7:0] config_mask(input [4:0] index);
        begin
            case (index)
                1: config_mask = 8'h3f; // CONFIG2 reserved bits may read as reset value
                2: config_mask = 8'hfe; // CONFIG3 bit 0 is read-only RLD_STAT
                default: config_mask = 8'hff;
            endcase
        end
    endfunction

    reg spi_start;
    reg [7:0] spi_bit_count;
    reg [23:0] spi_tx_prefix;
    wire spi_busy, spi_done;
    wire [215:0] spi_rx;
    wire spi_aresetn = aresetn && ctrl_enable && !reinit_pulse;
    ads1298_spi_master #(.CS_HOLD_CYCLES(CS_HOLD_CYCLES)) spi (
        .clk(aclk), .aresetn(spi_aresetn), .start(spi_start),
        .bit_count(spi_bit_count), .tx_prefix(spi_tx_prefix),
        .half_period(spi_half_period), .miso(ecg_miso),
        .mosi(ecg_mosi), .sclk(ecg_sclk), .cs_n(ecg_cs_n),
        .busy(spi_busy), .done(spi_done), .rx_bits(spi_rx)
    );

    reg [2:0] drdy_sync;
    wire drdy_fall = drdy_sync[2:1] == 2'b10;
    always @(posedge aclk) begin
        if (!aresetn) drdy_sync <= 3'b111;
        else drdy_sync <= {drdy_sync[1:0], ecg_ready_n};
    end

    function automatic [255:0] unpack_samples(input [215:0] frame);
        integer ch;
        reg [23:0] raw;
        begin
            unpack_samples = 0;
            for (ch = 0; ch < 8; ch = ch + 1) begin
                raw = frame[191 - ch*24 -: 24];
                unpack_samples[ch*32 +: 32] = {{8{raw[23]}}, raw};
            end
        end
    endfunction

    reg [255:0] fifo_data [0:1];
    reg [127:0] fifo_user [0:1];
    reg wr_ptr, rd_ptr;
    reg [1:0] fifo_count;
    wire fifo_pop = (fifo_count != 0) && m_axis_ecg8_tready;
    wire frame_header_ok = spi_rx[215:212] == 4'hc;
    wire frame_complete = state == S_FRAME_WAIT && spi_done;
    wire fifo_push = frame_complete && frame_header_ok &&
                     (fifo_count != 2 || fifo_pop);
    // ADS1298 status = {4'hC, LOFF_STATP[7:0], LOFF_STATN[7:0], GPIO[7:4]}.
    // Preserve raw status in [55:32] and propagate the combined physical
    // electrode fault mask into [63:56] for quality/lead12/packetizer.
    wire [7:0] electrode_off = ctrl_test ? 8'hff :
        (ctrl_lead_off ?
            (spi_rx[211:204] | (spi_rx[203:196] & 8'h03)) : 8'h00);
    wire [127:0] new_user = {64'b0, electrode_off,
                             spi_rx[215:192], active_seq};
    assign m_axis_ecg8_tdata = fifo_data[rd_ptr];
    assign m_axis_ecg8_tuser = fifo_user[rd_ptr];
    assign m_axis_ecg8_tvalid = fifo_count != 0;
    assign m_axis_ecg8_tlast = 1'b1;

    always @(posedge aclk) begin
        if (!aresetn) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            fifo_count <= 0;
        end else begin
            if (fifo_push) begin
                fifo_data[wr_ptr] <= unpack_samples(spi_rx);
                fifo_user[wr_ptr] <= new_user;
                wr_ptr <= ~wr_ptr;
            end
            if (fifo_pop) rd_ptr <= ~rd_ptr;
            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + 1'b1;
                2'b01: fifo_count <= fifo_count - 1'b1;
                default: ;
            endcase
            if (reinit_pulse || !ctrl_enable || state == S_IDLE || state == S_ERROR) begin
                wr_ptr <= 0;
                rd_ptr <= 0;
                fifo_count <= 0;
            end
        end
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            state <= S_DISABLED;
            wait_count <= 0;
            cfg_index <= 0;
            ecg_start <= 0;
            ecg_pwdn_n <= 0;
            ecg_reset_n <= 0;
            spi_start <= 0;
            spi_bit_count <= 0;
            spi_tx_prefix <= 0;
            last_id <= 0;
            config_error <= 0;
            frame_error <= 0;
            overflow <= 0;
            frame_seq <= 0;
            active_seq <= 0;
            frame_count <= 0;
            drop_count <= 0;
            config_error_count <= 0;
            frame_error_count <= 0;
            drdy_timeout_count <= 0;
        end else begin
            spi_start <= 0;
            if (clear_stats_pulse) begin
                config_error <= 0;
                frame_error <= 0;
                overflow <= 0;
                frame_count <= 0;
                drop_count <= 0;
                config_error_count <= 0;
                frame_error_count <= 0;
            end
            if (!ctrl_enable) begin
                state <= S_DISABLED;
                ecg_start <= 0;
                ecg_pwdn_n <= 0;
                ecg_reset_n <= 0;
                last_id <= 8'hff;
                drdy_timeout_count <= 0;
            end else if (reinit_pulse || state == S_DISABLED) begin
                state <= S_POWER_WAIT;
                wait_count <= POWER_WAIT_CYCLES;
                cfg_index <= 0;
                ecg_start <= 0;
                ecg_pwdn_n <= 0;
                ecg_reset_n <= 0;
                last_id <= 8'hff;
                drdy_timeout_count <= 0;
            end else case (state)
                S_POWER_WAIT: if (wait_count == 0) begin
                    ecg_pwdn_n <= 1;
                    ecg_reset_n <= 1;
                    wait_count <= PRE_RESET_CYCLES;
                    state <= S_RELEASE_PRE;
                end else wait_count <= wait_count - 1'b1;
                S_RELEASE_PRE: if (wait_count == 0) begin
                    ecg_reset_n <= 0;
                    wait_count <= RESET_PULSE_CYCLES;
                    state <= S_RESET_PULSE;
                end else wait_count <= wait_count - 1'b1;
                S_RESET_PULSE: if (wait_count == 0) begin
                    ecg_reset_n <= 1;
                    wait_count <= POST_RESET_CYCLES;
                    state <= S_RESET_WAIT;
                end else wait_count <= wait_count - 1'b1;
                S_RESET_WAIT: if (wait_count == 0) state <= S_SD_SEND;
                    else wait_count <= wait_count - 1'b1;
                S_SD_SEND: if (!spi_busy) begin
                    spi_tx_prefix <= 24'h110000; // SDATAC before register access
                    spi_bit_count <= 8;
                    spi_start <= 1;
                    state <= S_SD_WAIT;
                end
                S_SD_WAIT: if (spi_done) state <= S_ID_SEND;
                S_ID_SEND: if (!spi_busy) begin
                    spi_tx_prefix <= 24'h200000; // RREG ID, one byte
                    spi_bit_count <= 24;
                    spi_start <= 1;
                    state <= S_ID_WAIT;
                end
                S_ID_WAIT: if (spi_done) begin
                    last_id <= spi_rx[7:0];
                    if (spi_rx[7:0] != 8'h92) begin
                        wait_count <= RETRY_WAIT_CYCLES;
                        state <= S_ERROR;
                    end else begin
                        cfg_index <= 0;
                        state <= S_CFG_WRITE_SEND;
                    end
                end
                S_CFG_WRITE_SEND: if (!spi_busy) begin
                    spi_tx_prefix <= {8'h40 | config_address(cfg_index),
                                      8'h00, config_value(cfg_index)};
                    spi_bit_count <= 24;
                    spi_start <= 1;
                    state <= S_CFG_WRITE_WAIT;
                end
                S_CFG_WRITE_WAIT: if (spi_done) state <= S_CFG_READ_SEND;
                S_CFG_READ_SEND: if (!spi_busy) begin
                    spi_tx_prefix <= {8'h20 | config_address(cfg_index),16'h0000};
                    spi_bit_count <= 24;
                    spi_start <= 1;
                    state <= S_CFG_READ_WAIT;
                end
                S_CFG_READ_WAIT: if (spi_done) begin
                    if ((spi_rx[7:0] & config_mask(cfg_index)) !=
                        (config_value(cfg_index) & config_mask(cfg_index))) begin
                        config_error <= 1;
                        config_error_count <= config_error_count + 1'b1;
                        wait_count <= RETRY_WAIT_CYCLES;
                        state <= S_ERROR;
                    end else if (cfg_index == 18) begin
                        wait_count <= REF_WAIT_CYCLES;
                        state <= S_REF_WAIT;
                    end else begin
                        cfg_index <= cfg_index + 1'b1;
                        state <= S_CFG_WRITE_SEND;
                    end
                end
                S_REF_WAIT: if (wait_count == 0) begin
                    if (ctrl_stream_enable) state <= S_RDATAC_SEND;
                    else begin
                        ecg_start <= 0;
                        wait_count <= ID_POLL_CYCLES;
                        state <= S_IDLE;
                    end
                end
                    else wait_count <= wait_count - 1'b1;
                S_RDATAC_SEND: if (!spi_busy) begin
                    spi_tx_prefix <= 24'h100000;
                    spi_bit_count <= 8;
                    spi_start <= 1;
                    state <= S_RDATAC_WAIT;
                end
                S_RDATAC_WAIT: if (spi_done) begin
                    ecg_start <= 1;
                    drdy_timeout_count <= 0;
                    state <= S_RUN;
                end
                S_RUN: begin
                    if (!ctrl_stream_enable) state <= S_STOP_SEND;
                    else if (drdy_timeout_count >= DRDY_TIMEOUT_CYCLES) begin
                        ecg_start <= 0;
                        last_id <= 8'hff;
                        wait_count <= RETRY_WAIT_CYCLES;
                        drdy_timeout_count <= 0;
                        state <= S_ERROR;
                    end else begin
                        drdy_timeout_count <= drdy_timeout_count + 1'b1;
                        if (drdy_fall) begin
                            drdy_timeout_count <= 0;
                            active_seq <= frame_seq;
                            frame_seq <= frame_seq + 1'b1;
                            if (!spi_busy) begin
                                spi_tx_prefix <= 0;
                                spi_bit_count <= 216;
                                spi_start <= 1;
                                state <= S_FRAME_WAIT;
                            end else begin
                                overflow <= 1;
                                drop_count <= drop_count + 1'b1;
                            end
                        end
                    end
                end
                S_FRAME_WAIT: begin
                    if (drdy_timeout_count >= DRDY_TIMEOUT_CYCLES) begin
                        ecg_start <= 0;
                        last_id <= 8'hff;
                        wait_count <= RETRY_WAIT_CYCLES;
                        drdy_timeout_count <= 0;
                        state <= S_ERROR;
                    end else begin
                        drdy_timeout_count <= drdy_timeout_count + 1'b1;
                        if (drdy_fall) begin
                            drdy_timeout_count <= 0;
                            frame_seq <= frame_seq + 1'b1;
                            drop_count <= drop_count + 1'b1;
                            overflow <= 1;
                        end
                        if (spi_done) begin
                            if (!frame_header_ok) begin
                                frame_error <= 1;
                                frame_error_count <= frame_error_count + 1'b1;
                                drop_count <= drop_count + 1'b1;
                            end else if (!fifo_push) begin
                                overflow <= 1;
                                drop_count <= drop_count + 1'b1;
                            end else frame_count <= frame_count + 1'b1;
                            state <= S_RUN;
                        end
                    end
                end
                S_IDLE: begin
                    ecg_start <= 0;
                    if (ctrl_stream_enable) state <= S_RDATAC_SEND;
                    else if (wait_count == 0) state <= S_ID_POLL_SEND;
                    else wait_count <= wait_count - 1'b1;
                end
                S_ID_POLL_SEND: if (!spi_busy) begin
                    spi_tx_prefix <= 24'h200000; // RREG ID, one byte
                    spi_bit_count <= 24;
                    spi_start <= 1;
                    state <= S_ID_POLL_WAIT;
                end
                S_ID_POLL_WAIT: if (spi_done) begin
                    last_id <= spi_rx[7:0];
                    if (spi_rx[7:0] != 8'h92) begin
                        wait_count <= RETRY_WAIT_CYCLES;
                        state <= S_ERROR;
                    end else begin
                        wait_count <= ID_POLL_CYCLES;
                        state <= S_IDLE;
                    end
                end
                S_STOP_SEND: if (!spi_busy) begin
                    spi_tx_prefix <= 24'h110000; // SDATAC before idle polling
                    spi_bit_count <= 8;
                    spi_start <= 1;
                    state <= S_STOP_WAIT;
                end
                S_STOP_WAIT: if (spi_done) begin
                    ecg_start <= 0;
                    drdy_timeout_count <= 0;
                    wait_count <= ID_POLL_CYCLES;
                    state <= S_IDLE;
                end
                S_ERROR: begin
                    ecg_start <= 0;
                    if (wait_count == 0) begin
                        ecg_pwdn_n <= 0;
                        ecg_reset_n <= 0;
                        wait_count <= POWER_WAIT_CYCLES;
                        cfg_index <= 0;
                        state <= S_POWER_WAIT;
                    end else wait_count <= wait_count - 1'b1;
                end
                default: state <= S_DISABLED;
            endcase
        end
    end
endmodule
