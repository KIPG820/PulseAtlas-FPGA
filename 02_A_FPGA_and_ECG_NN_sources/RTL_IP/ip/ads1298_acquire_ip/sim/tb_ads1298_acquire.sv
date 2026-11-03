`timescale 1ns/1ps
module tb_ads1298_acquire;
    localparam [4:0] S_ERROR = 18, S_IDLE = 19;
    reg clk = 0;
    always #5 clk = ~clk; // 100 MHz
    reg aresetn = 0;
    reg miso = 0;
    reg drdy_n = 1;
    wire mosi, sclk, cs_n, adc_start, adc_pwdn_n, adc_reset_n;
    wire [255:0] tdata;
    wire [127:0] tuser;
    wire tvalid, tlast;
    reg tready = 1;

    reg [7:0] awaddr = 0, araddr = 0;
    reg awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;
    reg [31:0] wdata = 0;
    reg [3:0] wstrb = 4'hf;
    wire awready, wready, bvalid, arready, rvalid;
    wire [1:0] bresp, rresp;
    wire [31:0] rdata;

    ads1298_acquire_ip #(
        .POWER_WAIT_CYCLES(40), .PRE_RESET_CYCLES(10),
        .RESET_PULSE_CYCLES(10), .POST_RESET_CYCLES(10),
        .REF_WAIT_CYCLES(40), .CS_HOLD_CYCLES(10),
        .ID_POLL_CYCLES(5000), .RETRY_WAIT_CYCLES(5000),
        .DRDY_TIMEOUT_CYCLES(30000)
    ) dut (
        .aclk(clk), .aresetn(aresetn),
        .ecg_miso(miso), .ecg_ready_n(drdy_n),
        .ecg_mosi(mosi), .ecg_sclk(sclk), .ecg_cs_n(cs_n),
        .ecg_start(adc_start), .ecg_pwdn_n(adc_pwdn_n),
        .ecg_reset_n(adc_reset_n),
        .m_axis_ecg8_tdata(tdata), .m_axis_ecg8_tuser(tuser),
        .m_axis_ecg8_tlast(tlast), .m_axis_ecg8_tvalid(tvalid),
        .m_axis_ecg8_tready(tready),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb),
        .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid),
        .s_axi_rready(rready)
    );

    // Behavioral ADS1298 model: enough command decoding to verify actual
    // SDATAC, RREG, WREG, RDATAC transactions and 216-bit frame reception.
    reg [7:0] adc_regs [0:31];
    reg streaming = 1;
    reg adc_present = 0;
    reg frame_mode = 0;
    reg [7:0] command = 0, address = 0, rx_byte = 0;
    integer bits_seen = 0;
    reg [215:0] frame_word = {
        24'hc12345, 24'h000001, 24'hffffff, 24'h800000,
        24'h7fffff, 24'h000000, 24'h000123, 24'hfedcba, 24'h400000
    };
    integer k;
    initial begin
        for (k=0; k<32; k=k+1) adc_regs[k] = 0;
        adc_regs[0] = 8'h92;
    end
    always @(negedge adc_reset_n) begin
        for (k=1; k<32; k=k+1) adc_regs[k] = 0;
        adc_regs[0] = 8'h92;
        streaming = 1;
    end
    always @(negedge cs_n) begin
        bits_seen = 0;
        rx_byte = 0;
        frame_mode = streaming;
    end
    always @(posedge sclk) begin
        if (!cs_n) begin
            if (frame_mode && bits_seen < 216)
                miso <= frame_word[215-bits_seen];
            else if (command[7:5] == 3'b001 && bits_seen >= 16 && bits_seen < 24)
                miso <= adc_present ? adc_regs[address][23-bits_seen] : 1'b1;
            else
                miso <= 0;
        end
    end
    always @(negedge sclk) begin
        if (!cs_n) begin
            rx_byte = {rx_byte[6:0], mosi};
            bits_seen = bits_seen + 1;
            if (bits_seen == 8) begin
                command = rx_byte;
                address = {3'b000, rx_byte[4:0]};
                if (rx_byte == 8'h11) streaming = 0;
                if (rx_byte == 8'h10) streaming = 1;
            end
            if (bits_seen == 24 && command[7:5] == 3'b010)
                adc_regs[address] = rx_byte;
        end
    end

    integer observed = 0;
    always @(posedge clk) begin
        if (aresetn && tvalid && tready) begin
            if (!tlast) $fatal(1, "TLAST missing");
            if (tuser[55:32] !== 24'hc12345) $fatal(1, "status mismatch");
            if (tuser[31:0] !== (observed == 3 ? 4 : observed == 4 ? 6 : observed))
                $fatal(1, "sequence mismatch: got %0d", tuser[31:0]);
            if (tdata[31:0] !== 32'h00000001 ||
                tdata[63:32] !== 32'hffffffff ||
                tdata[95:64] !== 32'hff800000 ||
                tdata[127:96] !== 32'h007fffff ||
                tdata[255:224] !== 32'h00400000)
                $fatal(1, "channel unpack/sign extension mismatch");
            observed = observed + 1;
        end
    end

    task automatic pulse_drdy;
        begin
            @(negedge clk); drdy_n = 0;
            repeat (8) @(negedge clk);
            drdy_n = 1;
        end
    endtask

    task automatic axi_write(input [7:0] addr, input [31:0] value);
        begin
            @(negedge clk);
            wdata = value;
            wvalid = 1;
            @(negedge clk);
            wvalid = 0;
            repeat (2) @(negedge clk);
            awaddr = addr;
            awvalid = 1;
            @(negedge clk);
            awvalid = 0;
            wait (bvalid);
            @(negedge clk);
        end
    endtask

    task automatic axi_read(input [7:0] addr, output [31:0] value);
        begin
            @(negedge clk);
            araddr = addr;
            arvalid = 1;
            @(negedge clk);
            arvalid = 0;
            wait (rvalid);
            value = rdata;
            @(negedge clk);
        end
    endtask

    reg [31:0] readback;
    reg saw_pwdn_release = 0;
    reg saw_reset_pulse = 0;
    reg last_reset_n = 0;
    always @(posedge clk) begin
        if (aresetn) begin
            if (adc_pwdn_n) saw_pwdn_release <= 1;
            if (last_reset_n && !adc_reset_n && saw_pwdn_release)
                saw_reset_pulse <= 1;
            if (adc_start && (!adc_pwdn_n || !adc_reset_n || !saw_reset_pulse))
                $fatal(1, "START before separate PWDN release and RESET pulse");
        end
        last_reset_n <= adc_reset_n;
    end

    initial begin
        repeat (8) @(negedge clk);
        aresetn = 1;
        wait (dut.state == S_ERROR);
        if (dut.last_id == 8'h92 || dut.config_error)
            $fatal(1, "missing ADC was not reported as a retryable disconnect");
        adc_present = 1;
        wait (dut.state == S_IDLE && dut.last_id == 8'h92);
        if (adc_start !== 0 || dut.state == 5'd16)
            $fatal(1, "idle detection unexpectedly started ECG streaming");
        // The physical ID is polled while idle, so unplug/replug updates
        // status without starting the sampling pipeline.
        adc_present = 0;
        wait (dut.state == S_ERROR && dut.last_id != 8'h92);
        adc_present = 1;
        wait (dut.state == S_IDLE && dut.last_id == 8'h92);
        axi_write(8'h00, 32'h0000002b); // enable + reinit + 1000 SPS + stream
        wait (adc_start == 1);
        if (!saw_pwdn_release || !saw_reset_pulse)
            $fatal(1, "separate PWDN/RESET sequence missing");
        if (dut.last_id !== 8'h92) $fatal(1, "ID read failed");
        if (adc_regs[1] !== 8'h85 || adc_regs[3] !== 8'hcc ||
            adc_regs[8'h18] !== 8'h09 || adc_regs[8'h19] !== 8'hd8)
            $fatal(1, "ADC configuration failed");
        pulse_drdy();
        wait (observed == 1);
        tready = 0;
        repeat (200) @(negedge clk);
        pulse_drdy();
        repeat (14000) @(negedge clk);
        pulse_drdy();
        repeat (14000) @(negedge clk);
        pulse_drdy();
        repeat (14000) @(negedge clk);
        if (dut.drop_count !== 1 || !dut.overflow)
            $fatal(1, "FIFO overflow accounting failed: drops=%0d", dut.drop_count);
        tready = 1;
        wait (observed == 3);
        repeat (10) @(negedge clk);
        if (dut.frame_count !== 3) $fatal(1, "frame count mismatch");
        axi_read(8'h10, readback);
        if (readback !== 1) $fatal(1, "AXI-Lite drop counter read failed");
        // Reconfigure at the register interface, then confirm read-back
        // verified 1000 SPS and internal-test mode settings.
        axi_write(8'h00, 32'h0000002f);
        wait (adc_start == 0);
        wait (adc_start == 1);
        if (adc_regs[1] !== 8'h85 || adc_regs[2] !== 8'h11 ||
            adc_regs[5] !== 8'h05 || dut.config_error)
            $fatal(1, "AXI-Lite reconfiguration failed");
        pulse_drdy();
        wait (observed == 4);
        frame_word[215:192] = 24'hb12345;
        pulse_drdy();
        repeat (14000) @(negedge clk);
        if (dut.frame_error_count !== 1 || dut.drop_count !== 2 || observed !== 4)
            $fatal(1, "bad frame header was not rejected");
        frame_word[215:192] = 24'hc12345;
        pulse_drdy();
        wait (observed == 5);
        axi_write(8'h00, 32'h00000001); // keep ADC powered; stop stream, poll ID in idle
        wait (dut.state == S_IDLE);
        repeat (20) @(negedge clk);
        axi_read(8'h00, readback);
        if (adc_start !== 0 || streaming !== 0 || dut.last_id !== 8'h92 || readback[5] !== 0)
            $fatal(1, "stream stop did not leave ADS1298 in live idle status mode");
        $display("PASS: idle hot-plug detection, stream start/stop, SPI frame, sign extension, AXI backpressure, overflow, reconfiguration and bad-header recovery");
        $finish;
    end
    initial begin
        #5000000;
        $display("DEBUG state=%0d cfg_index=%0d last_id=%02x cfg_err=%0d frame_count=%0d observed=%0d",
                 dut.state, dut.cfg_index, dut.last_id, dut.config_error,
                 dut.frame_count, observed);
        $fatal(1, "test timed out");
    end
endmodule
