`timescale 1ns/1ps

module tb_filter_equiv;
  localparam int NFRAMES = 170; // crosses the 81-sample FIR ring twice
  logic clk = 0;
  always #5 clk = ~clk;
  logic rstn = 0;
  logic [255:0] in_data = '0;
  logic [127:0] in_user = '0;
  logic pending_ref = 0, pending_opt = 0;
  logic in_last = 0;
  logic [7:0] cfg_addr = 0;
  logic [31:0] cfg_data = 0;
  logic cfg_valid = 0;
  wire awready_ref, awready_opt, wready_ref, wready_opt;
  wire bvalid_ref, bvalid_opt;
  wire ready_ref, ready_opt;
  wire [255:0] data_ref, data_opt;
  wire [127:0] user_ref, user_opt;
  wire valid_ref, valid_opt, last_ref, last_opt;
  logic [255:0] capture_ref [0:NFRAMES-1];
  logic [255:0] capture_opt [0:NFRAMES-1];
  logic [127:0] metadata_ref [0:NFRAMES-1];
  logic [127:0] metadata_opt [0:NFRAMES-1];
  int n_ref = 0, n_opt = 0, ticks = 0;

`define FILTER_PORTS(RDY,PENDING,DATA,USER,VALID,LAST,AWREADY,WREADY,BVALID) \
    .aclk(clk), .aresetn(rstn), \
    .s_axi_awaddr(cfg_addr), .s_axi_awvalid(cfg_valid), .s_axi_awready(AWREADY), \
    .s_axi_wdata(cfg_data), .s_axi_wstrb(4'hf), .s_axi_wvalid(cfg_valid), .s_axi_wready(WREADY), \
    .s_axi_bresp(), .s_axi_bvalid(BVALID), .s_axi_bready(1'b1), \
    .s_axi_araddr(8'b0), .s_axi_arvalid(1'b0), .s_axi_arready(), \
    .s_axi_rdata(), .s_axi_rresp(), .s_axi_rvalid(), .s_axi_rready(1'b1), \
    .s_axis_ecg8_tdata(in_data), .s_axis_ecg8_tuser(in_user), \
    .s_axis_ecg8_tvalid(PENDING), .s_axis_ecg8_tready(RDY), \
    .s_axis_ecg8_tlast(in_last), \
    .m_axis_ecg8_tdata(DATA), .m_axis_ecg8_tuser(USER), \
    .m_axis_ecg8_tvalid(VALID), .m_axis_ecg8_tready(1'b1), .m_axis_ecg8_tlast(LAST)

  ecg_classic_filter_ip_r9q baseline (`FILTER_PORTS(ready_ref,pending_ref,data_ref,user_ref,valid_ref,last_ref,awready_ref,wready_ref,bvalid_ref));
  ecg_classic_filter_ip optimized (`FILTER_PORTS(ready_opt,pending_opt,data_opt,user_opt,valid_opt,last_opt,awready_opt,wready_opt,bvalid_opt));

  task automatic write_config(input logic [7:0] address, input logic [31:0] value);
    @(negedge clk);
    cfg_addr = address;
    cfg_data = value;
    cfg_valid = 1;
    do @(posedge clk); while (!(awready_ref && awready_opt && wready_ref && wready_opt));
    @(negedge clk);
    cfg_valid = 0;
    wait (bvalid_ref && bvalid_opt);
    repeat (3) @(negedge clk);
  endtask

  always @(posedge clk) if (rstn) begin
    ticks <= ticks + 1;
    if (ticks > 500000) $fatal(1, "filter output timeout ref=%0d opt=%0d", n_ref, n_opt);
    if (valid_ref) begin
      if (n_ref >= NFRAMES) $fatal(1, "extra baseline frame");
      capture_ref[n_ref] <= data_ref;
      metadata_ref[n_ref] <= user_ref;
      if (last_ref !== (n_ref == NFRAMES-1)) $fatal(1, "baseline TLAST mismatch");
      n_ref <= n_ref + 1;
    end
    if (valid_opt) begin
      if (n_opt >= NFRAMES) $fatal(1, "extra optimized frame");
      capture_opt[n_opt] <= data_opt;
      metadata_opt[n_opt] <= user_opt;
      if (last_opt !== (n_opt == NFRAMES-1)) $fatal(1, "optimized TLAST mismatch");
      n_opt <= n_opt + 1;
    end
  end

  initial begin : test
    int signed v;
    repeat (8) @(negedge clk);
    rstn = 1;
    if ($test$plusargs("BIQUAD")) write_config(8'h48, 32'h1);
    for (int frame = 0; frame < NFRAMES; frame++) begin
      @(negedge clk);
      for (int ch = 0; ch < 8; ch++) begin
        v = ((frame*731 + ch*1093) % 200001) - 100000;
        if ((frame % 19) == 0) v = v + 900000;
        in_data[ch*32 +: 32] = v;
      end
      in_user = 128'(frame);
      in_last = (frame == NFRAMES-1);
      pending_ref = 1;
      pending_opt = 1;
      do begin
        @(posedge clk);
        if (pending_ref && ready_ref) pending_ref <= 0;
        if (pending_opt && ready_opt) pending_opt <= 0;
        @(negedge clk);
      end while (pending_ref || pending_opt);
    end
    wait (n_ref == NFRAMES && n_opt == NFRAMES);
    @(negedge clk);
    for (int frame = 0; frame < NFRAMES; frame++) begin
      if (capture_ref[frame] !== capture_opt[frame] ||
          metadata_ref[frame] !== metadata_opt[frame])
        $fatal(1, "mismatch at frame %0d baseline=%h optimized=%h", frame,
               capture_ref[frame], capture_opt[frame]);
    end
    $display("PASS classic filter exact 8-channel output across %0d frames and two FIR ring wraps; biquad=%0d", NFRAMES, $test$plusargs("BIQUAD"));
    $finish;
  end
endmodule
