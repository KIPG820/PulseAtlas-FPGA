`timescale 1ns/1ps

module tb_qrs_equiv;
  logic aclk = 0;
  always #5 aclk = ~aclk;
  logic aresetn = 0;
  logic enable = 1;
  logic soft_reset = 0;
  logic bpf_en = 1;
  logic searchback_en = 1;
  logic [15:0] learn_samples = 16'd2000;
  logic [31:0] frame_idx = 0;
  logic beat_valid = 0;
  logic signed [31:0] x = 0;
  logic old_fire, new_fire;
  logic [31:0] old_frame, new_frame;
  logic [15:0] old_rr, new_rr;
  logic old_sat, new_sat, old_sb, new_sb;
  logic old_ready, new_ready;
  logic [47:0] old_met, new_met, old_thr, new_thr;

  ecg_qrs_detector old_det (
    .aclk, .aresetn, .enable, .soft_reset, .bpf_en, .searchback_en,
    .learn_samples, .frame_idx, .beat_valid, .x,
    .fire(old_fire), .fire_frame(old_frame), .fire_rr(old_rr),
    .sat(old_sat), .searchback_hit(old_sb), .ready(old_ready),
    .dbg_met(old_met), .dbg_thresh(old_thr)
  );
  ecg_qrs_detector_opt new_det (
    .aclk, .aresetn, .enable, .soft_reset, .bpf_en, .searchback_en,
    .learn_samples, .frame_idx, .beat_valid, .x,
    .fire(new_fire), .fire_frame(new_frame), .fire_rr(new_rr),
    .sat(new_sat), .searchback_hit(new_sb), .ready(new_ready),
    .dbg_met(new_met), .dbg_thresh(new_thr)
  );

  logic [48:0] old_events [0:63];
  logic [48:0] new_events [0:63];
  integer old_count = 0, new_count = 0;
  always @(posedge aclk) begin
    if (aresetn && old_fire) begin
      if (old_count >= 64) $fatal(1, "old event array overflow");
      old_events[old_count] = {old_sb, old_rr, old_frame};
      old_count = old_count + 1;
    end
    if (aresetn && new_fire) begin
      if (new_count >= 64) $fatal(1, "new event array overflow");
      new_events[new_count] = {new_sb, new_rr, new_frame};
      new_count = new_count + 1;
    end
  end

  task automatic run_case(input string name, input integer lead);
    integer fd, got, last, idx, wait_count;
    string header;
    logic [255:0] data;
    logic [127:0] user;
    string path;
    begin
      @(negedge aclk);
      aresetn = 0;
      beat_valid = 0;
      old_count = 0;
      new_count = 0;
      repeat (8) @(negedge aclk);
      aresetn = 1;
      path = {"vectors/ip5_", name, ".mem"};
      fd = $fopen(path, "r");
      if (fd == 0) $fatal(1, "cannot open %s", path);
      got = $fgets(header, fd);
      idx = 0;
      while (!$feof(fd)) begin
        got = $fscanf(fd, "%h %h %d\n", data, user, last);
        if (got != 3) break;
        wait_count = 0;
        while (!(old_ready && new_ready)) begin
          @(negedge aclk);
          wait_count = wait_count + 1;
          if (wait_count > 10000) $fatal(1, "detector stalled at %0d", idx);
        end
        frame_idx = idx;
        x = data[lead*32 +: 32];
        beat_valid = 1;
        @(negedge aclk);
        beat_valid = 0;
        idx = idx + 1;
      end
      $fclose(fd);
      repeat (500) @(negedge aclk);
      if (old_count != new_count)
        $fatal(1, "%s lead=%0d event count old=%0d new=%0d", name, lead, old_count, new_count);
      for (integer j = 0; j < old_count; j = j + 1) begin
        if (old_events[j] !== new_events[j])
          $fatal(1, "%s lead=%0d event[%0d] old=%h new=%h", name, lead, j, old_events[j], new_events[j]);
      end
      if (idx < 1000 || old_count == 0)
        $fatal(1, "%s lead=%0d insufficient coverage: samples=%0d events=%0d", name, lead, idx, old_count);
      $display("PASS %s lead=%0d samples=%0d identical_events=%0d", name, lead, idx, old_count);
    end
  endtask

  initial begin
    run_case("clean72", 0);
    run_case("fast150", 0);
    run_case("noisy", 0);
    run_case("slow45", 0);
    run_case("mode_a", 0);
    run_case("mode_and", 0);
    run_case("clean72", 7);
    run_case("slow45", 7);
    $display("PASS all QRS equivalence cases");
    $finish;
  end
endmodule
