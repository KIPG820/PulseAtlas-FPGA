from pathlib import Path

HERE = Path(__file__).resolve().parent
text = (HERE.parent / "qrs_timing/tb_qrs_equiv.sv").read_text(encoding="utf-8")
text = text.replace("module tb_qrs_equiv;", "module tb_qrs_storage_equiv;")
text = text.replace("ecg_qrs_detector_opt new_det", "ecg_qrs_detector_ram new_det")
text = text.replace('path = {"vectors/ip5_", name, ".mem"};',
                    'path = {"../../qrs_timing/sim_run/vectors/ip5_", name, ".mem"};')
text = text.replace("  logic [48:0] old_events", """  integer checked_cycles = 0;
  always @(negedge aclk) begin
    if (aresetn) begin
      if ({old_fire,old_frame,old_rr,old_sat,old_sb,old_ready,old_met,old_thr}
          !== {new_fire,new_frame,new_rr,new_sat,new_sb,new_ready,new_met,new_thr})
        $fatal(1, "cycle mismatch at %0t state=%0d sample=%0d", $time, old_det.state, frame_idx);
      checked_cycles = checked_cycles + 1;
    end
  end

  logic [48:0] old_events""")
text = text.replace("  initial begin\n    run_case", """  // Exercise partially filled histories, wraparound, soft/hard resets,
  // BPF bypass and aborts in every sequencer phase. Check outputs every cycle.
  task automatic run_controls;
    integer k, j, delay_cycles;
    logic [31:0] rng;
    begin
      @(negedge aclk);
      aresetn = 0;
      old_count = 0;
      new_count = 0;
      repeat (4) @(negedge aclk);
      aresetn = 1;
      learn_samples = 0;
      rng = 32'h98765432;
      for (k = 0; k < 3000; k = k + 1) begin
        while (!(old_ready && new_ready)) @(negedge aclk);
        rng = {rng[30:0], rng[31]^rng[21]^rng[1]^rng[0]};
        bpf_en = (k >= 1000);
        searchback_en = (k % 3 != 0);
        frame_idx = k;
        x = $signed(rng) >>> 7;
        beat_valid = 1;
        @(negedge aclk);
        beat_valid = 0;
        delay_cycles = (k % 31);
        repeat (delay_cycles) @(negedge aclk);
        if (k % 19 == 0) begin
          soft_reset = 1;
          @(negedge aclk);
          soft_reset = 0;
        end else if (k % 23 == 0) begin
          enable = 0;
          repeat (3) @(negedge aclk);
          enable = 1;
        end else if (k % 79 == 0) begin
          aresetn = 0;
          repeat (3) @(negedge aclk);
          aresetn = 1;
        end
        // Event storage is only used by the vector cases; prevent overflow here.
        old_count = 0;
        new_count = 0;
      end
      while (!(old_ready && new_ready)) @(negedge aclk);
      $display("PASS directed reset/abort/bypass cases");
    end
  endtask

  initial begin
    #1000000000;
    $fatal(1, "simulation watchdog expired");
  end

  initial begin
    run_case""")
text = text.replace('    $display("PASS all QRS equivalence cases");',
                    '    run_controls();\n    $display("PASS all QRS storage equivalence cases, checked_cycles=%0d", checked_cycles);')
(HERE / "tb_qrs_storage_equiv.sv").write_text(text, encoding="utf-8")
