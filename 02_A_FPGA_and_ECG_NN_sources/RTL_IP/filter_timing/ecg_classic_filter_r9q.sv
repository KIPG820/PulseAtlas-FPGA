//=============================================================================
// ecg_classic_filter_ip.sv -- IP3, classic filtering
//
//   DC blocker (IIR)  ->  mains notch  ->  FIR low-pass
//   all three independently bypassable; frame number and quality fields pass
//   through untouched.
//
// Interface per "PYNQ-Z2 ECG PL IP 接口与开发要求" (更新版):
//   one beat carries ALL EIGHT channels -- TDATA[255:0], CH1 in [31:0] --
//   with the frame number, ADS status and quality masks in TUSER[127:0].
//   That is a change from the previous revision, where a record was eight
//   beats of one channel each.
//
// Structure: all eight channels are processed in the same beat, one channel
// at a time through a single datapath.  With 100 MHz and 1000 SPS there are
// 100 000 cycles per sample and the cascade needs roughly 8 x (3 + 3 + 1 +
// 123 + 1) cycles, so the datapath runs at about 0.6 % utilisation --
// nowhere near needing eight parallel filter lanes.
//
// State lives in flip-flop arrays: about 34 k FF, 32 % of the xc7z020.  A
// BRAM-backed sample history would cut that to one BRAM36, at the cost of an
// address decode on every tap read.  FF first, because it is straightforward
// to verify against the model; the BRAM migration is a documented option, not
// a prerequisite.
//
// TIMING
//
// Every stage used to run in a single cycle, which does not close at 100 MHz.
// Synthesis reported -34.5 ns of slack: the biquad notch's two sections were
// wired in series combinationally, 44 ns, with the five products of each
// section mapped onto a five-deep DSP48E1 cascade and the round-and-saturate
// behind them.  The stages are now sequenced over several cycles each -- see
// the header of ecg_filter_stages.sv -- and the arithmetic is unchanged, so
// the bit-exact comparison against model/filter_ref.py still holds.
//=============================================================================

`include "ecg_frame_pkg.sv"
`include "ecg_filter_coeffs.svh"

module ecg_classic_filter_ip_r9q #(
    parameter int ADDR_W = 8
) (
    input  logic                aclk,
    input  logic                aresetn,

    // ---- AXI4-Lite slave ----
    input  logic [ADDR_W-1:0]   s_axi_awaddr,
    input  logic                s_axi_awvalid,
    output logic                s_axi_awready,
    input  logic [31:0]         s_axi_wdata,
    input  logic [3:0]          s_axi_wstrb,
    input  logic                s_axi_wvalid,
    output logic                s_axi_wready,
    output logic [1:0]          s_axi_bresp,
    output logic                s_axi_bvalid,
    input  logic                s_axi_bready,
    input  logic [ADDR_W-1:0]   s_axi_araddr,
    input  logic                s_axi_arvalid,
    output logic                s_axi_arready,
    output logic [31:0]         s_axi_rdata,
    output logic [1:0]          s_axi_rresp,
    output logic                s_axi_rvalid,
    input  logic                s_axi_rready,

    // ---- ecg8_axis slave ----
    input  logic [255:0]        s_axis_ecg8_tdata,
    input  logic [127:0]        s_axis_ecg8_tuser,
    input  logic                s_axis_ecg8_tvalid,
    output logic                s_axis_ecg8_tready,
    input  logic                s_axis_ecg8_tlast,

    // ---- ecg8_axis master ----
    output logic [255:0]        m_axis_ecg8_tdata,
    output logic [127:0]        m_axis_ecg8_tuser,
    output logic                m_axis_ecg8_tvalid,
    input  logic                m_axis_ecg8_tready,
    output logic                m_axis_ecg8_tlast
);

  import ecg_frame_pkg::*;

  localparam int NCH   = 8;
  localparam int NTAPS = FIR_TAPS;
  localparam int COMB_MAX_L = 20;      // L = fs/50 at 1000 SPS
  localparam int FIR_HALF = (NTAPS + 1) / 2;

  //--------------------------------------------------------------------------
  // AXI4-Lite front end
  //--------------------------------------------------------------------------
  logic              reg_wr;
  logic [ADDR_W-1:0] reg_waddr, reg_raddr;
  logic [31:0]       reg_wdata;
  logic [3:0]        reg_wstrb;
  logic [31:0]       reg_rdata;

  axi4lite_regfile #(.ADDR_W(ADDR_W)) u_regfile (
      .aclk, .aresetn,
      .s_axi_awaddr, .s_axi_awvalid, .s_axi_awready,
      .s_axi_wdata,  .s_axi_wstrb,   .s_axi_wvalid, .s_axi_wready,
      .s_axi_bresp,  .s_axi_bvalid,  .s_axi_bready,
      .s_axi_araddr, .s_axi_arvalid, .s_axi_arready,
      .s_axi_rdata,  .s_axi_rresp,   .s_axi_rvalid, .s_axi_rready,
      .reg_wr, .reg_waddr, .reg_wdata, .reg_wstrb, .reg_raddr, .reg_rdata
  );

  //--------------------------------------------------------------------------
  // Configuration
  //--------------------------------------------------------------------------
  localparam logic [31:0] IP_ID           = 32'hEC03_0001;
  localparam logic [31:0] IP_VERSION      = 32'h0001_0001;
  localparam logic [31:0] SCRATCH_PATTERN = 32'hECC0_5C06;

  logic       ctrl_enable, ctrl_soft_reset, ctrl_bypass;
  logic       hp_en, notch_en, lp_en;
  logic       notch_mode;               // 0 = comb, 1 = biquad pair
  logic [1:0] hp_fc_sel, lp_fc_sel;
  logic [4:0] notch_l;
  logic [17:0] notch_rho;
  logic [31:0] sat_count;

  logic [5:0]  coef_addr;               // indirect coefficient port
  logic        coef_we;
  logic [17:0] coef_wdata;

  // SAT_COUNT is cleared by a write to 0x60 but incremented by the filter
  // sequencer, so it may only be driven from ONE always_ff.  Previously the
  // configuration block drove it too, which synthesis reported as
  // "multi-driven net ... 2nd driver pin GND" and resolved by keeping the
  // constant -- so the saturation diagnostic the register map documents was
  // permanently zero on hardware.  Simulation applied both assignments and
  // therefore agreed with the register map, which is why the bit-exact
  // regression never noticed.  All sat_count logic now lives in the
  // sequencer; this block only raises the clear pulse.
  logic sat_clear;

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      ctrl_enable     <= 1'b1;
      ctrl_soft_reset <= 1'b0;
      ctrl_bypass     <= 1'b0;
      hp_en        <= 1'b1;
      notch_en     <= 1'b1;
      lp_en        <= 1'b1;
      notch_mode   <= 1'b0;
      hp_fc_sel    <= 2'd1;             // 0.5 Hz
      lp_fc_sel    <= 2'd2;             // 150 Hz
      notch_l      <= 5'd20;
      notch_rho    <= COMB_RHO_Q216[0];
      sat_clear    <= 1'b0;
      coef_addr    <= '0;
      coef_we      <= 1'b0;
      coef_wdata   <= '0;
    end else begin
      coef_we   <= 1'b0;
      sat_clear <= 1'b0;
      // CTRL[1] is a self-clearing soft reset: it flushes all filter state
      // and reloads coefficients, then drops on the next cycle so one write
      // is enough.
      if (ctrl_soft_reset) ctrl_soft_reset <= 1'b0;

      if (reg_wr) begin
        unique case (reg_waddr)
          8'h08: begin
            ctrl_enable     <= reg_wdata[0];
            ctrl_soft_reset <= reg_wdata[1];
            ctrl_bypass     <= reg_wdata[2];
          end
          8'h40: begin
            hp_en      <= reg_wdata[0];
            notch_en   <= reg_wdata[1];
            lp_en      <= reg_wdata[2];
          end
          8'h44: hp_fc_sel  <= reg_wdata[1:0];
          8'h48: notch_mode <= reg_wdata[0];
          8'h4C: notch_l    <= reg_wdata[4:0];
          8'h50: notch_rho  <= reg_wdata[17:0];
          8'h54: lp_fc_sel  <= reg_wdata[1:0];
          8'h58: coef_addr  <= reg_wdata[5:0];
          8'h5C: begin
            coef_wdata <= reg_wdata[17:0];
            coef_we    <= 1'b1;
          end
          8'h60: sat_clear  <= 1'b1;      // write clears, applied by the FSM
          default: ;
        endcase
      end
    end
  end

  //--------------------------------------------------------------------------
  // FIR coefficient ROM, with an indirect override port.
  //
  // An 81-tap bank cannot be laid out as one register per tap inside a 256-byte
  // AXI window, so writes go through COEF_ADDR/COEF_DATA instead.  Reset loads
  // bank 150 Hz at reset; a write to LP_FC_SEL (0x54) reloads it.
  //--------------------------------------------------------------------------
  logic signed [17:0] fir_coef [0:NTAPS-1];

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      for (int i = 0; i < NTAPS; i++) fir_coef[i] <= FIR_LP_Q216[2][i];
    end else if (reg_wr && reg_waddr == 8'h54) begin
      for (int i = 0; i < NTAPS; i++) fir_coef[i] <= FIR_LP_Q216[reg_wdata[1:0]][i];
    end else if (coef_we && coef_addr < NTAPS) begin
      fir_coef[coef_addr] <= coef_wdata;
    end
  end

  //--------------------------------------------------------------------------
  // Per-channel state
  //--------------------------------------------------------------------------
  logic signed [31:0] dc_x1 [0:NCH-1], dc_y1 [0:NCH-1];

  // Unreset LUTRAM histories avoid one reset/control set per stored bit.
  (* ram_style = "distributed" *) logic signed [31:0] cb_x_mem [0:NCH*COMB_MAX_L-1];
  (* ram_style = "distributed" *) logic signed [31:0] cb_y_mem [0:NCH*COMB_MAX_L-1];
  logic [4:0] cb_p [0:NCH-1];
  logic [5:0] cb_valid [0:NCH-1];
  logic signed [31:0] cb_xl, cb_yl;

  logic signed [31:0] nb_x1 [0:NCH-1][0:1], nb_x2 [0:NCH-1][0:1];
  logic signed [31:0] nb_y1 [0:NCH-1][0:1], nb_y2 [0:NCH-1][0:1];

  // Two identical LUTRAM copies implement the FIR delay line's two reads.
  (* ram_style = "distributed" *) logic signed [31:0] fr_mem_a [0:NCH*NTAPS-1];
  (* ram_style = "distributed" *) logic signed [31:0] fr_mem_b [0:NCH*NTAPS-1];
  logic [6:0] fr_p [0:NCH-1];
  logic [6:0] fr_valid [0:NCH-1];
  // Latch FIR address context once per channel. This removes the global ch
  // mux from the repeated FIR read path and makes the delay memory BRAM-ready.
  logic [2:0] fir_ch;
  logic [6:0] fir_p_r, fir_valid_r;

  logic [4:0] comb_len;
  always_comb comb_len = (notch_l > COMB_MAX_L) ? 5'(COMB_MAX_L) : notch_l;

  //--------------------------------------------------------------------------
  // Sequencer
  //
  // Two states per stage: one to assert start, one to wait for done.  The
  // stage takes several cycles internally and returns a single-cycle done.
  //
  // `start` is decoded COMBINATIONALLY from the state rather than registered.
  // Registered, it would still be high on the cycle the wait state begins --
  // and at that moment the stage has not accepted yet, so a done left over
  // from the PREVIOUS conversion is still asserted and the wait state would
  // commit on it immediately.  Driving it from the state decode means the
  // accept and the clearing of done both happen at the same clock edge.
  //--------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE,
    S_DC_A,  S_DC_W,                      // DC blocker
    S_CB_A,  S_CB_W,                      // comb notch
    S_BQ_A0, S_BQ_W0, S_BQ_A1, S_BQ_W1,   // biquad notch, two sections
    S_FIRIN, S_FIRRD, S_FIRMUL, S_FIRACC, S_FIRRND,
    S_NEXT, S_EMIT
  } state_t;

  state_t      state;
  logic [2:0]  ch;                       // channel counter, 0..7
  logic [6:0]  tap;                      // FIR tap / pair index
  // beat context carried from input to output
  logic [255:0] in_tdata;
  logic [127:0] in_tuser;
  logic         in_tlast;
  logic [255:0] out_tdata;

  logic signed [31:0] x_cur;             // running sample through the cascade
  logic signed [31:0] y_cur;
  logic signed [31:0] nb_mid;            // section 0's output, held for section 1

  // FIR tap values, read one cycle and multiplied the next.  The history is
  // a 8 x 81 x 32 flip-flop array, so the read alone is a 648:1 mux; putting
  // it in front of the multipliers in the same cycle is what made the FIR
  // stage long.
  logic signed [31:0] xa_r, xb_r;
  logic signed [17:0] ca_r, cb_r;
  logic signed [ACC_W-1:0] pa_r, pb_r, fir_acc;

  assign s_axis_ecg8_tready = (state == S_IDLE) && ctrl_enable;
  assign m_axis_ecg8_tvalid = (state == S_EMIT);
  assign m_axis_ecg8_tdata  = out_tdata;
  assign m_axis_ecg8_tuser  = in_tuser;   // frame idx + all quality fields
  assign m_axis_ecg8_tlast  = in_tlast;

  //--------------------------------------------------------------------------
  // Stage instances.  Each owns the arithmetic of one stage and sequences it
  // internally; the combinational statement of the same arithmetic lives in
  // ecg_filter_stages.sv and is what model/fixedpt.py mirrors.
  //--------------------------------------------------------------------------
  logic dc_start, dc_done, dc_sat;
  logic cb_start, cb_done, cb_sat;
  logic bq_start, bq_done, bq_sat, bq_sel;
  // cb_out, not cb_y: the per-channel comb history array is already cb_y.
  logic signed [31:0] dc_y, cb_out, bq_y;

  // Combinational, deliberately: see the note above the state typedef.
  assign dc_start = (state == S_DC_A);
  assign cb_start = (state == S_CB_A);
  assign bq_start = (state == S_BQ_A0) || (state == S_BQ_A1);
  assign bq_sel   = (state == S_BQ_A1) || (state == S_BQ_W1);

  ecg_dc_mc u_dc (
      .aclk, .aresetn,
      .start(dc_start), .busy(), .done(dc_done),
      .x(x_cur), .x1(dc_x1[ch]), .y1(dc_y1[ch]),
      .delta_q(32'(signed'(HP_DELTA_Q216[hp_fc_sel]))),
      .y(dc_y), .sat(dc_sat)
  );

  assign cb_xl = (cb_valid[ch] < {1'b0, comb_len}) ? 32'sd0 : cb_x_mem[ch*COMB_MAX_L + cb_p[ch]];
  assign cb_yl = (cb_valid[ch] < {1'b0, comb_len}) ? 32'sd0 : cb_y_mem[ch*COMB_MAX_L + cb_p[ch]];

  ecg_comb_notch_mc u_cb (
      .aclk, .aresetn,
      .start(cb_start), .busy(), .done(cb_done),
      .x(x_cur), .xl(cb_xl), .yl(cb_yl),
      .rho_q(32'(signed'(notch_rho))), .y(cb_out), .sat(cb_sat)
  );

  // Section 1's input is section 0's OUTPUT, not the stage input.  Wiring
  // both sections to x_cur runs two independent notches in parallel and
  // takes the second one's result, which is close enough to look plausible
  // (the errors were tens of LSB) but is not the cascade the model computes.
  // One engine runs both sections in turn; bq_sel picks which coefficients
  // and which state it sees.
  logic signed [31:0] bq_x, bq_x1, bq_x2, bq_y1, bq_y2;
  logic signed [17:0] bq_b0, bq_b1, bq_b2, bq_A1, bq_A2;

  always_comb begin
    if (!bq_sel) begin
      bq_x  = x_cur;        bq_x1 = nb_x1[ch][0]; bq_x2 = nb_x2[ch][0];
      bq_y1 = nb_y1[ch][0]; bq_y2 = nb_y2[ch][0];
      bq_b0 = NOTCH_COEF_Q216[0][0]; bq_b1 = NOTCH_COEF_Q216[0][1];
      bq_b2 = NOTCH_COEF_Q216[0][2]; bq_A1 = NOTCH_COEF_Q216[0][3];
      bq_A2 = NOTCH_COEF_Q216[0][4];
    end else begin
      bq_x  = nb_mid;       bq_x1 = nb_x1[ch][1]; bq_x2 = nb_x2[ch][1];
      bq_y1 = nb_y1[ch][1]; bq_y2 = nb_y2[ch][1];
      bq_b0 = NOTCH_COEF_Q216[1][0]; bq_b1 = NOTCH_COEF_Q216[1][1];
      bq_b2 = NOTCH_COEF_Q216[1][2]; bq_A1 = NOTCH_COEF_Q216[1][3];
      bq_A2 = NOTCH_COEF_Q216[1][4];
    end
  end

  ecg_biquad_mc u_bq (
      .aclk, .aresetn,
      .start(bq_start), .busy(), .done(bq_done),
      .x(bq_x), .x1(bq_x1), .x2(bq_x2), .y1(bq_y1), .y2(bq_y2),
      .b0(bq_b0), .b1(bq_b1), .b2(bq_b2), .A1(bq_A1), .A2(bq_A2),
      .y(bq_y), .sat(bq_sat)
  );

  // FIR tap addressing.
  //
  // Keyed off fr_p[ch], which points at the slot the NEXT sample will occupy,
  // so the just-written x[n] sits at fr_p-1 and x[n-k] at fr_p-1-k.
  //
  // Explicit modulo, not a mask: NTAPS is 81, which is not a power of two, so
  // `& (NTAPS-1)` would silently alias taps onto each other.  p and k are both
  // below 81, so one conditional add covers the underflow.
  function automatic logic [6:0] fir_idx(input logic [6:0] p, input logic [6:0] k);
    logic [7:0] t;
    // 8-bit intermediate on purpose: computing p-k in 7 bits wraps a
    // negative difference to a large positive value, and the +81 correction
    // then lands past the end of the delay line -- an out-of-range read returns X,
    // which shows up as the output stream going unknown rather than as a
    // wrong number.
    t = {1'b0, p} - {1'b0, k};
    fir_idx = t[7] ? 7'(t + 8'd81) : t[6:0];
  endfunction

  logic signed [31:0] fir_xa, fir_xb;
  logic signed [17:0] fir_ca, fir_cb;
  logic signed [ACC_W-1:0] fir_rnd;
  logic [6:0]         fir_ka, fir_kb;
  logic               fir_last, fir_rnd_sat;

  // The last pair of an odd tap count has no second tap: 2*40+2 = 82 exceeds
  // the 81-entry history.  Zeroing the COEFFICIENT is not enough -- an
  // out-of-range read returns X, and 0 * X is X, so the accumulator goes
  // unknown and the whole output with it.  The INDEX must be kept in range
  // too, which is why the guard is here and not only on fir_cb.
  assign fir_ka = 7'(2 * tap + 1);
  assign fir_kb = 7'(2 * tap + 2);
  assign fir_xa = (fir_ka <= fir_valid_r && fir_ka <= NTAPS)
                ? fr_mem_a[fir_ch*NTAPS + fir_idx(fir_p_r, fir_ka)] : 32'sd0;
  assign fir_xb = (fir_kb <= fir_valid_r && fir_kb <= NTAPS)
                ? fr_mem_b[fir_ch*NTAPS + fir_idx(fir_p_r, fir_kb)] : 32'sd0;
  assign fir_ca = fir_coef[2 * tap];
  assign fir_cb = (2 * tap + 1 < NTAPS) ? fir_coef[2 * tap + 1] : 18'sd0;
  assign fir_last = (tap == 7'(FIR_HALF - 1));

  // Once, not three times: as an expression inside three separate
  // assignments Vivado would share the cone anyway, but xsim would evaluate
  // it three times per cycle and the intent is clearer named.
  always_comb begin
    fir_rnd     = rshift_round(fir_acc, Q_COEF_FRAC);
    fir_rnd_sat = sat_hit(fir_rnd);
  end

  //--------------------------------------------------------------------------
  // FSM
  //--------------------------------------------------------------------------
  integer ci, si;
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      state      <= S_IDLE;
      ch         <= '0;
      tap        <= '0;
      x_cur      <= '0;
      y_cur      <= '0;
      nb_mid     <= '0;
      fir_acc    <= '0;
      pa_r       <= '0;
      pb_r       <= '0;
      xa_r       <= '0;
      xb_r       <= '0;
      ca_r       <= '0;
      cb_r       <= '0;
      sat_count  <= '0;
      out_tdata  <= '0;
      in_tdata   <= '0;
      in_tuser   <= '0;
      in_tlast   <= 1'b0;
      for (ci = 0; ci < NCH; ci++) begin
        dc_x1[ci] <= '0;
        dc_y1[ci] <= '0;
        cb_p[ci]     <= '0;
        cb_valid[ci] <= '0;
        fr_p[ci]     <= '0;
        fr_valid[ci] <= '0;
        for (si = 0; si < 2; si++) begin
          nb_x1[ci][si] <= '0;  nb_x2[ci][si] <= '0;
          nb_y1[ci][si] <= '0;  nb_y2[ci][si] <= '0;
        end
      end
    end else if (ctrl_soft_reset) begin
      state <= S_IDLE;
      ch    <= '0;
      tap   <= '0;
      fir_ch <= '0; fir_p_r <= '0; fir_valid_r <= '0;
      for (ci = 0; ci < NCH; ci++) begin
        dc_x1[ci] <= '0;
        dc_y1[ci] <= '0;
        cb_p[ci]     <= '0;
        cb_valid[ci] <= '0;
        fr_p[ci]     <= '0;
        fr_valid[ci] <= '0;
        for (si = 0; si < 2; si++) begin
          nb_x1[ci][si] <= '0;  nb_x2[ci][si] <= '0;
          nb_y1[ci][si] <= '0;  nb_y2[ci][si] <= '0;
        end
      end
    end else begin
      case (state)
        S_IDLE: begin
          if (s_axis_ecg8_tvalid && s_axis_ecg8_tready) begin
            in_tdata <= s_axis_ecg8_tdata;
            in_tuser <= s_axis_ecg8_tuser;
            in_tlast <= s_axis_ecg8_tlast;
            ch       <= 3'd0;
            x_cur    <= s_axis_ecg8_tdata[31:0];   // CH1
            state    <= S_DC_A;
          end
        end

        // ---- DC blocker, one channel at a time ----
        S_DC_A: state <= S_DC_W;
        S_DC_W: begin
          if (dc_done) begin
            if (hp_en && !ctrl_bypass) begin
              x_cur     <= dc_y;
              dc_x1[ch] <= x_cur;
              dc_y1[ch] <= dc_y;
              if (dc_sat) sat_count <= sat_count + 1'b1;
            end
            state <= (notch_en && !ctrl_bypass)
                     ? (notch_mode ? S_BQ_A0 : S_CB_A)
                     : S_FIRIN;
          end
        end

        // ---- comb notch ----
        S_CB_A: state <= S_CB_W;
        S_CB_W: begin
          if (cb_done) begin
            x_cur <= cb_out;
            cb_x_mem[ch*COMB_MAX_L + cb_p[ch]] <= x_cur;
            cb_y_mem[ch*COMB_MAX_L + cb_p[ch]] <= cb_out;
            cb_p[ch] <= (cb_p[ch] + 5'd1 == comb_len) ? 5'd0 : cb_p[ch] + 5'd1;
            if (cb_valid[ch] < {1'b0, comb_len}) cb_valid[ch] <= cb_valid[ch] + 1'b1;
            if (cb_sat) sat_count <= sat_count + 1'b1;
            state <= S_FIRIN;
          end
        end

        // ---- biquad notch, section 0 ----
        S_BQ_A0: state <= S_BQ_W0;
        S_BQ_W0: begin
          if (bq_done) begin
            nb_x2[ch][0] <= nb_x1[ch][0];  nb_x1[ch][0] <= x_cur;
            nb_y2[ch][0] <= nb_y1[ch][0];  nb_y1[ch][0] <= bq_y;
            nb_mid       <= bq_y;
            if (bq_sat) sat_count <= sat_count + 1'b1;
            state <= S_BQ_A1;
          end
        end

        // ---- biquad notch, section 1 ----
        S_BQ_A1: state <= S_BQ_W1;
        S_BQ_W1: begin
          if (bq_done) begin
            nb_x2[ch][1] <= nb_x1[ch][1];  nb_x1[ch][1] <= nb_mid;
            nb_y2[ch][1] <= nb_y1[ch][1];  nb_y1[ch][1] <= bq_y;
            x_cur        <= bq_y;
            if (bq_sat) sat_count <= sat_count + 1'b1;
            state <= S_FIRIN;
          end
        end

        // ---- FIR entry: commit the sample, reset the accumulator ----
        //
        // The history write has to happen BEFORE the tap loop reads it:
        // writing it after the sum leaves h[0]*x[n] out of the result
        // entirely, and a single-sample input then filters to exactly zero.
        // x_cur, not the stage input -- by now it holds the notch output, or
        // the DC output if the notch is bypassed, which is what the FIR must
        // see.  Reading the pre-notch sample here fills the delay line with
        // unfiltered data and the cascade is silently wrong even though each
        // stage is individually correct.
        S_FIRIN: begin
          tap     <= '0;
          fir_acc <= '0;
          fir_ch  <= ch;
          fir_p_r <= fr_p[ch];
          fir_valid_r <= (fr_valid[ch] < NTAPS) ? fr_valid[ch] + 1'b1 : fr_valid[ch];
          if (lp_en && !ctrl_bypass) begin
            fr_mem_a[ch*NTAPS + fr_p[ch]] <= x_cur;
            fr_mem_b[ch*NTAPS + fr_p[ch]] <= x_cur;
            fr_p[ch] <= (fr_p[ch] == 7'd80) ? 7'd0 : fr_p[ch] + 7'd1;
            if (fr_valid[ch] < NTAPS) fr_valid[ch] <= fr_valid[ch] + 1'b1;
            state <= S_FIRRD;
          end else begin
            state <= S_NEXT;
          end
        end

        // ---- FIR: one tap pair per three cycles ----
        //
        // The LAST pair is accumulated here too, whereas the single-cycle
        // version evaluated it separately in the rounding state.  Same
        // expression, same 48-bit accumulator, so the result is identical.
        S_FIRRD: begin
          xa_r <= fir_xa;
          xb_r <= fir_xb;
          ca_r <= fir_ca;
          cb_r <= fir_cb;
          state <= S_FIRMUL;
        end
        S_FIRMUL: begin
          pa_r <= ACC_W'(signed'(ca_r)) * ACC_W'(signed'(op25(xa_r)));
          pb_r <= ACC_W'(signed'(cb_r)) * ACC_W'(signed'(op25(xb_r)));
          state <= S_FIRACC;
        end
        S_FIRACC: begin
          fir_acc <= fir_acc + pa_r + pb_r;
          if (fir_last) state <= S_FIRRND;
          else begin
            tap   <= tap + 7'd1;
            state <= S_FIRRD;
          end
        end

        // One rounding for the whole 81-tap sum.  Rounding per tap would
        // inject 81 half-LSB errors instead of one.
        S_FIRRND: begin
          y_cur <= sat_int(fir_rnd);
          if (fir_rnd_sat) sat_count <= sat_count + 1'b1;
          state <= S_NEXT;
        end

        // ---- carry this channel's result into the output word ----
        S_NEXT: begin
          out_tdata[ch * 32 +: 32] <= lp_en ? y_cur : x_cur;
          if (ch == 3'd7) state <= S_EMIT;
          else begin
            ch    <= ch + 3'd1;
            x_cur <= in_tdata[(ch + 3'd1) * 32 +: 32];
            state <= S_DC_A;
          end
        end

        S_EMIT: begin
          if (m_axis_ecg8_tready) state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase

      // After the case, so a clear wins over an increment in the same cycle.
      // One always_ff drives sat_count; see its declaration.
      if (sat_clear) sat_count <= '0;
    end
  end

  //--------------------------------------------------------------------------
  // Register readback
  //--------------------------------------------------------------------------
  logic [31:0] frame_cnt;
  always_ff @(posedge aclk) begin
    if (!aresetn) frame_cnt <= '0;
    else if (m_axis_ecg8_tvalid && m_axis_ecg8_tready && in_tlast)
      frame_cnt <= frame_cnt + 1'b1;
  end

  always_comb begin
    reg_rdata = 32'h0;
    unique case (reg_raddr)
      8'h00: reg_rdata = IP_ID;
      8'h04: reg_rdata = IP_VERSION;
      8'h08: reg_rdata = {29'h0, ctrl_bypass, 1'b0, ctrl_enable};
      8'h0C: reg_rdata = {31'h0, (sat_count != 0)};
      8'h18: reg_rdata = frame_cnt;
      8'h1C: reg_rdata = SCRATCH_PATTERN;
      8'h40: reg_rdata = {29'h0, lp_en, notch_en, hp_en};
      8'h44: reg_rdata = {30'h0, hp_fc_sel};
      8'h48: reg_rdata = {31'h0, notch_mode};
      8'h4C: reg_rdata = {27'h0, notch_l};
      8'h50: reg_rdata = {14'h0, notch_rho};
      8'h54: reg_rdata = {30'h0, lp_fc_sel};
      8'h58: reg_rdata = {26'h0, coef_addr};
      8'h60: reg_rdata = sat_count;
      8'h64: reg_rdata = LAT_FILTER_SAMPLES;
      default: reg_rdata = 32'h0;
    endcase
  end

endmodule

