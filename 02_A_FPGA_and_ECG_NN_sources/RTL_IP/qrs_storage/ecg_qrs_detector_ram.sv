//=============================================================================
// ecg_qrs_detector.sv -- one Pan-Tompkins detection chain.
//
// Instantiated twice by ecg_qrs_ip, once per selected lead.  Mirrors
// model/qrs_ref.py:PanTompkins.  The model is FLOAT, so this is a faithful
// fixed-point realisation rather than a bit-exact twin -- IP5 is verified by
// metrics (Se, +P, peak-position error, latency), which is what the task book
// asks for, not by sample-for-sample comparison.
//
// Metric arithmetic -- see ecg_qrs_ip.sv for why this is not Q1.23:
//   d     derivative, Q1.23, |d| <= 2^24
//   sq_s  (d*d) >> 16,                <= 2^32
//   met   (sum of 64 sq_s) >> 6,      <= 2^32, 48-bit accumulator
// Every threshold operation is scale-invariant, so the constant 2^-16 folded
// in changes nothing about the algorithm.
//
// TIMING
//
// This chain used to compute everything from the band-pass output through the
// square, the moving-window sum, the threshold comparison and the candidacy
// decision in ONE cycle.  Synthesis reported -48.5 ns of slack, and the
// reported path shows why: it ran
//
//   biquad 0 -> biquad 1 -> derivative -> square (DSP48E1) -> moving-window
//   ring -> 48-bit max compare -> candidate -> register enable
//
// -- two biquads and the whole decision cone, 58 ns, against a 10 ns period.
// (The usage guide previously attributed this to the biquad MAC alone.  That
// was wrong: fixing the biquads removes only the first 41 ns.)
//
// The chain is now sequenced one arithmetic operation per cycle.  Nothing
// about the arithmetic changed, only when it happens.
//
// Sequencing note -- the candidacy DECISION is latched in S_DECIDE and only
// consumed in S_COMMIT.  Re-testing "met > thresh" later looks equivalent but
// is not: met and thresh are state that the intervening cycles advance, so
// the commit would act on numbers the decision was never made from.
//
// The same reasoning is why the decision reads met_r / thresh_r / sb_lim_r,
// all registered earlier in this beat, rather than recomputing them.  n_samples
// is still incremented in S_DECIDE, so every comparison there sees exactly the
// value it saw when this was a single cycle.
//=============================================================================

`include "ecg_frame_pkg.sv"
`include "ecg_qrs_coeffs.svh"

module ecg_qrs_detector_ram (
    input  logic                aclk,
    input  logic                aresetn,

    input  logic                enable,
    input  logic                soft_reset,
    input  logic                bpf_en,
    input  logic                searchback_en,
    input  logic [15:0]         learn_samples,

    input  logic [31:0]         frame_idx,
    input  logic                beat_valid,
    input  logic signed [31:0]  x,

    output logic                fire,
    output logic [31:0]         fire_frame,
    output logic [15:0]         fire_rr,
    output logic                sat,
    output logic                searchback_hit,
    output logic                ready,       // can accept a beat this cycle
    output logic [47:0]         dbg_met,
    output logic [47:0]         dbg_thresh
);

  import ecg_frame_pkg::*;

  localparam int MWI_LEN      = 128;  // 128 ms moving window at 1000 SPS
  localparam int REFRACTORY   = 200;
  localparam int FID_LEN      = 100;
  localparam int LEARN_SETTLE = 500;
  // 16 keeps sq_s at exactly 48 bits, so the slice needs no truncation and
  // the 128-sample sum stays below 2^47.
  //
  // Tried 8 to sharpen the small-signal floor: it changed nothing.  The value
  // is arbitrary because every threshold operation downstream is
  // scale-invariant -- the whole chain rescales with it.  Worth recording so
  // nobody spends another afternoon on it.
  localparam int SQ_SHIFT     = 16;

  // Declared before the instances that consume it: Verilog implicitly
  // declares an undeclared identifier as a 1-bit wire, which would silently
  // truncate the sample to its LSB.
  logic signed [31:0] bp_in;             // sample being filtered this beat

  //==========================================================================
  // Band-pass 5-15 Hz: two biquad sections, run in turn through one engine
  //==========================================================================
  logic bq_start, bq_sel, bq_done, bq_sat;
  logic signed [31:0] bq_y;
  logic signed [31:0] bq_x, bq_x1, bq_x2, bq_y1, bq_y2;
  logic signed [17:0] bq_b0, bq_b1, bq_b2, bq_A1, bq_A2;

  logic signed [31:0] bp_x1 [0:1], bp_x2 [0:1], bp_y1 [0:1], bp_y2 [0:1];
  logic signed [31:0] y_bq0_r, bp_out_r;

  always_comb begin
    if (!bq_sel) begin
      bq_x  = bp_in;        bq_x1 = bp_x1[0]; bq_x2 = bp_x2[0];
      bq_y1 = bp_y1[0];     bq_y2 = bp_y2[0];
      bq_b0 = BPF_Q216[0][0]; bq_b1 = BPF_Q216[0][1]; bq_b2 = BPF_Q216[0][2];
      bq_A1 = BPF_Q216[0][3]; bq_A2 = BPF_Q216[0][4];
    end else begin
      bq_x  = y_bq0_r;      bq_x1 = bp_x1[1]; bq_x2 = bp_x2[1];
      bq_y1 = bp_y1[1];     bq_y2 = bp_y2[1];
      bq_b0 = BPF_Q216[1][0]; bq_b1 = BPF_Q216[1][1]; bq_b2 = BPF_Q216[1][2];
      bq_A1 = BPF_Q216[1][3]; bq_A2 = BPF_Q216[1][4];
    end
  end

  ecg_biquad_mc u_bq (
      .aclk, .aresetn,
      .start(bq_start), .busy(), .done(bq_done),
      .x(bq_x), .x1(bq_x1), .x2(bq_x2), .y1(bq_y1), .y2(bq_y2),
      .b0(bq_b0), .b1(bq_b1), .b2(bq_b2), .A1(bq_A1), .A2(bq_A2),
      .y(bq_y), .sat(bq_sat)
  );

  logic signed [31:0] bp_cur;            // band-pass output (or raw if bypassed)
  always_comb bp_cur = bpf_en ? bp_out_r : bp_in;

  //==========================================================================
  // Derivative:  y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
  //==========================================================================
  logic signed [31:0] dv_h [0:4];        // [0] = x[n-1], [3] = x[n-4]

  // dv_c is the combinational result and dv_r is its register.  Leaving dv_r
  // a wire would put the derivative AND the square in the same cycle, which
  // is exactly the pairing this split exists to break.
  logic signed [ACC_W-1:0] dv_acc;
  logic signed [31:0]      dv_c, dv_r;
  always_comb begin
    dv_acc = 2 * ACC_W'(signed'(bp_cur))
           + ACC_W'(signed'(dv_h[0]))
           - ACC_W'(signed'(dv_h[2]))
           - 2 * ACC_W'(signed'(dv_h[3]));
    dv_c   = 32'(dv_acc >>> 3);
  end

  //==========================================================================
  // Square + moving-window integration
  //==========================================================================
  logic signed [63:0] sq_full;
  logic        [47:0] sq_c, sq_r;

  always_comb begin
    sq_full = 64'(signed'(dv_r)) * 64'(signed'(dv_r));
    sq_c    = sq_full[63:SQ_SHIFT];
  end

  (* ram_style = "distributed" *) logic [47:0] mwi_ring [0:MWI_LEN-1];
  logic [7:0] mwi_valid_count;
  logic [6:0]  mwi_p;
  logic [47:0] mwi_sum, oldest_r, sum_next;
  // Signed, because thresh_r goes negative while the trackers settle and an
  // unsigned comparison against it would read as a huge positive threshold
  // and never fire.  The metric itself is always non-negative here, so this
  // costs nothing and matches the signed comparison the inline form made.
  logic signed [47:0] met_r;

  // The two 48-bit adds of the sliding sum.  Read it only in S_MWI: after
  // that state mwi_sum holds the new value and re-evaluating this expression
  // would yield the NEXT sample's sum, which is why the decision reads the
  // registered met_r instead.
  always_comb sum_next = mwi_sum + sq_r - oldest_r;

  //==========================================================================
  // Adaptive threshold
  //==========================================================================
  logic signed [47:0] spki, npki, thresh_r;
  logic signed [47:0] sb_lim_r;

  //==========================================================================
  // Fiducial ring: |x| of the last FID_LEN band-passed samples
  //
  // MAGNITUDE, not the signed sample.  The search compares |x| and nothing
  // downstream wants the signed value, so storing the magnitude moves the
  // 32-bit negate to the write side -- where it runs alongside everything else
  // -- instead of onto the read-to-compare path.  That negate behind the 50:1
  // ring mux was the last path over budget, at 9.64 ns against a 10 ns period.
  //==========================================================================
  (* ram_style = "distributed" *) logic [31:0] fid_ring [0:FID_LEN-1];
  logic [7:0] fid_valid_count;
  logic [31:0] fid_read;
  logic [6:0]  fid_p;

  //==========================================================================
  // Sequencer
  //
  // One state per arithmetic step.  Each of S_SQ .. S_THR latches a value that
  // the decision below then reads, so no single cycle carries more than one
  // multiplier, one wide add or one comparator chain.
  //==========================================================================
  typedef enum logic [3:0] {
    S_IDLE,
    S_BQ_A0, S_BQ_W0, S_BQ_A1, S_BQ_W1,   // band-pass, two sections
    S_DERIV,                              // derivative
    S_SQ,                                 // square, and read the ring's oldest
    S_MWI,                                // sliding-window sum and metric
    S_THR,                                // threshold and search-back limit
    S_DECIDE,                             // candidate decision, trackers
    S_FID,                                // fiducial refinement
    S_COMMIT
  } state_t;

  state_t      state;

  // Combinational start, from the state decode: registered, it would still be
  // high on the first cycle of the wait state, before the engine has accepted,
  // and the stale done from the previous section would be read as this one's.
  assign bq_start = (state == S_BQ_A0) || (state == S_BQ_A1);
  assign bq_sel   = (state == S_BQ_A1) || (state == S_BQ_W1);

  logic [6:0]  fid_cnt;
  logic [6:0]  fid_wr;
  logic [6:0]  fid_best_off;
  logic [6:0]  fid_span;
  logic [31:0] fid_best_mag;

  logic [31:0] cur_frame;
  logic        cand;                     // latched candidacy decision
  logic [47:0] cand_met;                 // the metric it was decided from

  logic [31:0] learn_cnt;
  logic [47:0] learn_max;
  // Running exponential mean, NOT a sum with a division at the end.
  // `learn_sum / learn_cnt` is a 56-bit combinational divider: it synthesised
  // to a 489-level carry chain and a 132 ns path, i.e. -122 ns of slack at
  // 100 MHz.  The huge per-sample cycle budget does not help, because the
  // division is one long combinational cone rather than something that can be
  // spread over cycles.  An exponential mean needs no divider at all and
  // estimates the same noise floor.
  logic signed [47:0] learn_mean;
  logic [31:0] n_samples;
  logic [31:0] last_r;
  logic [31:0] rr_avg;
  logic        have_last;

  logic        fire_r, sb_hit_r, sat_r;
  logic [31:0] fire_frame_r;
  logic [15:0] fire_rr_r;

  logic in_refractory;
  always_comb in_refractory = have_last && ((n_samples - last_r) <= REFRACTORY);

  logic in_learn;
  always_comb in_learn = (n_samples <= {16'd0, learn_samples});

  // Histories are written in order from address zero after either reset.
  // Invalid slots read as zero, exactly as in R9's fully reset register arrays.
  // Separate, reset-free write processes allow LUTRAM inference without adding
  // a read cycle or changing the detector's externally observable timing.
  assign fid_read = ((fid_valid_count == FID_LEN) || (fid_wr < fid_valid_count))
                  ? fid_ring[fid_wr] : 32'd0;
  always_ff @(posedge aclk) begin
    if (aresetn && !soft_reset && enable && state == S_MWI)
      mwi_ring[mwi_p] <= sq_r;
    if (aresetn && !soft_reset && enable && state == S_DECIDE)
      fid_ring[fid_p] <= bp_cur[31] ? 32'(0 - bp_cur) : 32'(bp_cur);
  end

  integer i;
  always_ff @(posedge aclk) begin
    if (!aresetn || soft_reset) begin
      state        <= S_IDLE;
      fid_cnt      <= '0;
      // fid_p MUST be reset.  Left uninitialised it is X, so the history
      // write `fid_ring[fid_p] <= bp_cur` hits an unknown index and never
      // lands -- the whole fiducial refinement is silently dead, and every
      // event is reported at the threshold crossing instead of the R peak
      // (measured ~56 samples early, which is a 128-sample MWI lag).
      fid_p        <= '0;
      fid_wr       <= 7'(FID_LEN - 1);
      fid_span     <= 7'(FID_LEN - 1);
      fid_best_off <= '0;
      // 0, not the most-negative value.  The search compares MAGNITUDES now,
      // and |32'sh8000_0000| is 2^31 -- no real band-pass sample can exceed
      // that, so the argmax would never update and every event would report
      // the unrefined threshold crossing.
      fid_best_mag <= 32'sd0;
      cur_frame    <= '0;
      cand         <= 1'b0;
      cand_met     <= '0;
      mwi_p        <= '0;
      mwi_sum      <= '0;
      oldest_r     <= '0;
      met_r        <= '0;
      sq_r         <= '0;
      spki         <= '0;
      npki         <= '0;
      thresh_r     <= '0;
      sb_lim_r     <= '0;
      learn_cnt    <= '0;
      learn_max    <= '0;
      learn_mean   <= '0;
      n_samples    <= '0;
      last_r       <= '0;
      rr_avg       <= '0;
      have_last    <= 1'b0;
      fire_r       <= 1'b0;
      sb_hit_r     <= 1'b0;
      sat_r        <= 1'b0;
      fire_frame_r <= '0;
      fire_rr_r    <= '0;
      y_bq0_r      <= '0;
      bp_out_r     <= '0;
      for (i = 0; i < 2; i++) begin
        bp_x1[i] <= '0;  bp_x2[i] <= '0;
        bp_y1[i] <= '0;  bp_y2[i] <= '0;
      end
      for (i = 0; i < 5; i++) dv_h[i] <= '0;
      mwi_valid_count <= '0;
      fid_valid_count <= '0;
    end else begin
      fire_r <= 1'b0;

      if (!enable) begin
        state <= S_IDLE;
      end else begin
        case (state)
          S_IDLE: begin
            if (beat_valid) begin
              cur_frame <= frame_idx;
              bp_in     <= x;
              state     <= bpf_en ? S_BQ_A0 : S_DERIV;
            end
          end

          // ---- band-pass, section 0 ----
          S_BQ_A0: state <= S_BQ_W0;
          S_BQ_W0: begin
            if (bq_done) begin
              bp_x2[0] <= bp_x1[0];  bp_x1[0] <= bp_in;
              bp_y2[0] <= bp_y1[0];  bp_y1[0] <= bq_y;
              y_bq0_r  <= bq_y;
              if (bq_sat) sat_r <= 1'b1;
              state <= S_BQ_A1;
            end
          end

          // ---- band-pass, section 1 ----
          S_BQ_A1: state <= S_BQ_W1;
          S_BQ_W1: begin
            if (bq_done) begin
              bp_x2[1] <= bp_x1[1];  bp_x1[1] <= y_bq0_r;
              bp_y2[1] <= bp_y1[1];  bp_y1[1] <= bq_y;
              bp_out_r <= bq_y;
              if (bq_sat) sat_r <= 1'b1;
              state <= S_DERIV;
            end
          end

          // ---- derivative ----
          // History is committed here, not later, so that dv_acc reads the
          // same values the single-cycle version read.
          S_DERIV: begin
            dv_h[4] <= dv_h[3];  dv_h[3] <= dv_h[2];
            dv_h[2] <= dv_h[1];  dv_h[1] <= dv_h[0];
            dv_h[0] <= bp_cur;
            dv_r    <= dv_c;
            state   <= S_SQ;
          end

          // ---- square, and read the ring slot about to be overwritten ----
          S_SQ: begin
            sq_r     <= sq_c;
            oldest_r <= (mwi_valid_count == MWI_LEN) ? mwi_ring[mwi_p] : 48'd0;
            state    <= S_MWI;
          end

          // ---- sliding-window sum ----
          S_MWI: begin
            mwi_sum        <= sum_next;
            met_r          <= sum_next >> 7;
            if (mwi_valid_count < MWI_LEN) mwi_valid_count <= mwi_valid_count + 1'b1;
            mwi_p          <= (mwi_p == 7'(MWI_LEN - 1)) ? 7'd0 : mwi_p + 7'd1;
            state          <= S_THR;
          end

          // ---- threshold and search-back limit ----
          // Both are functions of the trackers, which do not move until
          // S_DECIDE or S_COMMIT, so computing them here is the same value
          // the decision would have computed inline.
          S_THR: begin
            thresh_r <= npki + ((spki - npki) >>> 2);
            // 32-bit context, as in the inline form this replaces: rr_avg is
            // an interval in samples, so rr_avg*53 never approaches 2^32 and
            // widening the multiply here would be a silent behaviour change.
            sb_lim_r <= 48'((rr_avg * 53) >> 5);
            state    <= S_DECIDE;
          end

          // ---- candidate decision ----
          S_DECIDE: begin
            logic is_cand;

            // histories
            // Magnitude: see the ring's declaration.  bp_cur is a register,
            // so this negate runs alongside the rest of S_DECIDE rather than
            // in front of the comparison in S_FID.
            if (fid_valid_count < FID_LEN) fid_valid_count <= fid_valid_count + 1'b1;
            fid_p <= (fid_p == 7'(FID_LEN - 1)) ? 7'd0 : fid_p + 7'd1;
            n_samples <= n_samples + 32'd1;

            cand_met <= met_r;

            is_cand = 1'b0;
            if (in_learn) begin
              // Settling is discarded before measuring: the band-pass starts
              // from zero state, so signal onset is a step and its transient
              // rings through the square and the integrator.  max/2 would
              // inherit that outlier directly into SPKI and start the
              // threshold far too high.
              // Accumulated in MET units (sum >> 6), not raw sum units.
              // Mixing the two scales puts spki exactly 2^6 above every value
              // it will ever be compared against, and the detector never
              // fires at all.
              if (n_samples > LEARN_SETTLE) begin
                if (met_r > learn_max) learn_max <= met_r;
                learn_mean <= learn_mean + ((met_r - learn_mean) >>> 6);
                learn_cnt  <= learn_cnt + 32'd1;
              end
              if (n_samples == {16'd0, learn_samples}) begin
                spki <= learn_max >> 1;
                npki <= learn_mean >>> 3;
              end
            end else if (met_r > thresh_r && !in_refractory) begin
              is_cand = 1'b1;
            end else if (!in_refractory) begin
              // NPKI must NOT update inside the refractory window: the signal
              // is still large there -- QRS and T wave decaying -- and folding
              // it into the noise estimate raises the threshold and swallows
              // the next beat.
              npki <= npki + ((met_r - npki) >>> 3);
              // (rr_avg * 53) >> 5 = 1.656, matching the model's 1.66 (an
              // 8-bit RR multiply would have given 1.625 and missed beats at
              // the margin).  sb_lim_r was formed in S_THR.
              if (searchback_en && have_last && rr_avg != 0 &&
                  (n_samples - last_r) > sb_lim_r &&
                  (met_r > (thresh_r >>> 1))) begin
                is_cand  = 1'b1;
                sb_hit_r <= 1'b1;
              end
            end

            cand <= is_cand;
            if (is_cand) begin
              fid_cnt      <= '0;
              fid_wr       <= fid_p;      // newest sample sits at fid_p-1
              // Never reach back past the previous beat: an unbounded search
              // would grab the preceding R wave and report a peak position
              // that is not in this beat at all.
              fid_span     <= (!have_last || (n_samples - last_r) >= FID_LEN)
                              ? 7'(FID_LEN - 1)
                              : 7'(n_samples - last_r);
              // 0, not the most-negative value -- see the reset block.
              fid_best_mag <= 32'sd0;
              fid_best_off <= '0;
              state        <= S_FID;
            end else begin
              state <= S_COMMIT;
            end
          end

          // ---- fiducial refinement, one ring entry per cycle ----
          //
          // Searched newest to oldest with a strict comparison, so the FIRST
          // maximum wins.
          //
          // The comparison is on MAGNITUDE, not signed value.  The detection
          // chain squares the derivative and is therefore polarity-blind, but
          // the fiducial peak is not: on an inverted lead (a chest lead with
          // an inverted R, which the two-lead fusion exists to cope with) the
          // signed maximum lands on a completely different part of the
          // complex -- measured 66 samples away -- and that wrong position
          // then suppressed the other detector's correct event through the
          // fusion refractory.
          S_FID: begin
            if (fid_read > fid_best_mag) begin
              fid_best_mag <= fid_read;
              fid_best_off <= fid_cnt;
            end
            if (fid_wr == 7'd0) fid_wr <= 7'(FID_LEN - 1);
            else                fid_wr <= fid_wr - 7'd1;

            if (fid_cnt >= fid_span) state   <= S_COMMIT;
            else                     fid_cnt <= fid_cnt + 7'd1;
          end

          // ---- commit ----
          S_COMMIT: begin
            logic [31:0] r_idx;
            r_idx = cur_frame - {26'd0, fid_best_off};

            if (cand) begin
              fire_r       <= 1'b1;
              fire_frame_r <= r_idx;
              fire_rr_r    <= (have_last && r_idx > last_r)
                              ? 16'(r_idx - last_r) : 16'd0;
              spki         <= spki + (($signed({4'd0, cand_met}) - spki) >>> 3);
              if (have_last && r_idx > last_r) begin
                if (rr_avg == 0) rr_avg <= r_idx - last_r;
                else             rr_avg <= rr_avg + (((r_idx - last_r) - rr_avg) >>> 3);
              end
              last_r    <= r_idx;
              have_last <= 1'b1;
            end
            sb_hit_r <= 1'b0;
            state    <= S_IDLE;
          end

          default: state <= S_IDLE;
        endcase
      end
    end
  end

  assign fire           = fire_r;
  assign fire_frame     = fire_frame_r;
  assign fire_rr        = fire_rr_r;
  assign sat            = sat_r;
  assign searchback_hit = sb_hit_r;
  // The chain needs ~85 cycles per beat.  Accepting a beat while busy drops
  // it silently, which is harmless at 500 SPS (200 000 cycles apart) and
  // catastrophic for any test that streams back to back.
  assign ready           = enable && (state == S_IDLE);
  assign dbg_met        = met_r;
  assign dbg_thresh     = thresh_r;

endmodule
