"""Create an isolated, cycle-equivalent RAM-history QRS variant from R9."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "bld/ecg_nn_r9/ip_repo_ecg/ecg_qrs_ip/src/ecg_qrs_detector.sv"


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def main():
    src = SOURCE.read_text(encoding="utf-8")
    (HERE / "ecg_qrs_detector_reference.sv").write_text(src, encoding="utf-8")
    text = replace_once(src, "module ecg_qrs_detector (", "module ecg_qrs_detector_ram (")
    text = replace_once(text, "  logic [47:0] mwi_ring [0:MWI_LEN-1];",
        '  (* ram_style = "distributed" *) logic [47:0] mwi_ring [0:MWI_LEN-1];\n'
        '  logic [7:0] mwi_valid_count;')
    text = replace_once(text, "  logic [31:0] fid_ring [0:FID_LEN-1];",
        '  (* ram_style = "distributed" *) logic [31:0] fid_ring [0:FID_LEN-1];\n'
        '  logic [7:0] fid_valid_count;\n'
        '  logic [31:0] fid_read;')
    text = replace_once(text, "  integer i;\n  always_ff @(posedge aclk) begin",
        """  // Histories are written in order from address zero after either reset.
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
  always_ff @(posedge aclk) begin""")
    text = replace_once(text, "      for (i = 0; i < MWI_LEN; i++) mwi_ring[i] <= '0;",
        "      mwi_valid_count <= '0;")
    text = replace_once(text, "      for (i = 0; i < FID_LEN; i++) fid_ring[i] <= '0;",
        "      fid_valid_count <= '0;")
    text = replace_once(text, "            oldest_r <= mwi_ring[mwi_p];",
        "            oldest_r <= (mwi_valid_count == MWI_LEN) ? mwi_ring[mwi_p] : 48'd0;")
    text = replace_once(text, "            mwi_ring[mwi_p] <= sq_r;",
        "            if (mwi_valid_count < MWI_LEN) mwi_valid_count <= mwi_valid_count + 1'b1;")
    text = replace_once(text, "            fid_ring[fid_p] <= bp_cur[31] ? 32'(0 - bp_cur) : 32'(bp_cur);",
        "            if (fid_valid_count < FID_LEN) fid_valid_count <= fid_valid_count + 1'b1;")
    text = text.replace("if (fid_ring[fid_wr] > fid_best_mag)", "if (fid_read > fid_best_mag)")
    text = text.replace("fid_best_mag <= fid_ring[fid_wr];", "fid_best_mag <= fid_read;")
    (HERE / "ecg_qrs_detector_ram.sv").write_text(text, encoding="utf-8")
    (HERE / "source_sha256.txt").write_text(hashlib.sha256(SOURCE.read_bytes()).hexdigest() + "\n", encoding="ascii")
    print("Created isolated QRS variant; source R9 is unchanged.")


if __name__ == "__main__":
    main()
