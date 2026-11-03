`timescale 1ns/1ps
module ecg_sync_fifo_prefetch #(parameter integer WIDTH=512, DEPTH=64, ADDR_W=$clog2(DEPTH))
(input logic clk,resetn,push,input logic [WIDTH-1:0] din,input logic pop,
output wire [WIDTH-1:0] dout,output wire full,empty,output logic [ADDR_W:0] count);
(* ram_style="block" *) logic [WIDTH-1:0] mem[0:DEPTH-1];
logic [ADDR_W-1:0] wr_ptr,rd_ptr;
wire take_push=push && !full, take_pop=pop && !empty;
wire [ADDR_W-1:0] rd_next=rd_ptr+take_pop;
logic [WIDTH-1:0] ram_data,forward_data;
logic forward_valid;
assign full=(count==DEPTH); assign empty=(count==0);
assign dout=forward_valid ? forward_data : ram_data;
always_ff @(posedge clk) begin
 if(resetn && take_push) mem[wr_ptr]<=din;
 ram_data<=mem[rd_next];
 forward_valid<=resetn && take_push && (wr_ptr==rd_next);
 forward_data<=din;
 if(!resetn) begin wr_ptr<='0;rd_ptr<='0;count<='0;end
 else begin
  if(take_push) wr_ptr<=wr_ptr+1'b1;
  rd_ptr<=rd_next;
  case({take_push,take_pop})
   2'b10:count<=count+1'b1;
   2'b01:count<=count-1'b1;
   default:count<=count;
  endcase
 end
end
endmodule
