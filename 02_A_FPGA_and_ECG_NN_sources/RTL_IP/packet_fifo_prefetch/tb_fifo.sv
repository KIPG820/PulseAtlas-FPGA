`timescale 1ns/1ps
module check_fifo #(parameter W=512,D=64)(output logic done=0);
logic clk=0;always #5 clk=~clk;
logic resetn=0,push=0,pop=0;
logic [W-1:0] din='0;
wire [W-1:0] od,nd;
wire ofu,oe,nf,ne;
wire [$clog2(D):0] oc,nc;
ecg_sync_fifo #(.WIDTH(W),.DEPTH(D)) ref_fifo(clk,resetn,push,din,pop,od,ofu,oe,oc);
ecg_sync_fifo_prefetch #(.WIDTH(W),.DEPTH(D)) dut(clk,resetn,push,din,pop,nd,nf,ne,nc);
integer i,j;integer full_hits=0,empty_hits=0,both_hits=0;
initial begin
 repeat(3) @(negedge clk);
 for(i=0;i<20000;i=i+1) begin
 resetn=(i%997!=0);
 if(i<150)begin push=1;pop=0;end
 else if(i<300)begin push=0;pop=1;end
 else if(i<450)begin push=1;pop=1;end
 else begin push=$urandom_range(0,1);pop=$urandom_range(0,1);end
 for(j=0;j<W;j=j+1)din[j]=$urandom_range(0,1);
 @(posedge clk);#1;
 if({ofu,oe,oc} !== {nf,ne,nc}) $fatal(1,"flags W=%0d cycle=%0d",W,i);
 if(!oe && od!==nd) $fatal(1,"data W=%0d cycle=%0d",W,i);
 if(ofu)full_hits++;if(oe)empty_hits++;if(push&&pop)both_hits++;
 @(negedge clk);
 end
 if(full_hits==0||empty_hits==0||both_hits==0)$fatal(1,"coverage");
 $display("PASS W=%0d D=%0d cycles=20000 full=%0d empty=%0d both=%0d",W,D,full_hits,empty_hits,both_hits);
 done=1;
end
endmodule
module tb_fifo;
wire a,b,c,d;
check_fifo #(.W(512),.D(64)) t0(a);
check_fifo #(.W(384),.D(64)) t1(b);
check_fifo #(.W(64),.D(64)) t2(c);
check_fifo #(.W(32),.D(4)) t3(d);
initial begin wait(a&&b&&c&&d);$display("PASS_ALL_FIFO_EQUIVALENCE");$finish;end
initial begin #1000000;$fatal(1,"timeout");end
endmodule
