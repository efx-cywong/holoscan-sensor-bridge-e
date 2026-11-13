// Fixed boot-time sequencing that replaces the firmware-controlled release of
// the legacy SerDes/video reset.  Camera register programming remains on the
// host through Hololink I2C.
module startup_config_fsm #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer SETTLE_MS = 1000
) (
    input  wire clk,
    input  wire rstn,
    output reg  release_reset
);

    localparam integer SETTLE_CYCLES = (CLK_HZ / 1000) * SETTLE_MS;
    localparam integer COUNT_WIDTH =
        (SETTLE_CYCLES <= 1) ? 1 : $clog2(SETTLE_CYCLES);

    reg [COUNT_WIDTH-1:0] count;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            count         <= {COUNT_WIDTH{1'b0}};
            release_reset <= 1'b0;
        end else if (!release_reset) begin
            if (count == SETTLE_CYCLES - 1) begin
                release_reset <= 1'b1;
            end else begin
                count <= count + 1'b1;
            end
        end
    end

endmodule
