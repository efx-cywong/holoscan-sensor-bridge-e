`timescale 1ns / 1ps

// APB3 slave for dual-servo (pan/tilt) control.
// Word-aligned register map (byte offsets):
//   0x00 CTRL       [0]=enable, [1]=pan_en, [2]=tilt_en
//   0x04 PAN_WIDTH  pulse high time in APB clock cycles
//   0x08 TILT_WIDTH pulse high time in APB clock cycles
//   0x0C PERIOD     full PWM period in APB clock cycles (~20 ms)
//
// Host address example for HSB slot 1:
//   0x2000_0000 + offset
module apb_servo_slave #(
    parameter ADDR_WIDTH = 10,
    parameter DATA_WIDTH = 32,
    // Defaults assume 100 MHz APB clk:
    //   pan center = 1.5 ms = 150_000 cycles
    //   tilt default = 0.6 ms = 60_000 cycles
    //   20  ms period = 2_000_000 cycles
    parameter [31:0] DEFAULT_PAN_PULSE  = 32'd150_000,
    parameter [31:0] DEFAULT_TILT_PULSE = 32'd60_000,
    // Hard mechanical tilt limits: 0.5 ms to 1.5 ms @ 100 MHz.
    parameter [31:0] TILT_MIN_PULSE = 32'd50_000,
    parameter [31:0] TILT_MAX_PULSE = 32'd150_000,
    parameter [31:0] DEFAULT_PERIOD = 32'd2_000_000
) (
    input                   clk,
    input                   resetn,
    input  [ADDR_WIDTH-1:0] PADDR,
    input                   PSEL,
    input                   PENABLE,
    output                  PREADY,
    input                   PWRITE,
    input  [DATA_WIDTH-1:0] PWDATA,
    output [DATA_WIDTH-1:0] PRDATA,
    output                  PSLVERROR,

    // Soft outputs for PWM generators (wired later)
    output                  o_enable,
    output                  o_pan_en,
    output                  o_tilt_en,
    output [31:0]           o_pan_width,
    output [31:0]           o_tilt_width,
    output [31:0]           o_period
);

    localparam [1:0] IDLE   = 2'b00;
    localparam [1:0] SETUP  = 2'b01;
    localparam [1:0] ACCESS = 2'b10;

    // Word index from byte address (ignore [1:0]).
    wire [3:0] reg_idx = PADDR[5:2];

    reg [1:0]            busState, busNext;
    reg [DATA_WIDTH-1:0] slaveRegOut;
    reg                  slaveReady;

    reg [31:0] ctrl_reg;
    reg [31:0] pan_width_reg;
    reg [31:0] tilt_width_reg;
    reg [31:0] period_reg;

    wire actWrite = PWRITE  & (busState == ACCESS);
    wire actRead  = !PWRITE & (busState == ACCESS);

    assign PSLVERROR = 1'b0;
    assign PRDATA    = slaveRegOut;
    assign PREADY    = slaveReady & (busState != IDLE);

    assign o_enable     = ctrl_reg[0];
    assign o_pan_en     = ctrl_reg[1];
    assign o_tilt_en    = ctrl_reg[2];
    assign o_pan_width  = pan_width_reg;
    assign o_tilt_width = tilt_width_reg;
    assign o_period     = period_reg;

    //--------------------------------------------------------------------
    // APB state machine (same style as apb3_slave.sv)
    //--------------------------------------------------------------------
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            busState <= IDLE;
        else
            busState <= busNext;
    end

    always @(*) begin
        busNext = busState;
        case (busState)
            IDLE: begin
                if (PSEL && !PENABLE)
                    busNext = SETUP;
                else
                    busNext = IDLE;
            end
            SETUP: begin
                if (PSEL && PENABLE)
                    busNext = ACCESS;
                else
                    busNext = IDLE;
            end
            ACCESS: begin
                if (PREADY)
                    busNext = IDLE;
                else
                    busNext = ACCESS;
            end
            default: busNext = IDLE;
        endcase
    end

    always @(posedge clk) begin
        slaveReady <= actWrite | actRead;
    end

    //--------------------------------------------------------------------
    // Write decode
    //--------------------------------------------------------------------
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            ctrl_reg       <= 32'h0;
            pan_width_reg  <= DEFAULT_PAN_PULSE;
            tilt_width_reg <= DEFAULT_TILT_PULSE;
            period_reg     <= DEFAULT_PERIOD;
        end else if (actWrite) begin
            case (reg_idx)
                4'd0: ctrl_reg       <= PWDATA;
                4'd1: pan_width_reg  <= PWDATA;
                // Enforce tilt mechanical limits in PL.  Host-side clipping
                // is secondary and must not be the only protection.
                4'd2: tilt_width_reg <=
                    (PWDATA < TILT_MIN_PULSE) ? TILT_MIN_PULSE :
                    (PWDATA > TILT_MAX_PULSE) ? TILT_MAX_PULSE :
                    PWDATA;
                4'd3: period_reg     <= PWDATA;
                default: ; // ignore unmapped writes
            endcase
        end
    end

    //--------------------------------------------------------------------
    // Read decode
    //--------------------------------------------------------------------
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            slaveRegOut <= 32'h0;
        end else if (actRead) begin
            case (reg_idx)
                4'd0: slaveRegOut <= ctrl_reg;
                4'd1: slaveRegOut <= pan_width_reg;
                4'd2: slaveRegOut <= tilt_width_reg;
                4'd3: slaveRegOut <= period_reg;
                default: slaveRegOut <= 32'hDEAD_BEEF;
            endcase
        end
    end

endmodule
