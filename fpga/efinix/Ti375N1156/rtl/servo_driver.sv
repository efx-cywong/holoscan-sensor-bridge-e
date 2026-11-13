module servo_driver #(
    parameter [31:0] DEFAULT_PULSE  = 32'd150_000,
    parameter [31:0] DEFAULT_PERIOD = 32'd2_000_000
) (
    //APB clock domain: 100MHz
    input           clk,
    input           rstn,

    //from  apb_servo_slave
    input           i_servo_enable,
    input           i_servo_pan_en,
    input           i_servo_tilt_en,
    input [31:0]    i_servo_pan_width,
    input   [31:0]  i_servo_tilt_width,
    input   [31:0]  i_servo_period,

    //Dual PWM
    output          o_pan_pwm,
    output          o_tilt_pwm
);

    // Single timebase for both channels.
    reg [31:0] counter = 32'd0;

    // Use programmed period if non-zero; otherwise fall back to DEFAULT_PERIOD.
    wire [31:0] period_eff =
        (i_servo_period != 32'd0) ? i_servo_period : DEFAULT_PERIOD;
    wire [31:0] period_minus1 = period_eff - 32'd1;

    // Clamp pulse widths so we never exceed the PWM window.
    wire [31:0] pan_pw_eff =
        (i_servo_pan_width <= period_eff) ? i_servo_pan_width : period_eff;
    wire [31:0] tilt_pw_eff =
        (i_servo_tilt_width <= period_eff) ? i_servo_tilt_width : period_eff;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            counter <= 32'd0;
        end else if (counter == period_minus1) begin
            counter <= 32'd0;
        end else begin
            counter <= counter + 32'd1;
        end
    end

    // Standard RC PWM from host/APB:
    // - counter < pulse_width => output is high
    // - enable bits gate the respective channels
    assign o_pan_pwm  = i_servo_enable && i_servo_pan_en  && (counter < pan_pw_eff);
    assign o_tilt_pwm = i_servo_enable && i_servo_tilt_en && (counter < tilt_pw_eff);

endmodule
