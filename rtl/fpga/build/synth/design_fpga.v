module fp32_add (
	a,
	b,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	input wire [31:0] b;
	output reg [31:0] y;
	localparam [31:0] QNAN = 32'h7fc00000;
	function automatic [31:0] msb64;
		input reg [63:0] v;
		reg signed [31:0] i;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			begin : sv2v_autoblock_1
				reg signed [31:0] _sv2v_value_on_break;
				for (i = 63; i >= 0; i = i - 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						if (v[i]) begin
							msb64 = i;
							_sv2v_jump = 2'b11;
						end
						_sv2v_value_on_break = i;
					end
				if (!(_sv2v_jump < 2'b10))
					i = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
			if (_sv2v_jump == 2'b00) begin
				msb64 = 0;
				_sv2v_jump = 2'b11;
			end
		end
	endfunction
	wire sa;
	wire sb;
	wire [7:0] ea;
	wire [7:0] eb;
	wire [22:0] ma;
	wire [22:0] mb;
	assign sa = a[31];
	assign ea = a[30:23];
	assign ma = a[22:0];
	assign sb = b[31];
	assign eb = b[30:23];
	assign mb = b[22:0];
	wire a_zero;
	wire b_zero;
	wire a_inf;
	wire b_inf;
	wire a_nan;
	wire b_nan;
	assign a_zero = (ea == 8'd0) && (ma == 23'd0);
	assign b_zero = (eb == 8'd0) && (mb == 23'd0);
	assign a_inf = (ea == 8'd255) && (ma == 23'd0);
	assign b_inf = (eb == 8'd255) && (mb == 23'd0);
	assign a_nan = (ea == 8'd255) && (ma != 23'd0);
	assign b_nan = (eb == 8'd255) && (mb != 23'd0);
	wire [23:0] sig_a;
	wire [23:0] sig_b;
	wire signed [9:0] exp_a;
	wire signed [9:0] exp_b;
	assign sig_a = (ea == 8'd0 ? {1'b0, ma} : {1'b1, ma});
	assign sig_b = (eb == 8'd0 ? {1'b0, mb} : {1'b1, mb});
	assign exp_a = (ea == 8'd0 ? -10'sd126 : $signed({2'b00, ea}) - 10'sd127);
	assign exp_b = (eb == 8'd0 ? -10'sd126 : $signed({2'b00, eb}) - 10'sd127);
	wire a_ge;
	assign a_ge = (exp_a > exp_b) || ((exp_a == exp_b) && (sig_a >= sig_b));
	reg s1;
	reg s2;
	reg signed [9:0] e1;
	reg signed [9:0] e2;
	reg [23:0] sg1;
	reg [23:0] sg2;
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_ge) begin
			s1 = sa;
			e1 = exp_a;
			sg1 = sig_a;
			s2 = sb;
			e2 = exp_b;
			sg2 = sig_b;
		end
		else begin
			s1 = sb;
			e1 = exp_b;
			sg1 = sig_b;
			s2 = sa;
			e2 = exp_a;
			sg2 = sig_a;
		end
	end
	wire eff_sub;
	assign eff_sub = s1 ^ s2;
	wire signed [10:0] shamt;
	assign shamt = e1 - e2;
	reg [63:0] ext1;
	reg [63:0] ext2_al;
	reg [63:0] raw;
	always @(*) begin
		if (_sv2v_0)
			;
		ext1 = {40'd0, sg1} << 27;
		if (shamt > 11'sd27)
			ext2_al = 64'd0;
		else
			ext2_al = ({40'd0, sg2} << 27) >> shamt[5:0];
		raw = (eff_sub ? ext1 - ext2_al : ext1 + ext2_al);
	end
	reg res_zero;
	reg [31:0] p;
	reg signed [11:0] exp0;
	reg signed [11:0] rpos;
	reg [64:0] sig;
	reg g;
	reg st;
	reg ru;
	reg signed [11:0] fexp;
	always @(*) begin
		if (_sv2v_0)
			;
		res_zero = raw == 64'd0;
		p = msb64(raw);
		exp0 = ($signed({1'b0, p[10:0]}) + {{2 {e1[9]}}, e1}) - 12'sd50;
		if (exp0 >= -12'sd126)
			rpos = $signed({1'b0, p[10:0]}) - 12'sd23;
		else
			rpos = -(12'sd99 + {{2 {e1[9]}}, e1});
		if (rpos <= 12'sd0) begin
			sig = {1'b0, raw} << -rpos[5:0];
			g = 1'b0;
			st = 1'b0;
		end
		else begin
			sig = {1'b0, raw >> rpos[5:0]};
			g = raw[rpos[5:0] - 6'd1];
			st = (rpos > 12'sd1 ? |(raw & ((64'd1 << (rpos[5:0] - 6'd1)) - 64'd1)) : 1'b0);
		end
		ru = g & (st | sig[0]);
		sig = sig + {64'd0, ru};
		if (exp0 >= -12'sd126)
			fexp = (sig[24] ? exp0 + 12'sd1 : exp0) + 12'sd127;
		else
			fexp = (sig[23] ? 12'sd1 : 12'sd0);
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_nan || b_nan)
			y = QNAN;
		else if (a_inf && b_inf)
			y = (sa == sb ? {sa, 31'h7f800000} : QNAN);
		else if (a_inf)
			y = {sa, 31'h7f800000};
		else if (b_inf)
			y = {sb, 31'h7f800000};
		else if (a_zero && b_zero)
			y = {sa & sb, 31'h00000000};
		else if (res_zero)
			y = 32'd0;
		else if (fexp >= 12'sd255)
			y = {s1, 31'h7f800000};
		else if (exp0 >= -12'sd126)
			y = (sig[24] ? {s1, fexp[7:0], 23'd0} : {s1, fexp[7:0], sig[22:0]});
		else
			y = (sig[23] ? {s1, 31'h00800000} : {s1, 8'd0, sig[22:0]});
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_mul (
	a,
	b,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	input wire [31:0] b;
	output reg [31:0] y;
	localparam [31:0] QNAN = 32'h7fc00000;
	function automatic [31:0] msb48;
		input reg [47:0] v;
		reg signed [31:0] i;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			begin : sv2v_autoblock_1
				reg signed [31:0] _sv2v_value_on_break;
				for (i = 47; i >= 0; i = i - 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						if (v[i]) begin
							msb48 = i;
							_sv2v_jump = 2'b11;
						end
						_sv2v_value_on_break = i;
					end
				if (!(_sv2v_jump < 2'b10))
					i = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
			if (_sv2v_jump == 2'b00) begin
				msb48 = 0;
				_sv2v_jump = 2'b11;
			end
		end
	endfunction
	wire sa;
	wire sb;
	wire [7:0] ea;
	wire [7:0] eb;
	wire [22:0] ma;
	wire [22:0] mb;
	assign sa = a[31];
	assign ea = a[30:23];
	assign ma = a[22:0];
	assign sb = b[31];
	assign eb = b[30:23];
	assign mb = b[22:0];
	wire a_zero;
	wire b_zero;
	wire a_inf;
	wire b_inf;
	wire a_nan;
	wire b_nan;
	assign a_zero = (ea == 8'd0) && (ma == 23'd0);
	assign b_zero = (eb == 8'd0) && (mb == 23'd0);
	assign a_inf = (ea == 8'd255) && (ma == 23'd0);
	assign b_inf = (eb == 8'd255) && (mb == 23'd0);
	assign a_nan = (ea == 8'd255) && (ma != 23'd0);
	assign b_nan = (eb == 8'd255) && (mb != 23'd0);
	wire [23:0] sig_a;
	wire [23:0] sig_b;
	wire signed [10:0] exp_a;
	wire signed [10:0] exp_b;
	wire signed [10:0] exp_sum;
	assign sig_a = (ea == 8'd0 ? {1'b0, ma} : {1'b1, ma});
	assign sig_b = (eb == 8'd0 ? {1'b0, mb} : {1'b1, mb});
	assign exp_a = (ea == 8'd0 ? -11'sd126 : $signed({3'b000, ea}) - 11'sd127);
	assign exp_b = (eb == 8'd0 ? -11'sd126 : $signed({3'b000, eb}) - 11'sd127);
	assign exp_sum = exp_a + exp_b;
	wire [47:0] raw;
	assign raw = sig_a * sig_b;
	wire s_y;
	assign s_y = sa ^ sb;
	reg res_zero;
	reg [31:0] p;
	reg signed [11:0] exp0;
	reg signed [11:0] rpos;
	reg [49:0] sig;
	reg g;
	reg st;
	reg ru;
	reg signed [11:0] fexp;
	always @(*) begin
		if (_sv2v_0)
			;
		res_zero = raw == 48'd0;
		p = msb48(raw);
		exp0 = ($signed({1'b0, p[10:0]}) + {exp_sum[10], exp_sum}) - 12'sd46;
		if (exp0 >= -12'sd126)
			rpos = $signed({1'b0, p[10:0]}) - 12'sd23;
		else
			rpos = -(12'sd103 + {exp_sum[10], exp_sum});
		if (rpos <= 12'sd0) begin
			sig = {2'b00, raw};
			g = 1'b0;
			st = 1'b0;
		end
		else if (rpos >= 12'sd49) begin
			sig = 50'd0;
			g = 1'b0;
			st = raw != 48'd0;
		end
		else if (rpos == 12'sd48) begin
			sig = 50'd0;
			g = raw[47];
			st = |raw[46:0];
		end
		else begin
			sig = {2'b00, raw >> rpos[5:0]};
			g = raw[rpos[5:0] - 6'd1];
			st = (rpos > 12'sd1 ? |(raw & ((48'd1 << (rpos[5:0] - 6'd1)) - 48'd1)) : 1'b0);
		end
		ru = g & (st | sig[0]);
		sig = sig + {49'd0, ru};
		if (exp0 >= -12'sd126)
			fexp = (sig[24] ? exp0 + 12'sd1 : exp0) + 12'sd127;
		else
			fexp = (sig[23] ? 12'sd1 : 12'sd0);
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_nan || b_nan)
			y = QNAN;
		else if ((a_inf && b_zero) || (a_zero && b_inf))
			y = QNAN;
		else if (a_inf || b_inf)
			y = {s_y, 31'h7f800000};
		else if (a_zero || b_zero)
			y = {s_y, 31'h00000000};
		else if (res_zero)
			y = {s_y, 31'h00000000};
		else if (fexp >= 12'sd255)
			y = {s_y, 31'h7f800000};
		else if (exp0 >= -12'sd126)
			y = (sig[24] ? {s_y, fexp[7:0], 23'd0} : {s_y, fexp[7:0], sig[22:0]});
		else
			y = (sig[23] ? {s_y, 31'h00800000} : {s_y, 8'd0, sig[22:0]});
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_div (
	a,
	b,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	input wire [31:0] b;
	output reg [31:0] y;
	localparam [31:0] QNAN = 32'h7fc00000;
	function automatic [31:0] msb23;
		input reg [22:0] v;
		reg signed [31:0] i;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			begin : sv2v_autoblock_1
				reg signed [31:0] _sv2v_value_on_break;
				for (i = 22; i >= 0; i = i - 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						if (v[i]) begin
							msb23 = i;
							_sv2v_jump = 2'b11;
						end
						_sv2v_value_on_break = i;
					end
				if (!(_sv2v_jump < 2'b10))
					i = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
			if (_sv2v_jump == 2'b00) begin
				msb23 = 0;
				_sv2v_jump = 2'b11;
			end
		end
	endfunction
	wire sa;
	wire sb;
	wire [7:0] ea;
	wire [7:0] eb;
	wire [22:0] ma;
	wire [22:0] mb;
	assign sa = a[31];
	assign ea = a[30:23];
	assign ma = a[22:0];
	assign sb = b[31];
	assign eb = b[30:23];
	assign mb = b[22:0];
	wire a_zero;
	wire b_zero;
	wire a_inf;
	wire b_inf;
	wire a_nan;
	wire b_nan;
	wire a_sub;
	wire b_sub;
	assign a_zero = (ea == 8'd0) && (ma == 23'd0);
	assign b_zero = (eb == 8'd0) && (mb == 23'd0);
	assign a_sub = (ea == 8'd0) && (ma != 23'd0);
	assign b_sub = (eb == 8'd0) && (mb != 23'd0);
	assign a_inf = (ea == 8'd255) && (ma == 23'd0);
	assign b_inf = (eb == 8'd255) && (mb == 23'd0);
	assign a_nan = (ea == 8'd255) && (ma != 23'd0);
	assign b_nan = (eb == 8'd255) && (mb != 23'd0);
	reg [31:0] mp_a;
	reg [31:0] mp_b;
	reg [23:0] sig_a;
	reg [23:0] sig_b;
	reg signed [9:0] exp_a;
	reg signed [9:0] exp_b;
	always @(*) begin
		if (_sv2v_0)
			;
		mp_a = (a_sub ? msb23(ma) : 0);
		mp_b = (b_sub ? msb23(mb) : 0);
		if (a_sub) begin
			sig_a = {1'b0, ma} << (5'd23 - mp_a[4:0]);
			exp_a = -(10'sd126 + {{5 {1'b0}}, 5'd23 - mp_a[4:0]});
		end
		else begin
			sig_a = {1'b1, ma};
			exp_a = $signed({2'b00, ea}) - 10'sd127;
		end
		if (b_sub) begin
			sig_b = {1'b0, mb} << (5'd23 - mp_b[4:0]);
			exp_b = -(10'sd126 + {{5 {1'b0}}, 5'd23 - mp_b[4:0]});
		end
		else begin
			sig_b = {1'b1, mb};
			exp_b = $signed({2'b00, eb}) - 10'sd127;
		end
	end
	wire [51:0] dividend;
	wire [28:0] q_full;
	wire [23:0] remainder;
	assign dividend = {sig_a, 28'd0};
	function automatic [28:0] sv2v_cast_29;
		input reg [28:0] inp;
		sv2v_cast_29 = inp;
	endfunction
	assign q_full = sv2v_cast_29(dividend / {28'd0, sig_b});
	function automatic [23:0] sv2v_cast_24;
		input reg [23:0] inp;
		sv2v_cast_24 = inp;
	endfunction
	assign remainder = sv2v_cast_24(dividend - ({23'd0, q_full} * {28'd0, sig_b}));
	wire a_ge_b;
	assign a_ge_b = sig_a >= sig_b;
	reg [23:0] mant24_pre;
	reg rb;
	reg st;
	reg ru;
	reg [24:0] mant24_rnd;
	reg signed [10:0] exp_y_unb;
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_ge_b) begin
			mant24_pre = q_full[28:5];
			rb = q_full[4];
			st = |q_full[3:0] | (|remainder);
			exp_y_unb = exp_a - exp_b;
		end
		else begin
			mant24_pre = q_full[27:4];
			rb = q_full[3];
			st = |q_full[2:0] | (|remainder);
			exp_y_unb = (exp_a - exp_b) - 11'sd1;
		end
		ru = rb & (st | mant24_pre[0]);
		mant24_rnd = {1'b0, mant24_pre} + {24'd0, ru};
	end
	reg carry;
	reg signed [10:0] exp_y_final_unb;
	reg [7:0] exp_y_biased;
	function automatic signed [7:0] sv2v_cast_8_signed;
		input reg signed [7:0] inp;
		sv2v_cast_8_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		carry = mant24_rnd[24];
		exp_y_final_unb = exp_y_unb + (carry ? 11'sd1 : 11'sd0);
		exp_y_biased = sv2v_cast_8_signed(exp_y_final_unb + 11'sd127);
	end
	reg signed [11:0] k_lsb_s;
	reg [4:0] k_lsb;
	reg [28:0] q_shifted_sub;
	reg [22:0] mant_sub_pre;
	reg sub_rb;
	reg sub_st;
	reg sub_ru;
	reg [23:0] mant_sub_rnd;
	always @(*) begin
		if (_sv2v_0)
			;
		k_lsb_s = ({{2 {exp_b[9]}}, exp_b} - {{2 {exp_a[9]}}, exp_a}) - 12'sd121;
		k_lsb = (k_lsb_s > 12'sd29 ? 5'd29 : k_lsb_s[4:0]);
		q_shifted_sub = q_full >> k_lsb;
		mant_sub_pre = q_shifted_sub[22:0];
		if ((k_lsb_s >= 12'sd1) && (k_lsb_s <= 12'sd29))
			sub_rb = q_full[k_lsb - 5'd1];
		else
			sub_rb = 1'b0;
		if (k_lsb_s > 12'sd29)
			sub_st = |q_full | (|remainder);
		else if (k_lsb_s >= 12'sd2)
			sub_st = |(q_full & ((29'd1 << (k_lsb - 5'd1)) - 29'd1)) | (|remainder);
		else if (k_lsb_s == 12'sd1)
			sub_st = |remainder;
		else
			sub_st = 1'b0;
		sub_ru = sub_rb & (sub_st | mant_sub_pre[0]);
		mant_sub_rnd = {1'b0, mant_sub_pre} + {23'd0, sub_ru};
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_nan || b_nan)
			y = QNAN;
		else if (a_inf && b_inf)
			y = QNAN;
		else if (a_zero && b_zero)
			y = QNAN;
		else if (a_inf)
			y = {sa ^ sb, 31'h7f800000};
		else if (b_inf)
			y = {sa ^ sb, 31'h00000000};
		else if (b_zero)
			y = {sa ^ sb, 31'h7f800000};
		else if (a_zero)
			y = {sa ^ sb, 31'h00000000};
		else if (exp_y_final_unb >= 11'sd128)
			y = {sa ^ sb, 31'h7f800000};
		else if (exp_y_final_unb < -11'sd126) begin
			if (mant_sub_rnd[23])
				y = {sa ^ sb, 31'h00800000};
			else
				y = {sa ^ sb, 8'd0, mant_sub_rnd[22:0]};
		end
		else if (carry)
			y = {sa ^ sb, exp_y_biased, 23'd0};
		else
			y = {sa ^ sb, exp_y_biased, mant24_rnd[22:0]};
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_sqrt (
	a,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	output reg [31:0] y;
	localparam [31:0] QNAN = 32'h7fc00000;
	function automatic [31:0] msb23;
		input reg [22:0] v;
		reg signed [31:0] i;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			begin : sv2v_autoblock_1
				reg signed [31:0] _sv2v_value_on_break;
				for (i = 22; i >= 0; i = i - 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						if (v[i]) begin
							msb23 = i;
							_sv2v_jump = 2'b11;
						end
						_sv2v_value_on_break = i;
					end
				if (!(_sv2v_jump < 2'b10))
					i = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
			if (_sv2v_jump == 2'b00) begin
				msb23 = 0;
				_sv2v_jump = 2'b11;
			end
		end
	endfunction
	wire s;
	wire [7:0] e;
	wire [22:0] m;
	assign s = a[31];
	assign e = a[30:23];
	assign m = a[22:0];
	wire a_zero;
	wire a_inf;
	wire a_nan;
	wire a_sub;
	wire a_neg_finite;
	assign a_zero = (e == 8'd0) && (m == 23'd0);
	assign a_sub = (e == 8'd0) && (m != 23'd0);
	assign a_inf = (e == 8'd255) && (m == 23'd0);
	assign a_nan = (e == 8'd255) && (m != 23'd0);
	assign a_neg_finite = ((s && !a_zero) && !a_inf) && !a_nan;
	reg [31:0] mp;
	reg [23:0] sig_a;
	reg signed [9:0] exp_a;
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_sub) begin
			mp = msb23(m);
			sig_a = {1'b0, m} << (5'd23 - mp[4:0]);
			exp_a = -(10'sd126 + {{5 {1'b0}}, 5'd23 - mp[4:0]});
		end
		else begin
			mp = 0;
			sig_a = {1'b1, m};
			exp_a = $signed({2'b00, e}) - 10'sd127;
		end
	end
	wire exp_a_odd;
	assign exp_a_odd = exp_a[0];
	reg [49:0] M_pad;
	reg signed [9:0] R_exp;
	always @(*) begin
		if (_sv2v_0)
			;
		if (exp_a_odd) begin
			M_pad = {sig_a, 26'd0};
			R_exp = (exp_a - 10'sd1) >>> 1;
		end
		else begin
			M_pad = {1'b0, sig_a, 25'd0};
			R_exp = exp_a >>> 1;
		end
	end
	reg [51:0] r [0:25];
	reg [24:0] q [0:25];
	wire [51:0] trial;
	integer ii;
	always @(*) begin
		if (_sv2v_0)
			;
		r[25] = 52'd0;
		q[25] = 25'd0;
		for (ii = 24; ii >= 0; ii = ii - 1)
			begin : sv2v_autoblock_2
				reg [51:0] r_next0;
				reg [51:0] trial0;
				reg [1:0] block;
				block = M_pad[2 * ii+:2];
				r_next0 = (r[ii + 1] << 2) | {50'd0, block};
				trial0 = ({27'd0, q[ii + 1]} << 2) | 52'd1;
				if (r_next0 >= trial0) begin
					r[ii] = r_next0 - trial0;
					q[ii] = (q[ii + 1] << 1) | 25'd1;
				end
				else begin
					r[ii] = r_next0;
					q[ii] = q[ii + 1] << 1;
				end
			end
	end
	wire [24:0] q_final;
	wire [51:0] r_final;
	wire sticky;
	assign q_final = q[0];
	assign r_final = r[0];
	assign sticky = r_final != 52'd0;
	wire rb_sq;
	wire ru_sq;
	wire [23:0] mant_pre;
	wire [23:0] mant_rnd;
	assign rb_sq = q_final[0];
	assign mant_pre = q_final[24:1];
	assign ru_sq = rb_sq & (sticky | mant_pre[0]);
	wire [24:0] mant24_rnd;
	assign mant24_rnd = {1'b0, mant_pre} + {24'd0, ru_sq};
	reg carry;
	reg signed [9:0] exp_y_unb;
	reg [7:0] exp_y_biased;
	function automatic signed [7:0] sv2v_cast_8_signed;
		input reg signed [7:0] inp;
		sv2v_cast_8_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		carry = mant24_rnd[24];
		exp_y_unb = R_exp + (carry ? 10'sd1 : 10'sd0);
		exp_y_biased = sv2v_cast_8_signed(exp_y_unb + 10'sd127);
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_nan)
			y = QNAN;
		else if (a_neg_finite)
			y = QNAN;
		else if (a_zero)
			y = {s, 31'h00000000};
		else if (a_inf)
			y = (s ? QNAN : 32'h7f800000);
		else if (exp_y_unb >= 10'sd128)
			y = 32'h7f800000;
		else if (exp_y_unb < -10'sd126)
			y = 32'h00000000;
		else if (carry)
			y = {1'b0, exp_y_biased, 23'd0};
		else
			y = {1'b0, exp_y_biased, mant24_rnd[22:0]};
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_to_fp16 (
	a,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	output reg [15:0] y;
	localparam [15:0] QNAN_H = 16'h7e00;
	wire s;
	wire [7:0] e;
	wire [22:0] m;
	assign s = a[31];
	assign e = a[30:23];
	assign m = a[22:0];
	wire a_zero;
	wire a_inf;
	wire a_nan;
	wire a_sub;
	assign a_zero = (e == 8'd0) && (m == 23'd0);
	assign a_sub = (e == 8'd0) && (m != 23'd0);
	assign a_inf = (e == 8'd255) && (m == 23'd0);
	assign a_nan = (e == 8'd255) && (m != 23'd0);
	wire signed [9:0] E;
	assign E = $signed({2'b00, e}) - 10'sd127;
	reg [4:0] exp_h_n;
	reg [9:0] mant_h_n;
	reg rb_n;
	reg st_n;
	reg ru_n;
	reg [10:0] mant_n_rnd;
	reg [4:0] exp_n_final;
	reg [9:0] mant_n_final;
	reg ovfl_n;
	function automatic signed [4:0] sv2v_cast_5_signed;
		input reg signed [4:0] inp;
		sv2v_cast_5_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		exp_h_n = sv2v_cast_5_signed(E + 10'sd15);
		mant_h_n = m[22:13];
		rb_n = m[12];
		st_n = |m[11:0];
		ru_n = rb_n & (st_n | mant_h_n[0]);
		mant_n_rnd = {1'b0, mant_h_n} + {10'd0, ru_n};
		if (mant_n_rnd[10]) begin
			exp_n_final = exp_h_n + 5'd1;
			mant_n_final = 10'd0;
			ovfl_n = exp_h_n == 5'd30;
		end
		else begin
			exp_n_final = exp_h_n;
			mant_n_final = mant_n_rnd[9:0];
			ovfl_n = 1'b0;
		end
	end
	reg [4:0] shamt;
	reg [23:0] full_sig;
	reg [23:0] sub_shifted;
	reg [9:0] mant_s_pre;
	reg rb_s;
	reg st_s;
	reg ru_s;
	reg [10:0] mant_s_rnd;
	reg [4:0] exp_s_final;
	reg [9:0] mant_s_final;
	always @(*) begin
		if (_sv2v_0)
			;
		full_sig = {1'b1, m};
		shamt = sv2v_cast_5_signed(-(E + 10'sd1));
		sub_shifted = full_sig >> shamt;
		mant_s_pre = sub_shifted[9:0];
		rb_s = full_sig[shamt - 5'd1];
		if (shamt >= 5'd2)
			st_s = |(full_sig & ((24'd1 << (shamt - 5'd1)) - 24'd1));
		else
			st_s = 1'b0;
		ru_s = rb_s & (st_s | mant_s_pre[0]);
		mant_s_rnd = {1'b0, mant_s_pre} + {10'd0, ru_s};
		if (mant_s_rnd[10]) begin
			exp_s_final = 5'd1;
			mant_s_final = 10'd0;
		end
		else begin
			exp_s_final = 5'd0;
			mant_s_final = mant_s_rnd[9:0];
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_nan)
			y = QNAN_H;
		else if (a_inf)
			y = {s, 15'h7c00};
		else if (a_zero || a_sub)
			y = {s, 15'h0000};
		else if (E > 10'sd15)
			y = {s, 15'h7c00};
		else if (E < -10'sd25)
			y = {s, 15'h0000};
		else if (E >= -10'sd14)
			y = (ovfl_n ? {s, 15'h7c00} : {s, exp_n_final, mant_n_final});
		else
			y = {s, exp_s_final, mant_s_final};
	end
	initial _sv2v_0 = 0;
endmodule
module fp16_to_fp32 (
	a,
	y
);
	reg _sv2v_0;
	input wire [15:0] a;
	output reg [31:0] y;
	localparam [31:0] QNAN = 32'h7fc00000;
	function automatic [31:0] msb10;
		input reg [9:0] v;
		reg signed [31:0] i;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			begin : sv2v_autoblock_1
				reg signed [31:0] _sv2v_value_on_break;
				for (i = 9; i >= 0; i = i - 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						if (v[i]) begin
							msb10 = i;
							_sv2v_jump = 2'b11;
						end
						_sv2v_value_on_break = i;
					end
				if (!(_sv2v_jump < 2'b10))
					i = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
			if (_sv2v_jump == 2'b00) begin
				msb10 = 0;
				_sv2v_jump = 2'b11;
			end
		end
	endfunction
	wire s;
	wire [4:0] e;
	wire [9:0] m;
	assign s = a[15];
	assign e = a[14:10];
	assign m = a[9:0];
	wire is_zero;
	wire is_inf;
	wire is_nan;
	wire is_sub;
	assign is_zero = (e == 5'd0) && (m == 10'd0);
	assign is_sub = (e == 5'd0) && (m != 10'd0);
	assign is_inf = (e == 5'd31) && (m == 10'd0);
	assign is_nan = (e == 5'd31) && (m != 10'd0);
	reg [31:0] msb_p;
	reg [4:0] shamt_u;
	reg [23:0] m_shifted;
	reg [22:0] sub_mant23;
	reg [7:0] sub_exp32;
	always @(*) begin
		if (_sv2v_0)
			;
		msb_p = msb10(m);
		shamt_u = 5'd23 - msb_p[4:0];
		m_shifted = {14'd0, m} << shamt_u;
		sub_mant23 = m_shifted[22:0];
		sub_exp32 = 8'd103 + {3'd0, msb_p[4:0]};
	end
	function automatic [7:0] sv2v_cast_8;
		input reg [7:0] inp;
		sv2v_cast_8 = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		if (is_zero)
			y = {s, 31'h00000000};
		else if (is_inf)
			y = {s, 31'h7f800000};
		else if (is_nan)
			y = QNAN;
		else if (is_sub)
			y = {s, sub_exp32, sub_mant23};
		else
			y = {s, sv2v_cast_8(e) + 8'd112, m, 13'd0};
	end
	initial _sv2v_0 = 0;
endmodule
module i32_to_fp32 (
	a,
	y
);
	reg _sv2v_0;
	input wire signed [31:0] a;
	output reg [31:0] y;
	function automatic [31:0] msb32;
		input reg [31:0] v;
		reg signed [31:0] i;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			begin : sv2v_autoblock_1
				reg signed [31:0] _sv2v_value_on_break;
				for (i = 31; i >= 0; i = i - 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						if (v[i]) begin
							msb32 = i;
							_sv2v_jump = 2'b11;
						end
						_sv2v_value_on_break = i;
					end
				if (!(_sv2v_jump < 2'b10))
					i = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
			if (_sv2v_jump == 2'b00) begin
				msb32 = 0;
				_sv2v_jump = 2'b11;
			end
		end
	endfunction
	wire s;
	wire [31:0] abs_a;
	assign s = a[31];
	assign abs_a = (a[31] ? ~a + 32'd1 : a);
	reg [31:0] p;
	reg [7:0] exp_y;
	reg [23:0] sig24;
	reg g;
	reg st;
	reg ru;
	reg [24:0] sig25;
	reg is_zero;
	reg [31:0] shifted_r;
	reg [31:0] shifted_l;
	always @(*) begin
		if (_sv2v_0)
			;
		is_zero = abs_a == 32'd0;
		p = msb32(abs_a);
		exp_y = 8'd127 + {3'd0, p[4:0]};
		shifted_r = abs_a >> (p[4:0] - 5'd23);
		shifted_l = {8'd0, abs_a[23:0]} << (5'd23 - p[4:0]);
		if (p > 23) begin
			sig24 = shifted_r[23:0];
			g = abs_a[p[4:0] - 5'd24];
			st = (p > 24 ? |(abs_a & ((32'd1 << (p[4:0] - 5'd24)) - 32'd1)) : 1'b0);
		end
		else begin
			sig24 = shifted_l[23:0];
			g = 1'b0;
			st = 1'b0;
		end
		ru = g & (st | sig24[0]);
		sig25 = {1'b0, sig24} + {24'd0, ru};
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (is_zero)
			y = {s, 31'h00000000};
		else if (sig25[24])
			y = {s, exp_y + 8'd1, sig25[23:1]};
		else
			y = {s, exp_y, sig25[22:0]};
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_quantize_i8 (
	a,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	output reg signed [7:0] y;
	wire s;
	wire [7:0] e;
	wire [22:0] m;
	assign s = a[31];
	assign e = a[30:23];
	assign m = a[22:0];
	wire is_zero;
	wire is_inf;
	wire is_nan;
	assign is_zero = (e == 8'd0) && (m == 23'd0);
	assign is_inf = (e == 8'd255) && (m == 23'd0);
	assign is_nan = (e == 8'd255) && (m != 23'd0);
	wire signed [9:0] E;
	assign E = $signed({2'b00, e}) - 10'sd127;
	wire [23:0] sig24;
	assign sig24 = {1'b1, m};
	reg [4:0] shamt;
	reg [31:0] sig_shifted;
	reg rb;
	reg st;
	reg ru;
	reg [31:0] abs_int;
	function automatic signed [4:0] sv2v_cast_5_signed;
		input reg signed [4:0] inp;
		sv2v_cast_5_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		shamt = 5'd0;
		sig_shifted = 32'd0;
		rb = 1'b0;
		st = 1'b0;
		abs_int = 32'd0;
		if (((((E >= -10'sd1) && !is_zero) && !is_inf) && !is_nan) && (e != 8'd0)) begin
			if (E >= 10'sd23) begin
				abs_int = 32'h7fffffff;
				rb = 1'b0;
				st = 1'b0;
			end
			else begin
				shamt = sv2v_cast_5_signed(10'sd23 - E);
				sig_shifted = {8'd0, sig24} >> shamt;
				abs_int = sig_shifted;
				if (shamt >= 5'd1) begin
					rb = sig24[shamt - 5'd1];
					if (shamt >= 5'd2)
						st = |(sig24 & ((24'd1 << (shamt - 5'd1)) - 24'd1));
					else
						st = 1'b0;
				end
				else begin
					rb = 1'b0;
					st = 1'b0;
				end
			end
		end
		ru = rb & (st | abs_int[0]);
	end
	reg [31:0] abs_rnd;
	reg signed [31:0] signed_val;
	always @(*) begin
		if (_sv2v_0)
			;
		abs_rnd = abs_int + {31'd0, ru};
		signed_val = (s ? -$signed(abs_rnd) : $signed(abs_rnd));
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (is_nan)
			y = 8'sd0;
		else if (is_inf)
			y = (s ? -8'sd128 : 8'sd127);
		else if (signed_val > 32'sd127)
			y = 8'sd127;
		else if (signed_val < -32'sd128)
			y = -8'sd128;
		else
			y = signed_val[7:0];
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_exp (
	a,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	output reg [31:0] y;
	localparam [31:0] QNAN = 32'h7fc00000;
	localparam [31:0] POS_INF = 32'h7f800000;
	localparam [31:0] POS_ZERO = 32'h00000000;
	localparam [31:0] C_LOG2E = 32'h3fb8aa3b;
	localparam [31:0] C_LN2_HI = 32'h3f317200;
	localparam [31:0] C_LN2_LO = 32'h35bfbe8e;
	localparam [31:0] C_ONE = 32'h3f800000;
	localparam [31:0] C_HALF = 32'h3f000000;
	localparam [31:0] C_1_6 = 32'h3e2aaaab;
	localparam [31:0] C_1_24 = 32'h3d2aaaab;
	localparam [31:0] C_1_120 = 32'h3c088889;
	localparam [31:0] C_1_720 = 32'h3ab60b61;
	wire sa;
	wire [7:0] ea;
	wire [22:0] ma;
	assign sa = a[31];
	assign ea = a[30:23];
	assign ma = a[22:0];
	wire a_zero;
	wire a_inf;
	wire a_nan;
	assign a_zero = (ea == 8'd0) && (ma == 23'd0);
	assign a_inf = (ea == 8'd255) && (ma == 23'd0);
	assign a_nan = (ea == 8'd255) && (ma != 23'd0);
	wire a_overflow;
	wire a_underflow;
	assign a_overflow = !sa && (a > 32'h42b17218);
	assign a_underflow = sa && ((a & 32'h7fffffff) > 32'h42cff1b5);
	wire [31:0] k_f;
	fp32_mul u_mul1(
		.a(a),
		.b(C_LOG2E),
		.y(k_f)
	);
	wire ksgn;
	wire [7:0] kexp;
	wire [22:0] kmant;
	reg signed [9:0] k_int;
	assign ksgn = k_f[31];
	assign kexp = k_f[30:23];
	assign kmant = k_f[22:0];
	function automatic [4:0] sv2v_cast_5;
		input reg [4:0] inp;
		sv2v_cast_5 = inp;
	endfunction
	function automatic [9:0] sv2v_cast_10;
		input reg [9:0] inp;
		sv2v_cast_10 = inp;
	endfunction
	function automatic signed [9:0] sv2v_cast_10_signed;
		input reg signed [9:0] inp;
		sv2v_cast_10_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		if (kexp < 8'd126)
			k_int = 10'sd0;
		else begin : sv2v_autoblock_1
			reg [4:0] kshamt;
			reg [23:0] ksig24;
			reg [9:0] kabs_int;
			reg kg;
			reg kst;
			reg kru;
			ksig24 = {1'b1, kmant};
			kshamt = sv2v_cast_5(8'd150 - kexp);
			if (kshamt == 5'd0) begin
				kabs_int = ksig24[9:0];
				kg = 1'b0;
				kst = 1'b0;
			end
			else begin
				kabs_int = sv2v_cast_10(ksig24 >> kshamt);
				kg = ksig24[kshamt - 5'd1];
				if (kshamt >= 5'd2)
					kst = |(ksig24 & ((24'd1 << (kshamt - 5'd1)) - 24'd1));
				else
					kst = 1'b0;
			end
			kru = kg & (kst | kabs_int[0]);
			kabs_int = kabs_int + {9'd0, kru};
			k_int = (ksgn ? sv2v_cast_10_signed(-$signed({1'b0, kabs_int})) : sv2v_cast_10_signed($signed({1'b0, kabs_int})));
		end
	end
	reg [31:0] k_fp32;
	reg [9:0] kabs;
	reg kabs_sign;
	reg [3:0] kp;
	reg [22:0] km23;
	reg [7:0] ke8;
	reg [22:0] km23_shifted;
	always @(*) begin : sv2v_autoblock_2
		reg [0:1] _sv2v_jump;
		_sv2v_jump = 2'b00;
		if (_sv2v_0)
			;
		kabs_sign = k_int < 10'sd0;
		kabs = (kabs_sign ? -k_int : k_int);
		kp = 4'd0;
		begin : sv2v_autoblock_3
			reg signed [31:0] j;
			begin : sv2v_autoblock_4
				reg signed [31:0] _sv2v_value_on_break;
				for (j = 9; j >= 0; j = j - 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						if (kabs[j]) begin
							kp = j[3:0];
							_sv2v_jump = 2'b10;
						end
						_sv2v_value_on_break = j;
					end
				if (!(_sv2v_jump < 2'b10))
					j = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
		end
		if (_sv2v_jump == 2'b00) begin
			ke8 = 8'd127 + {4'd0, kp};
			km23_shifted = {13'd0, kabs} << (5'd23 - {1'b0, kp});
			km23 = km23_shifted & 23'h7fffff;
			if (kabs == 10'd0)
				k_fp32 = 32'd0;
			else
				k_fp32 = {kabs_sign, ke8, km23};
		end
	end
	wire [31:0] k_ln2_hi;
	wire [31:0] k_ln2_lo;
	wire [31:0] r_hi;
	wire [31:0] r;
	fp32_mul u_mul2_hi(
		.a(k_fp32),
		.b(C_LN2_HI),
		.y(k_ln2_hi)
	);
	fp32_mul u_mul2_lo(
		.a(k_fp32),
		.b(C_LN2_LO),
		.y(k_ln2_lo)
	);
	fp32_add u_sub_hi(
		.a(a),
		.b({~k_ln2_hi[31], k_ln2_hi[30:0]}),
		.y(r_hi)
	);
	fp32_add u_sub_lo(
		.a(r_hi),
		.b({~k_ln2_lo[31], k_ln2_lo[30:0]}),
		.y(r)
	);
	wire [31:0] t0a;
	wire [31:0] t1s;
	wire [31:0] t1m;
	wire [31:0] t2s;
	wire [31:0] t2m;
	wire [31:0] t3s;
	wire [31:0] t3m;
	wire [31:0] t4s;
	wire [31:0] t4m;
	wire [31:0] t5s;
	wire [31:0] t5m;
	wire [31:0] exp_r;
	fp32_mul m0(
		.a(r),
		.b(C_1_720),
		.y(t0a)
	);
	fp32_add a1(
		.a(t0a),
		.b(C_1_120),
		.y(t1s)
	);
	fp32_mul m1(
		.a(t1s),
		.b(r),
		.y(t1m)
	);
	fp32_add a2(
		.a(t1m),
		.b(C_1_24),
		.y(t2s)
	);
	fp32_mul m2(
		.a(t2s),
		.b(r),
		.y(t2m)
	);
	fp32_add a3(
		.a(t2m),
		.b(C_1_6),
		.y(t3s)
	);
	fp32_mul m3(
		.a(t3s),
		.b(r),
		.y(t3m)
	);
	fp32_add a4(
		.a(t3m),
		.b(C_HALF),
		.y(t4s)
	);
	fp32_mul m4(
		.a(t4s),
		.b(r),
		.y(t4m)
	);
	fp32_add a5(
		.a(t4m),
		.b(C_ONE),
		.y(t5s)
	);
	fp32_mul m5(
		.a(t5s),
		.b(r),
		.y(t5m)
	);
	fp32_add a6(
		.a(t5m),
		.b(C_ONE),
		.y(exp_r)
	);
	wire s_er;
	wire [7:0] e_er;
	wire [22:0] m_er;
	wire signed [10:0] e_scaled;
	assign s_er = exp_r[31];
	assign e_er = exp_r[30:23];
	assign m_er = exp_r[22:0];
	assign e_scaled = $signed({3'b000, e_er}) + {k_int[9], k_int};
	reg [4:0] sub_shamt;
	reg [23:0] sub_sig_in;
	reg [23:0] sub_sig_shifted;
	reg [22:0] sub_mant;
	function automatic signed [4:0] sv2v_cast_5_signed;
		input reg signed [4:0] inp;
		sv2v_cast_5_signed = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		sub_shamt = sv2v_cast_5_signed(11'sd1 - e_scaled);
		sub_sig_in = {1'b1, m_er};
		sub_sig_shifted = sub_sig_in >> sub_shamt;
		sub_mant = sub_sig_shifted[22:0];
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_nan)
			y = QNAN;
		else if (a_inf)
			y = (sa ? POS_ZERO : POS_INF);
		else if (a_zero)
			y = C_ONE;
		else if (a_overflow)
			y = POS_INF;
		else if (a_underflow)
			y = POS_ZERO;
		else if (e_scaled >= 11'sd255)
			y = POS_INF;
		else if (e_scaled <= -11'sd22)
			y = POS_ZERO;
		else if (e_scaled <= 11'sd0)
			y = {9'h000, sub_mant};
		else
			y = {s_er, e_scaled[7:0], m_er};
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_gelu_new (
	a,
	y
);
	input wire [31:0] a;
	output wire [31:0] y;
	localparam [31:0] C_HALF = 32'h3f000000;
	localparam [31:0] C_ONE = 32'h3f800000;
	localparam [31:0] C_TWO = 32'h40000000;
	localparam [31:0] C_K_SQRT2PI = 32'h3f4c4229;
	localparam [31:0] C_044715 = 32'h3d372713;
	wire [31:0] x_sq;
	fp32_mul m_xx(
		.a(a),
		.b(a),
		.y(x_sq)
	);
	wire [31:0] x_cb;
	fp32_mul m_x3(
		.a(x_sq),
		.b(a),
		.y(x_cb)
	);
	wire [31:0] c_x_cb;
	fp32_mul m_c(
		.a(C_044715),
		.b(x_cb),
		.y(c_x_cb)
	);
	wire [31:0] inner_add;
	fp32_add a_in(
		.a(a),
		.b(c_x_cb),
		.y(inner_add)
	);
	wire [31:0] z;
	fp32_mul m_z(
		.a(C_K_SQRT2PI),
		.b(inner_add),
		.y(z)
	);
	wire [31:0] z2;
	fp32_mul m_z2(
		.a(z),
		.b(C_TWO),
		.y(z2)
	);
	wire [31:0] exp_2z;
	fp32_exp e_2z(
		.a(z2),
		.y(exp_2z)
	);
	wire [31:0] denom;
	fp32_add a_dn(
		.a(exp_2z),
		.b(C_ONE),
		.y(denom)
	);
	wire [31:0] ratio;
	fp32_div d_r(
		.a(exp_2z),
		.b(denom),
		.y(ratio)
	);
	fp32_mul m_out(
		.a(a),
		.b(ratio),
		.y(y)
	);
endmodule
module decode_unit (
	insn_data,
	insn
);
	reg _sv2v_0;
	input wire [63:0] insn_data;
	output reg [286:0] insn;
	wire [4:0] opcode_raw;
	assign opcode_raw = insn_data[63:59];
	wire illegal_opcode;
	assign illegal_opcode = ((((((opcode_raw == 5'h0e) || (opcode_raw == 5'h0f)) || (opcode_raw == 5'h10)) || (opcode_raw == 5'h12)) || (opcode_raw == 5'h15)) || (opcode_raw == 5'h16)) || (opcode_raw == 5'h1c);
	wire illegal_attn_reserved;
	assign illegal_attn_reserved = (opcode_raw == 5'h14) && |insn_data[32:0];
	reg illegal_buf;
	always @(*) begin
		if (_sv2v_0)
			;
		illegal_buf = 1'b0;
		if (!illegal_opcode)
			case (opcode_raw)
				5'h0a, 5'h0b, 5'h0c, 5'h0d, 5'h11, 5'h13, 5'h17, 5'h18, 5'h19, 5'h1a, 5'h1b, 5'h1d, 5'h1e, 5'h1f:
					if (((insn_data[58:57] == 2'b11) || (insn_data[40:39] == 2'b11)) || (insn_data[22:21] == 2'b11))
						illegal_buf = 1'b1;
				5'h07, 5'h08:
					if (insn_data[58:57] == 2'b11)
						illegal_buf = 1'b1;
				5'h09:
					if ((insn_data[58:57] == 2'b11) || (insn_data[40:39] == 2'b11))
						illegal_buf = 1'b1;
				default:
					;
			endcase
	end
	always @(*) begin
		if (_sv2v_0)
			;
		insn[286-:5] = opcode_raw;
		insn[281] = (illegal_opcode || illegal_buf) || illegal_attn_reserved;
		insn[280-:2] = insn_data[58:57];
		insn[278-:16] = insn_data[56:41];
		insn[262-:2] = insn_data[40:39];
		insn[260-:16] = insn_data[38:23];
		insn[244-:2] = insn_data[22:21];
		insn[242-:16] = insn_data[20:5];
		insn[226-:4] = insn_data[4:1];
		insn[222] = insn_data[0];
		insn[221-:2] = insn_data[58:57];
		insn[219-:16] = insn_data[56:41];
		insn[203-:16] = insn_data[40:25];
		insn[187-:2] = insn_data[24:23];
		insn[185-:16] = insn_data[22:7];
		insn[169-:2] = insn_data[58:57];
		insn[167-:16] = insn_data[56:41];
		insn[151-:2] = insn_data[40:39];
		insn[149-:16] = insn_data[38:23];
		insn[133-:16] = insn_data[22:7];
		insn[117-:6] = insn_data[6:1];
		insn[111] = insn_data[0];
		insn[110-:2] = insn_data[58:57];
		insn[108-:28] = insn_data[56:29];
		insn[80-:10] = insn_data[58:49];
		insn[70-:10] = insn_data[48:39];
		insn[60-:10] = insn_data[38:29];
		insn[50-:12] = insn_data[58:47];
		insn[38-:12] = insn_data[46:35];
		insn[26-:2] = insn_data[34:33];
		insn[24-:4] = insn_data[58:55];
		insn[20-:2] = insn_data[54:53];
		insn[18-:16] = insn_data[52:37];
		insn[2-:3] = insn_data[58:56];
	end
	initial _sv2v_0 = 0;
endmodule
module fetch_unit (
	clk,
	rst_n,
	pc,
	fetch_req,
	insn_valid,
	insn_data,
	fetch_fault,
	fetch_fault_code,
	m_axi_ar_addr,
	m_axi_ar_valid,
	m_axi_ar_len,
	m_axi_ar_size,
	m_axi_ar_burst,
	m_axi_ar_ready,
	m_axi_r_data,
	m_axi_r_resp,
	m_axi_r_valid,
	m_axi_r_last,
	m_axi_r_ready
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire [55:0] pc;
	input wire fetch_req;
	output reg insn_valid;
	output reg [63:0] insn_data;
	output reg fetch_fault;
	output reg [3:0] fetch_fault_code;
	localparam signed [31:0] taccel_pkg_AXI_ADDR_W = 56;
	output reg [55:0] m_axi_ar_addr;
	output reg m_axi_ar_valid;
	output wire [7:0] m_axi_ar_len;
	output wire [2:0] m_axi_ar_size;
	output wire [1:0] m_axi_ar_burst;
	input wire m_axi_ar_ready;
	localparam signed [31:0] taccel_pkg_AXI_DATA_W = 128;
	input wire [127:0] m_axi_r_data;
	input wire [1:0] m_axi_r_resp;
	input wire m_axi_r_valid;
	input wire m_axi_r_last;
	output reg m_axi_r_ready;
	assign m_axi_ar_len = 8'h00;
	assign m_axi_ar_size = 3'b100;
	assign m_axi_ar_burst = 2'b01;
	wire [55:0] byte_addr;
	wire [55:0] aligned_addr;
	wire pc_odd;
	assign byte_addr = pc << 3;
	assign aligned_addr = {byte_addr[55:4], 4'b0000};
	assign pc_odd = pc[0];
	function automatic [63:0] bswap64;
		input [63:0] x;
		bswap64 = {x[7:0], x[15:8], x[23:16], x[31:24], x[39:32], x[47:40], x[55:48], x[63:56]};
	endfunction
	reg [1:0] state;
	reg [1:0] next_state;
	reg [3:0] fault_code_r;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 2'd0;
			fault_code_r <= 4'h0;
		end
		else begin
			state <= next_state;
			if (((state == 2'd2) && m_axi_r_valid) && ((m_axi_r_resp != 2'b00) || !m_axi_r_last))
				fault_code_r <= 4'h2;
		end
	reg pc_odd_q;
	always @(posedge clk)
		if (((state == 2'd1) && m_axi_ar_ready) && m_axi_ar_valid)
			pc_odd_q <= pc_odd;
	always @(*) begin
		if (_sv2v_0)
			;
		next_state = state;
		m_axi_ar_valid = 1'b0;
		m_axi_ar_addr = aligned_addr;
		m_axi_r_ready = 1'b0;
		insn_valid = 1'b0;
		insn_data = 64'h0000000000000000;
		fetch_fault = 1'b0;
		fetch_fault_code = fault_code_r;
		case (state)
			2'd0:
				if (fetch_req)
					next_state = 2'd1;
			2'd1: begin
				m_axi_ar_valid = 1'b1;
				m_axi_ar_addr = aligned_addr;
				if (m_axi_ar_ready)
					next_state = 2'd2;
			end
			2'd2: begin
				m_axi_r_ready = 1'b1;
				if (m_axi_r_valid) begin
					if ((m_axi_r_resp != 2'b00) || !m_axi_r_last)
						next_state = 2'd3;
					else begin
						if (pc_odd_q)
							insn_data = bswap64(m_axi_r_data[127:64]);
						else
							insn_data = bswap64(m_axi_r_data[63:0]);
						insn_valid = 1'b1;
						next_state = 2'd0;
					end
				end
			end
			2'd3: fetch_fault = 1'b1;
			default: next_state = 2'd0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
module control_unit (
	clk,
	rst_n,
	start,
	pc,
	fetch_req,
	insn_valid,
	insn,
	scale_we,
	scale_waddr,
	scale_wdata,
	addr_lo_we,
	addr_hi_we,
	addr_wsel,
	addr_imm28,
	tile_we,
	tile_m_in,
	tile_n_in,
	tile_k_in,
	attn_we,
	attn_query_row_base_in,
	attn_valid_kv_len_in,
	attn_mode_in,
	tile_valid,
	tile_n,
	tile_k,
	attn_valid,
	attn_valid_kv_len,
	attn_mode,
	dma_dispatch,
	sys_dispatch,
	sfu_dispatch,
	helper_dispatch,
	dma_busy,
	sys_busy,
	sfu_busy,
	helper_busy,
	ext_fault,
	ext_fault_code,
	done,
	fault,
	fault_code,
	obs_retire_pulse,
	obs_retire_pc,
	obs_retire_opcode,
	obs_ctrl_fault_pulse,
	obs_ctrl_fault_code,
	obs_ctrl_fault_pc,
	obs_ctrl_fault_opcode,
	obs_sync_wait_dma,
	obs_sync_wait_sys,
	obs_sync_wait_sfu
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire start;
	output wire [55:0] pc;
	output reg fetch_req;
	input wire insn_valid;
	input wire [286:0] insn;
	output reg scale_we;
	output reg [3:0] scale_waddr;
	output reg [15:0] scale_wdata;
	output reg addr_lo_we;
	output reg addr_hi_we;
	output reg [1:0] addr_wsel;
	output reg [27:0] addr_imm28;
	output reg tile_we;
	output reg [9:0] tile_m_in;
	output reg [9:0] tile_n_in;
	output reg [9:0] tile_k_in;
	output reg attn_we;
	output reg [11:0] attn_query_row_base_in;
	output reg [11:0] attn_valid_kv_len_in;
	output reg [1:0] attn_mode_in;
	input wire tile_valid;
	input wire [9:0] tile_n;
	input wire [9:0] tile_k;
	input wire attn_valid;
	input wire [11:0] attn_valid_kv_len;
	input wire [1:0] attn_mode;
	output reg dma_dispatch;
	output reg sys_dispatch;
	output reg sfu_dispatch;
	output reg helper_dispatch;
	input wire dma_busy;
	input wire sys_busy;
	input wire sfu_busy;
	input wire helper_busy;
	input wire ext_fault;
	input wire [3:0] ext_fault_code;
	output reg done;
	output reg fault;
	output wire [3:0] fault_code;
	output reg obs_retire_pulse;
	output reg [55:0] obs_retire_pc;
	output reg [4:0] obs_retire_opcode;
	output reg obs_ctrl_fault_pulse;
	output reg [3:0] obs_ctrl_fault_code;
	output reg [55:0] obs_ctrl_fault_pc;
	output reg [4:0] obs_ctrl_fault_opcode;
	output wire obs_sync_wait_dma;
	output wire obs_sync_wait_sys;
	output wire obs_sync_wait_sfu;
	reg [2:0] state;
	reg [55:0] pc_reg;
	reg [2:0] sync_mask_q;
	reg [3:0] fault_code_r;
	assign pc = pc_reg;
	assign fault_code = fault_code_r;
	wire sync_clear_q;
	wire sync_clear_now;
	assign sync_clear_q = (sync_mask_q & {sfu_busy, sys_busy, dma_busy}) == 3'b000;
	assign sync_clear_now = (insn[2-:3] & {sfu_busy, sys_busy, dma_busy}) == 3'b000;
	wire [14:0] attn_softmax_key_cols_w;
	wire [14:0] attn_valid_kv_len_ext_w;
	wire config_attn_valid_now;
	reg masked_attn_valid_now;
	wire is_masked_sfu_op_w;
	assign is_masked_sfu_op_w = ((insn[286-:5] == 5'h15) || (insn[286-:5] == 5'h16)) || (insn[286-:5] == 5'h1d);
	assign attn_softmax_key_cols_w = (insn[286-:5] == 5'h16 ? ({5'h00, tile_k} + 15'd1) << 4 : ({5'h00, tile_n} + 15'd1) << 4);
	assign attn_valid_kv_len_ext_w = {3'h0, attn_valid_kv_len};
	assign config_attn_valid_now = (tile_valid && (insn[26-:2] != 2'b00)) && (insn[38-:12] != 12'h000);
	always @(*) begin
		if (_sv2v_0)
			;
		masked_attn_valid_now = ((tile_valid && attn_valid) && (attn_mode != 2'b00)) && (attn_valid_kv_len != 12'h000);
		if (masked_attn_valid_now) begin
			if (attn_mode == 2'b10)
				masked_attn_valid_now = attn_softmax_key_cols_w == attn_valid_kv_len_ext_w;
			else if (attn_mode[0])
				masked_attn_valid_now = attn_softmax_key_cols_w >= attn_valid_kv_len_ext_w;
		end
	end
	function automatic unsupported_op;
		input reg [4:0] op;
		input reg [1:0] s_src_mode;
		input reg fp_flags;
		case (op)
			5'h04: unsupported_op = s_src_mode != 2'b00;
			5'h19, 5'h1a, 5'h1b, 5'h17, 5'h18, 5'h1d, 5'h1e, 5'h1f: unsupported_op = fp_flags == 1'b0;
			default: unsupported_op = 1'b0;
		endcase
	endfunction
	wire unsupported_now;
	assign unsupported_now = unsupported_op(insn[286-:5], insn[20-:2], insn[222]);
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 3'd0;
			pc_reg <= 56'h00000000000000;
			sync_mask_q <= 3'h0;
			fault_code_r <= 4'h0;
			obs_retire_pulse <= 1'b0;
			obs_retire_pc <= 56'h00000000000000;
			obs_retire_opcode <= 5'h00;
			obs_ctrl_fault_pulse <= 1'b0;
			obs_ctrl_fault_code <= 4'h0;
			obs_ctrl_fault_pc <= 56'h00000000000000;
			obs_ctrl_fault_opcode <= 5'h00;
		end
		else begin
			obs_retire_pulse <= 1'b0;
			obs_ctrl_fault_pulse <= 1'b0;
			case (state)
				3'd0:
					if (start) begin
						pc_reg <= 56'h00000000000000;
						state <= 3'd1;
					end
				3'd1:
					if (ext_fault) begin
						fault_code_r <= ext_fault_code;
						state <= 3'd6;
					end
					else if (insn_valid)
						state <= 3'd2;
				3'd2:
					if (ext_fault) begin
						fault_code_r <= ext_fault_code;
						state <= 3'd6;
					end
					else if (insn[281]) begin
						if (insn[286-:5] == 5'h1c) begin
							fault_code_r <= 4'h1;
							obs_ctrl_fault_code <= 4'h1;
						end
						else if (insn[286-:5] == 5'h14) begin
							fault_code_r <= 4'h1;
							obs_ctrl_fault_code <= 4'h1;
						end
						else begin
							fault_code_r <= 4'h5;
							obs_ctrl_fault_code <= 4'h5;
						end
						obs_ctrl_fault_pulse <= 1'b1;
						obs_ctrl_fault_pc <= pc_reg;
						obs_ctrl_fault_opcode <= insn[286-:5];
						state <= 3'd6;
					end
					else if (unsupported_now) begin
						fault_code_r <= 4'h6;
						obs_ctrl_fault_pulse <= 1'b1;
						obs_ctrl_fault_code <= 4'h6;
						obs_ctrl_fault_pc <= pc_reg;
						obs_ctrl_fault_opcode <= insn[286-:5];
						state <= 3'd6;
					end
					else
						case (insn[286-:5])
							5'h00: begin
								obs_retire_pulse <= 1'b1;
								obs_retire_pc <= pc_reg;
								obs_retire_opcode <= insn[286-:5];
								pc_reg <= pc_reg + 56'h00000000000001;
								state <= 3'd1;
							end
							5'h01: begin
								obs_retire_pulse <= 1'b1;
								obs_retire_pc <= pc_reg;
								obs_retire_opcode <= insn[286-:5];
								state <= 3'd5;
							end
							5'h02:
								if (sync_clear_now) begin
									obs_retire_pulse <= 1'b1;
									obs_retire_pc <= pc_reg;
									obs_retire_opcode <= insn[286-:5];
									pc_reg <= pc_reg + 56'h00000000000001;
									state <= 3'd1;
								end
								else begin
									sync_mask_q <= insn[2-:3];
									state <= 3'd3;
								end
							5'h03: begin
								obs_retire_pulse <= 1'b1;
								obs_retire_pc <= pc_reg;
								obs_retire_opcode <= insn[286-:5];
								pc_reg <= pc_reg + 56'h00000000000001;
								state <= 3'd1;
							end
							5'h14:
								if (!config_attn_valid_now) begin
									fault_code_r <= 4'h4;
									obs_ctrl_fault_pulse <= 1'b1;
									obs_ctrl_fault_code <= 4'h4;
									obs_ctrl_fault_pc <= pc_reg;
									obs_ctrl_fault_opcode <= insn[286-:5];
									state <= 3'd6;
								end
								else begin
									obs_retire_pulse <= 1'b1;
									obs_retire_pc <= pc_reg;
									obs_retire_opcode <= insn[286-:5];
									pc_reg <= pc_reg + 56'h00000000000001;
									state <= 3'd1;
								end
							5'h04: begin
								obs_retire_pulse <= 1'b1;
								obs_retire_pc <= pc_reg;
								obs_retire_opcode <= insn[286-:5];
								pc_reg <= pc_reg + 56'h00000000000001;
								state <= 3'd1;
							end
							5'h05: begin
								obs_retire_pulse <= 1'b1;
								obs_retire_pc <= pc_reg;
								obs_retire_opcode <= insn[286-:5];
								pc_reg <= pc_reg + 56'h00000000000001;
								state <= 3'd1;
							end
							5'h06: begin
								obs_retire_pulse <= 1'b1;
								obs_retire_pc <= pc_reg;
								obs_retire_opcode <= insn[286-:5];
								pc_reg <= pc_reg + 56'h00000000000001;
								state <= 3'd1;
							end
							5'h09:
								if (((dma_busy || sys_busy) || helper_busy) || sfu_busy)
									state <= 3'd2;
								else
									state <= 3'd4;
							5'h07, 5'h08:
								if (sfu_busy || dma_busy)
									state <= 3'd2;
								else begin
									obs_retire_pulse <= 1'b1;
									obs_retire_pc <= pc_reg;
									obs_retire_opcode <= insn[286-:5];
									pc_reg <= pc_reg + 56'h00000000000001;
									state <= 3'd1;
								end
							5'h0b, 5'h11, 5'h0c, 5'h0d, 5'h13:
								if (!tile_valid) begin
									fault_code_r <= 4'h4;
									obs_ctrl_fault_pulse <= 1'b1;
									obs_ctrl_fault_code <= 4'h4;
									obs_ctrl_fault_pc <= pc_reg;
									obs_ctrl_fault_opcode <= insn[286-:5];
									state <= 3'd6;
								end
								else if (((dma_busy || sys_busy) || helper_busy) || sfu_busy)
									state <= 3'd2;
								else
									state <= 3'd4;
							5'h0a:
								if (!tile_valid) begin
									fault_code_r <= 4'h4;
									obs_ctrl_fault_pulse <= 1'b1;
									obs_ctrl_fault_code <= 4'h4;
									obs_ctrl_fault_pc <= pc_reg;
									obs_ctrl_fault_opcode <= insn[286-:5];
									state <= 3'd6;
								end
								else if (sfu_busy || helper_busy)
									state <= 3'd2;
								else begin
									obs_retire_pulse <= 1'b1;
									obs_retire_pc <= pc_reg;
									obs_retire_opcode <= insn[286-:5];
									pc_reg <= pc_reg + 56'h00000000000001;
									state <= 3'd1;
								end
							5'h0e, 5'h15, 5'h12, 5'h16, 5'h0f, 5'h10, 5'h19, 5'h1a, 5'h1b, 5'h17, 5'h18, 5'h1d, 5'h1e, 5'h1f:
								if (!tile_valid) begin
									fault_code_r <= 4'h4;
									obs_ctrl_fault_pulse <= 1'b1;
									obs_ctrl_fault_code <= 4'h4;
									obs_ctrl_fault_pc <= pc_reg;
									obs_ctrl_fault_opcode <= insn[286-:5];
									state <= 3'd6;
								end
								else if (is_masked_sfu_op_w && !masked_attn_valid_now) begin
									fault_code_r <= 4'h4;
									obs_ctrl_fault_pulse <= 1'b1;
									obs_ctrl_fault_code <= 4'h4;
									obs_ctrl_fault_pc <= pc_reg;
									obs_ctrl_fault_opcode <= insn[286-:5];
									state <= 3'd6;
								end
								else if (((dma_busy || sys_busy) || helper_busy) || sfu_busy)
									state <= 3'd2;
								else begin
									obs_retire_pulse <= 1'b1;
									obs_retire_pc <= pc_reg;
									obs_retire_opcode <= insn[286-:5];
									pc_reg <= pc_reg + 56'h00000000000001;
									state <= 3'd1;
								end
							default: begin
								fault_code_r <= 4'h1;
								obs_ctrl_fault_pulse <= 1'b1;
								obs_ctrl_fault_code <= 4'h1;
								obs_ctrl_fault_pc <= pc_reg;
								obs_ctrl_fault_opcode <= insn[286-:5];
								state <= 3'd6;
							end
						endcase
				3'd3:
					if (ext_fault) begin
						fault_code_r <= ext_fault_code;
						state <= 3'd6;
					end
					else if (sync_clear_q) begin
						obs_retire_pulse <= 1'b1;
						obs_retire_pc <= pc_reg;
						obs_retire_opcode <= 5'h02;
						pc_reg <= pc_reg + 56'h00000000000001;
						state <= 3'd1;
					end
				3'd4:
					if (ext_fault) begin
						fault_code_r <= ext_fault_code;
						state <= 3'd6;
					end
					else if (!helper_busy) begin
						obs_retire_pulse <= 1'b1;
						obs_retire_pc <= pc_reg;
						obs_retire_opcode <= insn[286-:5];
						pc_reg <= pc_reg + 56'h00000000000001;
						state <= 3'd1;
					end
				3'd5, 3'd6:
					;
				default: state <= 3'd0;
			endcase
		end
	always @(*) begin : sv2v_autoblock_1
		reg helper_ready_now;
		reg sfu_ready_now;
		if (_sv2v_0)
			;
		helper_ready_now = ((!dma_busy && !sys_busy) && !helper_busy) && !sfu_busy;
		sfu_ready_now = ((!dma_busy && !sys_busy) && !helper_busy) && !sfu_busy;
		fetch_req = 1'b0;
		done = 1'b0;
		fault = 1'b0;
		scale_we = 1'b0;
		scale_waddr = insn[24-:4];
		scale_wdata = insn[18-:16];
		addr_lo_we = 1'b0;
		addr_hi_we = 1'b0;
		addr_wsel = insn[110-:2];
		addr_imm28 = insn[108-:28];
		tile_we = 1'b0;
		tile_m_in = insn[80-:10];
		tile_n_in = insn[70-:10];
		tile_k_in = insn[60-:10];
		attn_we = 1'b0;
		attn_query_row_base_in = insn[50-:12];
		attn_valid_kv_len_in = insn[38-:12];
		attn_mode_in = insn[26-:2];
		dma_dispatch = 1'b0;
		sys_dispatch = 1'b0;
		sfu_dispatch = 1'b0;
		helper_dispatch = 1'b0;
		case (state)
			3'd1: fetch_req = 1'b1;
			3'd2:
				if ((!insn[281] && !ext_fault) && !unsupported_now)
					case (insn[286-:5])
						5'h03: tile_we = 1'b1;
						5'h14: attn_we = config_attn_valid_now;
						5'h04: scale_we = 1'b1;
						5'h05: addr_lo_we = 1'b1;
						5'h06: addr_hi_we = 1'b1;
						5'h07, 5'h08: dma_dispatch = !sfu_busy && !dma_busy;
						5'h0a: sys_dispatch = (tile_valid && !sfu_busy) && !helper_busy;
						5'h09: helper_dispatch = helper_ready_now;
						5'h0b, 5'h11, 5'h0c, 5'h0d, 5'h13: helper_dispatch = tile_valid && helper_ready_now;
						5'h0e, 5'h12, 5'h0f, 5'h10, 5'h19, 5'h1a, 5'h1b, 5'h17, 5'h18, 5'h1e, 5'h1f: sfu_dispatch = tile_valid && sfu_ready_now;
						5'h15, 5'h16, 5'h1d: sfu_dispatch = masked_attn_valid_now && sfu_ready_now;
						default:
							;
					endcase
			3'd5: done = 1'b1;
			3'd6: fault = 1'b1;
			default:
				;
		endcase
	end
	localparam signed [31:0] taccel_pkg_SYNC_DMA_BIT = 0;
	assign obs_sync_wait_dma = ((state == 3'd3) && sync_mask_q[taccel_pkg_SYNC_DMA_BIT]) && dma_busy;
	localparam signed [31:0] taccel_pkg_SYNC_SYS_BIT = 1;
	assign obs_sync_wait_sys = ((state == 3'd3) && sync_mask_q[taccel_pkg_SYNC_SYS_BIT]) && sys_busy;
	localparam signed [31:0] taccel_pkg_SYNC_SFU_BIT = 2;
	assign obs_sync_wait_sfu = ((state == 3'd3) && sync_mask_q[taccel_pkg_SYNC_SFU_BIT]) && sfu_busy;
	initial _sv2v_0 = 0;
endmodule
module blocking_helper_engine (
	clk,
	rst_n,
	dispatch,
	opcode,
	src1_buf,
	src1_off,
	src2_buf,
	src2_off,
	dst_buf,
	dst_off,
	sreg,
	b_length,
	b_src_rows,
	b_transpose,
	tile_m,
	tile_n,
	scale0_data,
	scale1_data,
	helper_busy,
	helper_fault,
	helper_fault_code,
	sram_a_en,
	sram_a_we,
	sram_a_buf,
	sram_a_row,
	sram_a_wdata,
	sram_a_rdata,
	sram_a_fault,
	sram_b_en,
	sram_b_buf,
	sram_b_row,
	sram_b_rdata,
	sram_b_fault
);
	reg _sv2v_0;
	parameter signed [31:0] HELPER_SYNTH_MODE = 0;
	input wire clk;
	input wire rst_n;
	input wire dispatch;
	input wire [4:0] opcode;
	input wire [1:0] src1_buf;
	input wire [15:0] src1_off;
	input wire [1:0] src2_buf;
	input wire [15:0] src2_off;
	input wire [1:0] dst_buf;
	input wire [15:0] dst_off;
	input wire [3:0] sreg;
	input wire [15:0] b_length;
	input wire [5:0] b_src_rows;
	input wire b_transpose;
	input wire [9:0] tile_m;
	input wire [9:0] tile_n;
	input wire [15:0] scale0_data;
	input wire [15:0] scale1_data;
	output reg helper_busy;
	output reg helper_fault;
	output reg [3:0] helper_fault_code;
	output reg sram_a_en;
	output reg sram_a_we;
	output reg [1:0] sram_a_buf;
	output reg [15:0] sram_a_row;
	output reg [127:0] sram_a_wdata;
	input wire [127:0] sram_a_rdata;
	input wire sram_a_fault;
	output reg sram_b_en;
	output reg [1:0] sram_b_buf;
	output reg [15:0] sram_b_row;
	input wire [127:0] sram_b_rdata;
	input wire sram_b_fault;
	reg [5:0] state;
	reg [4:0] opcode_q;
	reg [1:0] src1_buf_q;
	reg [1:0] src2_buf_q;
	reg [1:0] dst_buf_q;
	reg [15:0] src1_off_q;
	reg [15:0] src2_off_q;
	reg [15:0] dst_off_q;
	reg [3:0] sreg_q;
	reg [15:0] b_length_q;
	reg [5:0] b_src_rows_q;
	reg b_transpose_q;
	reg [15:0] scale0_q;
	reg [15:0] scale1_q;
	reg [14:0] m_rows_q;
	reg [10:0] n_tiles_q;
	reg [12:0] n_chunks_i32_q;
	reg [3:0] fault_code_r;
	reg [31:0] step_idx_q;
	reg flat_backward_q;
	reg [15:0] trans_row_count_q;
	reg [15:0] trans_cols_q;
	reg [15:0] trans_rbase_q;
	reg [15:0] trans_cbase_q;
	reg [4:0] trans_height_q;
	reg [4:0] trans_width_q;
	reg [4:0] trans_src_row_idx_q;
	reg [4:0] trans_dst_row_idx_q;
	reg [127:0] trans_first_row_q;
	reg [127:0] trans_scratch_q [0:15];
	reg [15:0] bias_chunk_q;
	reg [14:0] bias_row_idx_q;
	reg [127:0] bias_data_q;
	reg [14:0] rq_row_idx_q;
	reg [10:0] rq_col_chunk_q;
	reg [1:0] rq_part_q;
	reg [127:0] rq_row0_q;
	reg [127:0] rq_row1_q;
	reg [127:0] rq_row2_q;
	reg [127:0] rq_row3_q;
	reg [127:0] skip_row_q;
	reg [15:0] pc_scale_chunk_q [0:15];
	localparam signed [31:0] taccel_pkg_ABUF_ROWS = 8192;
	localparam signed [31:0] taccel_pkg_ACCUM_ROWS = 4096;
	localparam signed [31:0] taccel_pkg_WBUF_ROWS = 16384;
	function automatic signed [15:0] sv2v_cast_16_signed;
		input reg signed [15:0] inp;
		sv2v_cast_16_signed = inp;
	endfunction
	function automatic [15:0] buf_rows;
		input reg [1:0] bid;
		case (bid)
			2'b00: buf_rows = sv2v_cast_16_signed(taccel_pkg_ABUF_ROWS);
			2'b01: buf_rows = sv2v_cast_16_signed(taccel_pkg_WBUF_ROWS);
			2'b10: buf_rows = sv2v_cast_16_signed(taccel_pkg_ACCUM_ROWS);
			default: buf_rows = 16'h0000;
		endcase
	endfunction
	function automatic [4:0] block_span;
		input reg [15:0] total;
		input reg [15:0] base;
		reg [15:0] rem;
		begin
			rem = total - base;
			if (rem > 16)
				block_span = 5'd16;
			else
				block_span = rem[4:0];
		end
	endfunction
	function automatic [7:0] get_byte;
		input reg [127:0] row;
		input integer idx;
		get_byte = row[idx * 8+:8];
	endfunction
	function automatic [15:0] get_u16;
		input reg [127:0] row;
		input integer idx;
		get_u16 = row[idx * 16+:16];
	endfunction
	function automatic [127:0] sat_add_int8_row;
		input reg [127:0] a_row;
		input reg [127:0] b_row;
		reg signed [8:0] sum;
		reg signed [7:0] a_i8;
		reg signed [7:0] b_i8;
		reg [127:0] out_row;
		integer i;
		begin
			out_row = 128'h00000000000000000000000000000000;
			for (i = 0; i < 16; i = i + 1)
				begin
					a_i8 = a_row[i * 8+:8];
					b_i8 = b_row[i * 8+:8];
					sum = $signed(a_i8) + $signed(b_i8);
					if (sum > 9'sd127)
						out_row[i * 8+:8] = 8'h7f;
					else if (sum < -9'sd128)
						out_row[i * 8+:8] = 8'h80;
					else
						out_row[i * 8+:8] = sum[7:0];
				end
			sat_add_int8_row = out_row;
		end
	endfunction
	function automatic [127:0] add_wrap_int32_row;
		input reg [127:0] a_row;
		input reg [127:0] b_row;
		reg signed [31:0] a_i32;
		reg signed [31:0] b_i32;
		reg signed [31:0] sum_i32;
		reg [127:0] out_row;
		integer i;
		begin
			out_row = 128'h00000000000000000000000000000000;
			for (i = 0; i < 4; i = i + 1)
				begin
					a_i32 = a_row[i * 32+:32];
					b_i32 = b_row[i * 32+:32];
					sum_i32 = $signed(a_i32) + $signed(b_i32);
					out_row[i * 32+:32] = sum_i32;
				end
			add_wrap_int32_row = out_row;
		end
	endfunction
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	function automatic signed [63:0] fp16_mul_round_even;
		input reg signed [31:0] src_val;
		input reg [15:0] fp16_val;
		reg sign_h;
		reg [4:0] exp_h;
		reg [9:0] frac_h;
		reg signed [12:0] mant;
		integer shift_amt;
		reg signed [63:0] prod;
		reg signed [63:0] abs_prod;
		reg signed [63:0] quot;
		reg signed [63:0] rem;
		reg signed [63:0] half;
		begin
			sign_h = fp16_val[15];
			exp_h = fp16_val[14:10];
			frac_h = fp16_val[9:0];
			if ((exp_h == 5'h00) && (frac_h == 10'h000))
				fp16_mul_round_even = 64'sd0;
			else begin
				if (exp_h == 5'h00) begin
					mant = $signed({3'b000, frac_h});
					shift_amt = -24;
				end
				else begin
					mant = $signed({3'b001, frac_h});
					shift_amt = sv2v_cast_32_signed(exp_h) - 25;
				end
				prod = $signed(src_val) * $signed(mant);
				if (sign_h)
					prod = -prod;
				if (shift_amt >= 0)
					fp16_mul_round_even = prod <<< shift_amt;
				else begin
					abs_prod = (prod < 0 ? -prod : prod);
					quot = abs_prod >>> -shift_amt;
					rem = abs_prod & ((64'sd1 <<< -shift_amt) - 64'sd1);
					half = 64'sd1 <<< -(shift_amt + 1);
					if ((rem > half) || ((rem == half) && quot[0]))
						quot = quot + 64'sd1;
					fp16_mul_round_even = (prod < 0 ? -quot : quot);
				end
			end
		end
	endfunction
	function automatic [127:0] scale_mul_i8_row;
		input reg [127:0] row;
		input reg [15:0] scale_val;
		reg signed [7:0] src_i8;
		reg signed [63:0] scaled;
		reg [127:0] out_row;
		integer i;
		begin
			out_row = 128'h00000000000000000000000000000000;
			for (i = 0; i < 16; i = i + 1)
				begin
					src_i8 = row[i * 8+:8];
					scaled = fp16_mul_round_even({{24 {src_i8[7]}}, src_i8}, scale_val);
					if (scaled > 64'sd127)
						out_row[i * 8+:8] = 8'h7f;
					else if (scaled < -64'sd128)
						out_row[i * 8+:8] = 8'h80;
					else
						out_row[i * 8+:8] = scaled[7:0];
				end
			scale_mul_i8_row = out_row;
		end
	endfunction
	function automatic [127:0] scale_mul_i32_row;
		input reg [127:0] row;
		input reg [15:0] scale_val;
		reg signed [31:0] src_i32;
		reg signed [63:0] scaled;
		reg [127:0] out_row;
		integer i;
		begin
			out_row = 128'h00000000000000000000000000000000;
			for (i = 0; i < 4; i = i + 1)
				begin
					src_i32 = row[i * 32+:32];
					scaled = fp16_mul_round_even(src_i32, scale_val);
					if (scaled > 64'sd2147483647)
						out_row[i * 32+:32] = 32'h7fffffff;
					else if (scaled < -64'sd2147483648)
						out_row[i * 32+:32] = 32'h80000000;
					else
						out_row[i * 32+:32] = scaled[31:0];
				end
			scale_mul_i32_row = out_row;
		end
	endfunction
	function automatic [127:0] requant_pack;
		input reg [127:0] row0;
		input reg [127:0] row1;
		input reg [127:0] row2;
		input reg [127:0] row3;
		input reg [15:0] scale_val;
		reg signed [31:0] src_i32;
		reg signed [63:0] scaled;
		reg [127:0] out_row;
		integer i;
		begin
			out_row = 128'h00000000000000000000000000000000;
			for (i = 0; i < 4; i = i + 1)
				begin
					src_i32 = row0[i * 32+:32];
					scaled = fp16_mul_round_even(src_i32, scale_val);
					if (scaled > 64'sd127)
						out_row[i * 8+:8] = 8'h7f;
					else if (scaled < -64'sd128)
						out_row[i * 8+:8] = 8'h80;
					else
						out_row[i * 8+:8] = scaled[7:0];
					src_i32 = row1[i * 32+:32];
					scaled = fp16_mul_round_even(src_i32, scale_val);
					if (scaled > 64'sd127)
						out_row[(i + 4) * 8+:8] = 8'h7f;
					else if (scaled < -64'sd128)
						out_row[(i + 4) * 8+:8] = 8'h80;
					else
						out_row[(i + 4) * 8+:8] = scaled[7:0];
					src_i32 = row2[i * 32+:32];
					scaled = fp16_mul_round_even(src_i32, scale_val);
					if (scaled > 64'sd127)
						out_row[(i + 8) * 8+:8] = 8'h7f;
					else if (scaled < -64'sd128)
						out_row[(i + 8) * 8+:8] = 8'h80;
					else
						out_row[(i + 8) * 8+:8] = scaled[7:0];
					src_i32 = row3[i * 32+:32];
					scaled = fp16_mul_round_even(src_i32, scale_val);
					if (scaled > 64'sd127)
						out_row[(i + 12) * 8+:8] = 8'h7f;
					else if (scaled < -64'sd128)
						out_row[(i + 12) * 8+:8] = 8'h80;
					else
						out_row[(i + 12) * 8+:8] = scaled[7:0];
				end
			requant_pack = out_row;
		end
	endfunction
	function automatic [127:0] extract_window;
		input reg [127:0] row0;
		input reg [127:0] row1;
		input reg [3:0] start_byte;
		input reg [4:0] width;
		reg [127:0] out_row;
		integer i;
		integer src_idx;
		begin
			out_row = 128'h00000000000000000000000000000000;
			for (i = 0; i < 16; i = i + 1)
				if (i < sv2v_cast_32_signed(width)) begin
					src_idx = sv2v_cast_32_signed(start_byte) + i;
					if (src_idx < 16)
						out_row[i * 8+:8] = get_byte(row0, src_idx);
					else
						out_row[i * 8+:8] = get_byte(row1, src_idx - 16);
				end
			extract_window = out_row;
		end
	endfunction
	wire [14:0] dispatch_m_rows_w;
	wire [10:0] dispatch_n_tiles_w;
	wire [12:0] dispatch_n_chunks_i32_w;
	wire [31:0] dispatch_int8_units_w;
	wire [31:0] dispatch_int32_units_w;
	wire [31:0] dispatch_scale_rows_w;
	wire [31:0] dispatch_copy_units_w;
	wire [15:0] dispatch_src_rows_w;
	wire [15:0] dispatch_trans_cols_w;
	wire [15:0] dispatch_src_buf_rows_w;
	wire [15:0] dispatch_src2_buf_rows_w;
	wire [15:0] dispatch_dst_buf_rows_w;
	reg dispatch_unsupported_w;
	reg dispatch_sram_oob_w;
	wire dispatch_is_vadd_int8_w;
	wire dispatch_is_vadd_bias_w;
	wire dispatch_is_scale_mul_int8_w;
	wire dispatch_is_scale_mul_int32_w;
	wire dispatch_is_requant_pc_w;
	wire dispatch_is_dequant_add_w;
	wire dispatch_same_buf_overlap_w;
	wire [15:0] flat_src_row_w;
	wire [15:0] flat_dst_row_w;
	wire [31:0] trans_src_byte_addr_w;
	wire [15:0] trans_src_row0_w;
	wire [15:0] trans_src_row1_w;
	wire [3:0] trans_src_lane_w;
	wire trans_need_row1_w;
	wire [15:0] trans_dst_row_w;
	wire [127:0] trans_dst_data_w;
	wire [127:0] trans_dst_merge_w;
	wire [15:0] v8_src1_row_w;
	wire [15:0] v8_src2_row_w;
	wire [15:0] v8_dst_row_w;
	wire [15:0] vbias_row_w;
	wire [15:0] vacc_row_w;
	wire [15:0] rq_src_row_w;
	wire [15:0] rq_dst_row_w;
	wire [15:0] rqpc_scale_row_w;
	wire [15:0] dq_skip_row_w;
	reg [127:0] trans_col_data_w;
	reg [127:0] trans_partial_row_w;
	wire [31:0] rq_src_row_full_w;
	wire [31:0] rq_dst_row_full_w;
	reg [127:0] rqpc_write_data_w;
	reg [127:0] dq_write_data_w;
	reg [127:0] scale_mul_write_data_w;
	assign dispatch_m_rows_w = ({5'h00, tile_m} + 15'd1) << 4;
	assign dispatch_n_tiles_w = {1'b0, tile_n} + 11'd1;
	assign dispatch_n_chunks_i32_w = dispatch_n_tiles_w << 2;
	assign dispatch_int8_units_w = dispatch_m_rows_w * dispatch_n_tiles_w;
	assign dispatch_int32_units_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
	assign dispatch_scale_rows_w = {20'h00000, dispatch_n_tiles_w, 1'b0};
	assign dispatch_copy_units_w = {16'h0000, b_length};
	assign dispatch_src_rows_w = {6'h00, b_src_rows, 4'h0};
	assign dispatch_trans_cols_w = (b_src_rows == 6'h00 ? 16'h0000 : b_length / {10'h000, b_src_rows});
	assign dispatch_src_buf_rows_w = buf_rows(src1_buf);
	assign dispatch_src2_buf_rows_w = buf_rows(src2_buf);
	assign dispatch_dst_buf_rows_w = buf_rows(dst_buf);
	assign dispatch_is_vadd_int8_w = ((src1_buf == 2'b00) && ((src2_buf == 2'b00) || (src2_buf == 2'b01))) && (dst_buf == 2'b00);
	assign dispatch_is_vadd_bias_w = ((src1_buf == 2'b10) && (src2_buf == 2'b01)) && (dst_buf == 2'b10);
	assign dispatch_is_scale_mul_int8_w = (src1_buf != 2'b10) && (dst_buf != 2'b10);
	assign dispatch_is_scale_mul_int32_w = (src1_buf == 2'b10) && (dst_buf == 2'b10);
	assign dispatch_is_requant_pc_w = ((src1_buf == 2'b10) && (src2_buf != 2'b10)) && (dst_buf != 2'b10);
	assign dispatch_is_dequant_add_w = ((src1_buf == 2'b10) && (src2_buf != 2'b10)) && (dst_buf != 2'b10);
	assign dispatch_same_buf_overlap_w = ((src1_buf == dst_buf) && ({16'h0000, src1_off} < ({16'h0000, dst_off} + dispatch_copy_units_w))) && ({16'h0000, dst_off} < ({16'h0000, src1_off} + dispatch_copy_units_w));
	always @(*) begin
		if (_sv2v_0)
			;
		dispatch_unsupported_w = 1'b0;
		dispatch_sram_oob_w = 1'b0;
		case (opcode)
			5'h09: begin
				dispatch_sram_oob_w = (({1'b0, src1_off} + {1'b0, b_length}) > {1'b0, dispatch_src_buf_rows_w}) || (({1'b0, dst_off} + {1'b0, b_length}) > {1'b0, dispatch_dst_buf_rows_w});
				if (b_transpose) begin
					if ((b_length != 16'h0000) && (((b_src_rows == 6'h00) || ((b_length % {10'h000, b_src_rows}) != 16'h0000)) || (src1_buf == dst_buf)))
						dispatch_unsupported_w = 1'b1;
				end
			end
			5'h0d:
				if (dispatch_is_vadd_int8_w)
					dispatch_sram_oob_w = ((({16'h0000, src1_off} + dispatch_int8_units_w) > {16'h0000, dispatch_src_buf_rows_w}) || (({16'h0000, src2_off} + dispatch_int8_units_w) > {16'h0000, dispatch_src2_buf_rows_w})) || (({16'h0000, dst_off} + dispatch_int8_units_w) > {16'h0000, dispatch_dst_buf_rows_w});
				else if (dispatch_is_vadd_bias_w)
					dispatch_sram_oob_w = ((({16'h0000, src1_off} + dispatch_int32_units_w) > {16'h0000, dispatch_src_buf_rows_w}) || (({16'h0000, src2_off} + {19'h00000, dispatch_n_chunks_i32_w}) > {16'h0000, dispatch_src2_buf_rows_w})) || (({16'h0000, dst_off} + dispatch_int32_units_w) > {16'h0000, dispatch_dst_buf_rows_w});
				else
					dispatch_unsupported_w = 1'b1;
			5'h0b:
				if ((src1_buf != 2'b10) || (dst_buf == 2'b10))
					dispatch_unsupported_w = 1'b1;
				else
					dispatch_sram_oob_w = (({16'h0000, src1_off} + dispatch_int32_units_w) > {16'h0000, dispatch_src_buf_rows_w}) || (({16'h0000, dst_off} + dispatch_int8_units_w) > {16'h0000, dispatch_dst_buf_rows_w});
			5'h11:
				if (!dispatch_is_requant_pc_w)
					dispatch_unsupported_w = 1'b1;
				else
					dispatch_sram_oob_w = ((({16'h0000, src1_off} + dispatch_int32_units_w) > {16'h0000, dispatch_src_buf_rows_w}) || (({16'h0000, src2_off} + dispatch_scale_rows_w) > {16'h0000, dispatch_src2_buf_rows_w})) || (({16'h0000, dst_off} + dispatch_int8_units_w) > {16'h0000, dispatch_dst_buf_rows_w});
			5'h0c:
				if (dispatch_is_scale_mul_int32_w)
					dispatch_sram_oob_w = (({16'h0000, src1_off} + dispatch_int32_units_w) > {16'h0000, dispatch_src_buf_rows_w}) || (({16'h0000, dst_off} + dispatch_int32_units_w) > {16'h0000, dispatch_dst_buf_rows_w});
				else if (dispatch_is_scale_mul_int8_w)
					dispatch_sram_oob_w = (({16'h0000, src1_off} + dispatch_int8_units_w) > {16'h0000, dispatch_src_buf_rows_w}) || (({16'h0000, dst_off} + dispatch_int8_units_w) > {16'h0000, dispatch_dst_buf_rows_w});
				else
					dispatch_unsupported_w = 1'b1;
			5'h13:
				if (!dispatch_is_dequant_add_w || (sreg == 4'hf))
					dispatch_unsupported_w = 1'b1;
				else
					dispatch_sram_oob_w = ((({16'h0000, src1_off} + dispatch_int32_units_w) > {16'h0000, dispatch_src_buf_rows_w}) || (({16'h0000, src2_off} + dispatch_int8_units_w) > {16'h0000, dispatch_src2_buf_rows_w})) || (({16'h0000, dst_off} + dispatch_int8_units_w) > {16'h0000, dispatch_dst_buf_rows_w});
			default: dispatch_unsupported_w = 1'b1;
		endcase
	end
	function automatic [15:0] sv2v_cast_16;
		input reg [15:0] inp;
		sv2v_cast_16 = inp;
	endfunction
	assign flat_src_row_w = (flat_backward_q ? ((src1_off_q + b_length_q) - sv2v_cast_16(step_idx_q)) - 16'h0001 : src1_off_q + sv2v_cast_16(step_idx_q));
	assign flat_dst_row_w = (flat_backward_q ? ((dst_off_q + b_length_q) - sv2v_cast_16(step_idx_q)) - 16'h0001 : dst_off_q + sv2v_cast_16(step_idx_q));
	assign trans_src_byte_addr_w = (({16'h0000, src1_off_q} << 4) + (({16'h0000, trans_rbase_q} + {27'h0000000, trans_src_row_idx_q}) * {16'h0000, trans_cols_q})) + {16'h0000, trans_cbase_q};
	assign trans_src_row0_w = trans_src_byte_addr_w[19:4];
	assign trans_src_row1_w = trans_src_row0_w + 16'h0001;
	assign trans_src_lane_w = trans_src_byte_addr_w[3:0];
	assign trans_need_row1_w = ({1'b0, trans_src_lane_w} + {1'b0, trans_width_q}) > 6'd16;
	assign trans_dst_row_w = (dst_off_q + ((trans_cbase_q + {11'h000, trans_dst_row_idx_q}) * {10'h000, b_src_rows_q})) + (trans_rbase_q >> 4);
	assign v8_src1_row_w = src1_off_q + sv2v_cast_16(step_idx_q);
	assign v8_src2_row_w = src2_off_q + sv2v_cast_16(step_idx_q);
	assign v8_dst_row_w = dst_off_q + sv2v_cast_16(step_idx_q);
	assign vbias_row_w = src2_off_q + bias_chunk_q;
	assign vacc_row_w = (src1_off_q + (bias_row_idx_q * n_chunks_i32_q)) + bias_chunk_q;
	assign rq_src_row_full_w = (({16'h0000, src1_off_q} + ({17'h00000, rq_row_idx_q} * {19'h00000, n_chunks_i32_q})) + ({19'h00000, rq_col_chunk_q} << 2)) + {30'h00000000, rq_part_q};
	assign rq_dst_row_full_w = ({16'h0000, dst_off_q} + ({17'h00000, rq_row_idx_q} * {21'h000000, n_tiles_q})) + {21'h000000, rq_col_chunk_q};
	assign rq_src_row_w = rq_src_row_full_w[15:0];
	assign rq_dst_row_w = rq_dst_row_full_w[15:0];
	assign rqpc_scale_row_w = (src2_off_q + ({4'h0, rq_col_chunk_q} << 1)) + {15'h0000, rq_part_q[0]};
	assign dq_skip_row_w = (src2_off_q + (rq_row_idx_q * n_tiles_q)) + {5'h00, rq_col_chunk_q};
	always @(*) begin
		if (_sv2v_0)
			;
		trans_col_data_w = 128'h00000000000000000000000000000000;
		begin : sv2v_autoblock_1
			reg signed [31:0] j;
			for (j = 0; j < 16; j = j + 1)
				if (j < sv2v_cast_32_signed(trans_height_q))
					trans_col_data_w[j * 8+:8] = trans_scratch_q[j[3:0]][sv2v_cast_32_signed(trans_dst_row_idx_q) * 8+:8];
		end
		trans_partial_row_w = sram_a_rdata;
		begin : sv2v_autoblock_2
			reg signed [31:0] j;
			for (j = 0; j < 16; j = j + 1)
				if (j < sv2v_cast_32_signed(trans_height_q))
					trans_partial_row_w[j * 8+:8] = trans_col_data_w[j * 8+:8];
		end
	end
	assign trans_dst_data_w = trans_col_data_w;
	assign trans_dst_merge_w = trans_partial_row_w;
	wire [31:0] synth_dq_acc_scale_bits;
	wire [31:0] synth_dq_skip_scale_bits;
	fp16_to_fp32 u_synth_dq_h2f_acc(
		.a(scale0_q),
		.y(synth_dq_acc_scale_bits)
	);
	fp16_to_fp32 u_synth_dq_h2f_skip(
		.a(scale1_q),
		.y(synth_dq_skip_scale_bits)
	);
	wire [127:0] synth_dq_write_data_w;
	genvar _gv_g_lane_1;
	generate
		for (_gv_g_lane_1 = 0; _gv_g_lane_1 < 16; _gv_g_lane_1 = _gv_g_lane_1 + 1) begin : g_synth_dq
			localparam g_lane = _gv_g_lane_1;
			reg signed [31:0] src_i32_l;
			reg signed [7:0] skip_i8_l;
			reg signed [31:0] skip_i32_l;
			wire [31:0] src_fp32_l;
			wire [31:0] skip_fp32_l;
			wire [31:0] acc_term_l;
			wire [31:0] skip_term_l;
			wire [31:0] sum_l;
			wire signed [7:0] q_i8_l;
			always @(*) begin
				if (_sv2v_0)
					;
				case (g_lane[3:2])
					2'd0: src_i32_l = rq_row0_q[g_lane[1:0] * 32+:32];
					2'd1: src_i32_l = rq_row1_q[g_lane[1:0] * 32+:32];
					2'd2: src_i32_l = rq_row2_q[g_lane[1:0] * 32+:32];
					default: src_i32_l = rq_row3_q[g_lane[1:0] * 32+:32];
				endcase
				skip_i8_l = skip_row_q[g_lane * 8+:8];
				skip_i32_l = {{24 {skip_i8_l[7]}}, skip_i8_l};
			end
			i32_to_fp32 u_src(
				.a(src_i32_l),
				.y(src_fp32_l)
			);
			i32_to_fp32 u_skp(
				.a(skip_i32_l),
				.y(skip_fp32_l)
			);
			fp32_mul u_acc_mul(
				.a(src_fp32_l),
				.b(synth_dq_acc_scale_bits),
				.y(acc_term_l)
			);
			fp32_mul u_skp_mul(
				.a(skip_fp32_l),
				.b(synth_dq_skip_scale_bits),
				.y(skip_term_l)
			);
			fp32_add u_dq_add(
				.a(acc_term_l),
				.b(skip_term_l),
				.y(sum_l)
			);
			fp32_quantize_i8 u_dq_q(
				.a(sum_l),
				.y(q_i8_l)
			);
			assign synth_dq_write_data_w[g_lane * 8+:8] = q_i8_l;
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		rqpc_write_data_w = 128'h00000000000000000000000000000000;
		dq_write_data_w = 128'h00000000000000000000000000000000;
		scale_mul_write_data_w = 128'h00000000000000000000000000000000;
		begin : sv2v_autoblock_3
			reg signed [31:0] lane;
			for (lane = 0; lane < 16; lane = lane + 1)
				begin : sv2v_autoblock_4
					reg signed [31:0] src_i32;
					reg signed [63:0] scaled;
					case (lane[3:2])
						2'd0: src_i32 = rq_row0_q[lane[1:0] * 32+:32];
						2'd1: src_i32 = rq_row1_q[lane[1:0] * 32+:32];
						2'd2: src_i32 = rq_row2_q[lane[1:0] * 32+:32];
						default: src_i32 = rq_row3_q[lane[1:0] * 32+:32];
					endcase
					scaled = fp16_mul_round_even(src_i32, pc_scale_chunk_q[lane]);
					if (scaled > 64'sd127)
						rqpc_write_data_w[lane * 8+:8] = 8'h7f;
					else if (scaled < -64'sd128)
						rqpc_write_data_w[lane * 8+:8] = 8'h80;
					else
						rqpc_write_data_w[lane * 8+:8] = scaled[7:0];
				end
		end
		if (HELPER_SYNTH_MODE == 1)
			dq_write_data_w = synth_dq_write_data_w;
		if (src1_buf_q == 2'b10)
			scale_mul_write_data_w = scale_mul_i32_row(sram_b_rdata, scale0_q);
		else
			scale_mul_write_data_w = scale_mul_i8_row(sram_b_rdata, scale0_q);
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 6'd0;
			opcode_q <= 5'h00;
			src1_buf_q <= 2'b00;
			src2_buf_q <= 2'b00;
			dst_buf_q <= 2'b00;
			src1_off_q <= 16'h0000;
			src2_off_q <= 16'h0000;
			dst_off_q <= 16'h0000;
			sreg_q <= 4'h0;
			b_length_q <= 16'h0000;
			b_src_rows_q <= 6'h00;
			b_transpose_q <= 1'b0;
			scale0_q <= 16'h0000;
			scale1_q <= 16'h0000;
			m_rows_q <= 15'h0000;
			n_tiles_q <= 11'h000;
			n_chunks_i32_q <= 13'h0000;
			fault_code_r <= 4'h0;
			step_idx_q <= 32'h00000000;
			flat_backward_q <= 1'b0;
			trans_row_count_q <= 16'h0000;
			trans_cols_q <= 16'h0000;
			trans_rbase_q <= 16'h0000;
			trans_cbase_q <= 16'h0000;
			trans_height_q <= 5'h00;
			trans_width_q <= 5'h00;
			trans_src_row_idx_q <= 5'h00;
			trans_dst_row_idx_q <= 5'h00;
			trans_first_row_q <= 128'h00000000000000000000000000000000;
			bias_chunk_q <= 16'h0000;
			bias_row_idx_q <= 15'h0000;
			bias_data_q <= 128'h00000000000000000000000000000000;
			rq_row_idx_q <= 15'h0000;
			rq_col_chunk_q <= 11'h000;
			rq_part_q <= 2'h0;
			rq_row0_q <= 128'h00000000000000000000000000000000;
			rq_row1_q <= 128'h00000000000000000000000000000000;
			rq_row2_q <= 128'h00000000000000000000000000000000;
			rq_row3_q <= 128'h00000000000000000000000000000000;
			skip_row_q <= 128'h00000000000000000000000000000000;
			begin : sv2v_autoblock_5
				reg signed [31:0] i;
				for (i = 0; i < 16; i = i + 1)
					pc_scale_chunk_q[i] <= 16'h0000;
			end
			begin : sv2v_autoblock_6
				reg signed [31:0] j;
				for (j = 0; j < 16; j = j + 1)
					trans_scratch_q[j] <= 128'h00000000000000000000000000000000;
			end
		end
		else
			case (state)
				6'd0:
					if (dispatch) begin
						opcode_q <= opcode;
						src1_buf_q <= src1_buf;
						src2_buf_q <= src2_buf;
						dst_buf_q <= dst_buf;
						src1_off_q <= src1_off;
						src2_off_q <= src2_off;
						dst_off_q <= dst_off;
						sreg_q <= sreg;
						b_length_q <= b_length;
						b_src_rows_q <= b_src_rows;
						b_transpose_q <= b_transpose;
						scale0_q <= scale0_data;
						scale1_q <= scale1_data;
						m_rows_q <= dispatch_m_rows_w;
						n_tiles_q <= dispatch_n_tiles_w;
						n_chunks_i32_q <= dispatch_n_chunks_i32_w;
						if (dispatch_unsupported_w) begin
							fault_code_r <= 4'h6;
							state <= 6'd32;
						end
						else if (dispatch_sram_oob_w) begin
							fault_code_r <= 4'h3;
							state <= 6'd32;
						end
						else
							case (opcode)
								5'h09:
									if (b_length == 16'h0000)
										state <= 6'd0;
									else if (b_transpose) begin
										trans_row_count_q <= dispatch_src_rows_w;
										trans_cols_q <= dispatch_trans_cols_w;
										trans_rbase_q <= 16'h0000;
										trans_cbase_q <= 16'h0000;
										trans_height_q <= block_span(dispatch_src_rows_w, 16'h0000);
										trans_width_q <= block_span(dispatch_trans_cols_w, 16'h0000);
										trans_src_row_idx_q <= 5'h00;
										trans_dst_row_idx_q <= 5'h00;
										state <= 6'd3;
									end
									else begin
										step_idx_q <= 32'h00000000;
										flat_backward_q <= dispatch_same_buf_overlap_w && (dst_off > src1_off);
										state <= 6'd1;
									end
								5'h0d:
									if (dispatch_is_vadd_int8_w) begin
										step_idx_q <= 32'h00000000;
										state <= 6'd9;
									end
									else begin
										bias_chunk_q <= 16'h0000;
										bias_row_idx_q <= 15'h0000;
										state <= 6'd11;
									end
								5'h0b: begin
									rq_row_idx_q <= 15'h0000;
									rq_col_chunk_q <= 11'h000;
									rq_part_q <= 2'h0;
									state <= 6'd15;
								end
								5'h11: begin
									rq_row_idx_q <= 15'h0000;
									rq_col_chunk_q <= 11'h000;
									rq_part_q <= 2'h0;
									state <= 6'd20;
								end
								5'h0c: begin
									step_idx_q <= 32'h00000000;
									state <= 6'd18;
								end
								5'h13: begin
									rq_row_idx_q <= 15'h0000;
									rq_col_chunk_q <= 11'h000;
									rq_part_q <= 2'h0;
									state <= 6'd27;
								end
								default: begin
									fault_code_r <= 4'h6;
									state <= 6'd32;
								end
							endcase
					end
				6'd1:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd2;
				6'd2:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else if ((step_idx_q + 32'd1) >= {16'h0000, b_length_q})
						state <= 6'd0;
					else begin
						step_idx_q <= step_idx_q + 32'd1;
						state <= 6'd1;
					end
				6'd3:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd4;
				6'd4:
					if (trans_need_row1_w) begin
						trans_first_row_q <= sram_a_rdata;
						state <= 6'd5;
					end
					else begin
						trans_scratch_q[trans_src_row_idx_q[3:0]] <= extract_window(sram_a_rdata, 128'h00000000000000000000000000000000, trans_src_lane_w, trans_width_q);
						if ((trans_src_row_idx_q + 5'd1) >= trans_height_q) begin
							trans_dst_row_idx_q <= 5'h00;
							if (trans_height_q == 5'd16)
								state <= 6'd8;
							else
								state <= 6'd7;
						end
						else begin
							trans_src_row_idx_q <= trans_src_row_idx_q + 5'd1;
							state <= 6'd3;
						end
					end
				6'd5:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd6;
				6'd6: begin
					trans_scratch_q[trans_src_row_idx_q[3:0]] <= extract_window(trans_first_row_q, sram_a_rdata, trans_src_lane_w, trans_width_q);
					if ((trans_src_row_idx_q + 5'd1) >= trans_height_q) begin
						trans_dst_row_idx_q <= 5'h00;
						if (trans_height_q == 5'd16)
							state <= 6'd8;
						else
							state <= 6'd7;
					end
					else begin
						trans_src_row_idx_q <= trans_src_row_idx_q + 5'd1;
						state <= 6'd3;
					end
				end
				6'd7:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd8;
				6'd8:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else if ((trans_dst_row_idx_q + 5'd1) < trans_width_q) begin
						trans_dst_row_idx_q <= trans_dst_row_idx_q + 5'd1;
						if (trans_height_q == 5'd16)
							state <= 6'd8;
						else
							state <= 6'd7;
					end
					else if (({16'h0000, trans_cbase_q} + {27'h0000000, trans_width_q}) < {16'h0000, trans_cols_q}) begin
						trans_cbase_q <= trans_cbase_q + 16'd16;
						trans_width_q <= block_span(trans_cols_q, trans_cbase_q + 16'd16);
						trans_src_row_idx_q <= 5'h00;
						state <= 6'd3;
					end
					else if (({16'h0000, trans_rbase_q} + {27'h0000000, trans_height_q}) < {16'h0000, trans_row_count_q}) begin
						trans_rbase_q <= trans_rbase_q + 16'd16;
						trans_cbase_q <= 16'h0000;
						trans_height_q <= block_span(trans_row_count_q, trans_rbase_q + 16'd16);
						trans_width_q <= block_span(trans_cols_q, 16'h0000);
						trans_src_row_idx_q <= 5'h00;
						state <= 6'd3;
					end
					else
						state <= 6'd0;
				6'd9:
					if (sram_a_fault || sram_b_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd10;
				6'd10:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else if ((step_idx_q + 32'd1) >= (m_rows_q * n_tiles_q))
						state <= 6'd0;
					else begin
						step_idx_q <= step_idx_q + 32'd1;
						state <= 6'd9;
					end
				6'd11:
					if (sram_b_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd12;
				6'd12: begin
					bias_data_q <= sram_b_rdata;
					bias_row_idx_q <= 15'h0000;
					state <= 6'd13;
				end
				6'd13:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd14;
				6'd14:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else if ((bias_row_idx_q + 15'd1) < m_rows_q) begin
						bias_row_idx_q <= bias_row_idx_q + 15'd1;
						state <= 6'd13;
					end
					else if (({16'h0000, bias_chunk_q} + 32'd1) < {19'h00000, n_chunks_i32_q}) begin
						bias_chunk_q <= bias_chunk_q + 16'd1;
						state <= 6'd11;
					end
					else
						state <= 6'd0;
				6'd15:
					if (sram_b_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd16;
				6'd16: begin
					case (rq_part_q)
						2'd0: rq_row0_q <= sram_b_rdata;
						2'd1: rq_row1_q <= sram_b_rdata;
						2'd2: rq_row2_q <= sram_b_rdata;
						default: rq_row3_q <= sram_b_rdata;
					endcase
					if (rq_part_q == 2'd3)
						state <= 6'd17;
					else begin
						rq_part_q <= rq_part_q + 2'd1;
						state <= 6'd15;
					end
				end
				6'd17:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else if ((rq_col_chunk_q + 11'd1) < n_tiles_q) begin
						rq_col_chunk_q <= rq_col_chunk_q + 11'd1;
						rq_part_q <= 2'd0;
						state <= 6'd15;
					end
					else if ((rq_row_idx_q + 15'd1) < m_rows_q) begin
						rq_row_idx_q <= rq_row_idx_q + 15'd1;
						rq_col_chunk_q <= 11'd0;
						rq_part_q <= 2'd0;
						state <= 6'd15;
					end
					else
						state <= 6'd0;
				6'd18:
					if (sram_b_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd19;
				6'd19: begin : sv2v_autoblock_7
					reg [31:0] total_rows_w;
					total_rows_w = (src1_buf_q == 2'b10 ? m_rows_q * n_chunks_i32_q : m_rows_q * n_tiles_q);
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else if ((step_idx_q + 32'd1) >= total_rows_w)
						state <= 6'd0;
					else begin
						step_idx_q <= step_idx_q + 32'd1;
						state <= 6'd18;
					end
				end
				6'd20:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd21;
				6'd21: begin
					begin : sv2v_autoblock_8
						reg signed [31:0] lane;
						for (lane = 0; lane < 8; lane = lane + 1)
							pc_scale_chunk_q[lane] <= get_u16(sram_a_rdata, lane);
					end
					rq_part_q <= 2'd1;
					state <= 6'd22;
				end
				6'd22:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd23;
				6'd23: begin
					begin : sv2v_autoblock_9
						reg signed [31:0] lane;
						for (lane = 0; lane < 8; lane = lane + 1)
							pc_scale_chunk_q[lane + 8] <= get_u16(sram_a_rdata, lane);
					end
					rq_row_idx_q <= 15'h0000;
					rq_part_q <= 2'd0;
					state <= 6'd24;
				end
				6'd24:
					if (sram_b_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd25;
				6'd25: begin
					case (rq_part_q)
						2'd0: rq_row0_q <= sram_b_rdata;
						2'd1: rq_row1_q <= sram_b_rdata;
						2'd2: rq_row2_q <= sram_b_rdata;
						default: rq_row3_q <= sram_b_rdata;
					endcase
					if (rq_part_q == 2'd3)
						state <= 6'd26;
					else begin
						rq_part_q <= rq_part_q + 2'd1;
						state <= 6'd24;
					end
				end
				6'd26:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else if ((rq_row_idx_q + 15'd1) < m_rows_q) begin
						rq_row_idx_q <= rq_row_idx_q + 15'd1;
						rq_part_q <= 2'd0;
						state <= 6'd24;
					end
					else if ((rq_col_chunk_q + 11'd1) < n_tiles_q) begin
						rq_col_chunk_q <= rq_col_chunk_q + 11'd1;
						rq_part_q <= 2'd0;
						state <= 6'd20;
					end
					else
						state <= 6'd0;
				6'd27:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd28;
				6'd28: begin
					skip_row_q <= sram_a_rdata;
					rq_part_q <= 2'd0;
					state <= 6'd29;
				end
				6'd29:
					if (sram_b_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else
						state <= 6'd30;
				6'd30: begin
					case (rq_part_q)
						2'd0: rq_row0_q <= sram_b_rdata;
						2'd1: rq_row1_q <= sram_b_rdata;
						2'd2: rq_row2_q <= sram_b_rdata;
						default: rq_row3_q <= sram_b_rdata;
					endcase
					if (rq_part_q == 2'd3)
						state <= 6'd31;
					else begin
						rq_part_q <= rq_part_q + 2'd1;
						state <= 6'd29;
					end
				end
				6'd31:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd32;
					end
					else if ((rq_col_chunk_q + 11'd1) < n_tiles_q) begin
						rq_col_chunk_q <= rq_col_chunk_q + 11'd1;
						state <= 6'd27;
					end
					else if ((rq_row_idx_q + 15'd1) < m_rows_q) begin
						rq_row_idx_q <= rq_row_idx_q + 15'd1;
						rq_col_chunk_q <= 11'd0;
						state <= 6'd27;
					end
					else
						state <= 6'd0;
				6'd32:
					;
				default: state <= 6'd0;
			endcase
	always @(*) begin
		if (_sv2v_0)
			;
		helper_busy = (state != 6'd0) && (state != 6'd32);
		helper_fault = state == 6'd32;
		helper_fault_code = fault_code_r;
		sram_a_en = 1'b0;
		sram_a_we = 1'b0;
		sram_a_buf = src1_buf_q;
		sram_a_row = 16'h0000;
		sram_a_wdata = 128'h00000000000000000000000000000000;
		sram_b_en = 1'b0;
		sram_b_buf = src1_buf_q;
		sram_b_row = 16'h0000;
		case (state)
			6'd1: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = src1_buf_q;
				sram_a_row = flat_src_row_w;
			end
			6'd2: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = flat_dst_row_w;
				sram_a_wdata = sram_a_rdata;
			end
			6'd3: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = src1_buf_q;
				sram_a_row = trans_src_row0_w;
			end
			6'd5: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = src1_buf_q;
				sram_a_row = trans_src_row1_w;
			end
			6'd7: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = dst_buf_q;
				sram_a_row = trans_dst_row_w;
			end
			6'd8: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = trans_dst_row_w;
				if (trans_height_q == 5'd16)
					sram_a_wdata = trans_dst_data_w;
				else
					sram_a_wdata = trans_dst_merge_w;
			end
			6'd9: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = src2_buf_q;
				sram_a_row = v8_src2_row_w;
				sram_b_en = 1'b1;
				sram_b_buf = src1_buf_q;
				sram_b_row = v8_src1_row_w;
			end
			6'd10: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = v8_dst_row_w;
				sram_a_wdata = sat_add_int8_row(sram_b_rdata, sram_a_rdata);
			end
			6'd11: begin
				sram_b_en = 1'b1;
				sram_b_buf = src2_buf_q;
				sram_b_row = vbias_row_w;
			end
			6'd13: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = src1_buf_q;
				sram_a_row = vacc_row_w;
			end
			6'd14: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = (dst_off_q + (bias_row_idx_q * n_chunks_i32_q)) + bias_chunk_q;
				sram_a_wdata = add_wrap_int32_row(sram_a_rdata, bias_data_q);
			end
			6'd15: begin
				sram_b_en = 1'b1;
				sram_b_buf = src1_buf_q;
				sram_b_row = rq_src_row_w;
			end
			6'd17: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = rq_dst_row_w;
				sram_a_wdata = requant_pack(rq_row0_q, rq_row1_q, rq_row2_q, rq_row3_q, scale0_q);
			end
			6'd18: begin
				sram_b_en = 1'b1;
				sram_b_buf = src1_buf_q;
				sram_b_row = src1_off_q + sv2v_cast_16(step_idx_q);
			end
			6'd19: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = dst_off_q + sv2v_cast_16(step_idx_q);
				sram_a_wdata = scale_mul_write_data_w;
			end
			6'd20: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = src2_buf_q;
				sram_a_row = rqpc_scale_row_w;
			end
			6'd22: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = src2_buf_q;
				sram_a_row = rqpc_scale_row_w;
			end
			6'd24: begin
				sram_b_en = 1'b1;
				sram_b_buf = src1_buf_q;
				sram_b_row = rq_src_row_w;
			end
			6'd26: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = rq_dst_row_w;
				sram_a_wdata = rqpc_write_data_w;
			end
			6'd27: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = src2_buf_q;
				sram_a_row = dq_skip_row_w;
			end
			6'd29: begin
				sram_b_en = 1'b1;
				sram_b_buf = src1_buf_q;
				sram_b_row = rq_src_row_w;
			end
			6'd31: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = rq_dst_row_w;
				sram_a_wdata = dq_write_data_w;
			end
			default:
				;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
module sfu_engine (
	clk,
	rst_n,
	dispatch,
	opcode,
	src1_buf,
	src1_off,
	src2_buf,
	src2_off,
	dst_buf,
	dst_off,
	sreg,
	tile_m,
	tile_n,
	tile_k,
	attn_valid,
	attn_query_row_base,
	attn_valid_kv_len,
	attn_mode,
	scale0_data,
	scale1_data,
	scale2_data,
	scale3_data,
	sfu_busy,
	sfu_fault,
	sfu_fault_code,
	sram_a_en,
	sram_a_we,
	sram_a_buf,
	sram_a_row,
	sram_a_wdata,
	sram_a_fault,
	sram_b_en,
	sram_b_buf,
	sram_b_row,
	sram_b_rdata,
	sram_b_fault,
	sfu_scale_we,
	sfu_scale_waddr,
	sfu_scale_wdata
);
	reg _sv2v_0;
	parameter signed [31:0] SFU_SYNTH_MODE = 0;
	input wire clk;
	input wire rst_n;
	input wire dispatch;
	input wire [4:0] opcode;
	input wire [1:0] src1_buf;
	input wire [15:0] src1_off;
	input wire [1:0] src2_buf;
	input wire [15:0] src2_off;
	input wire [1:0] dst_buf;
	input wire [15:0] dst_off;
	input wire [3:0] sreg;
	input wire [9:0] tile_m;
	input wire [9:0] tile_n;
	input wire [9:0] tile_k;
	input wire attn_valid;
	input wire [11:0] attn_query_row_base;
	input wire [11:0] attn_valid_kv_len;
	input wire [1:0] attn_mode;
	input wire [15:0] scale0_data;
	input wire [15:0] scale1_data;
	input wire [15:0] scale2_data;
	input wire [15:0] scale3_data;
	output reg sfu_busy;
	output reg sfu_fault;
	output reg [3:0] sfu_fault_code;
	output reg sram_a_en;
	output reg sram_a_we;
	output reg [1:0] sram_a_buf;
	output reg [15:0] sram_a_row;
	output reg [127:0] sram_a_wdata;
	input wire sram_a_fault;
	output reg sram_b_en;
	output reg [1:0] sram_b_buf;
	output reg [15:0] sram_b_row;
	input wire [127:0] sram_b_rdata;
	input wire sram_b_fault;
	output reg sfu_scale_we;
	output reg [3:0] sfu_scale_waddr;
	output reg [15:0] sfu_scale_wdata;
	localparam signed [31:0] SFU_MAX_ROW_ELEMS = 1024;
	localparam real LN_EPS = 1.0e-6;
	localparam real LN_FP32_EPS = 1.0e-5;
	reg [5:0] state;
	reg [4:0] opcode_q;
	reg [1:0] src1_buf_q;
	reg [1:0] src2_buf_q;
	reg [1:0] dst_buf_q;
	reg [15:0] src1_off_q;
	reg [15:0] src2_off_q;
	reg [15:0] dst_off_q;
	reg [3:0] sreg_q;
	reg [14:0] m_rows_q;
	reg [10:0] n_tiles_q;
	reg [10:0] k_tiles_q;
	reg [12:0] n_chunks_i32_q;
	reg [12:0] k_chunks_i32_q;
	reg [15:0] n_elems_q;
	reg [15:0] k_elems_q;
	reg [15:0] ln_gamma_rows_q;
	reg [15:0] ln_param_rows_q;
	reg attn_valid_q;
	reg [11:0] attn_query_row_base_q;
	reg [11:0] attn_valid_kv_len_q;
	reg [1:0] attn_mode_q;
	reg [3:0] fault_code_r;
	reg [14:0] row_idx_q;
	reg [12:0] read_idx_q;
	reg [10:0] iter_idx_q;
	reg [31:0] ln_sum_acc_q;
	reg [31:0] ln_var_acc_q;
	reg [31:0] ln_mean_q;
	reg [31:0] ln_denom_q;
	reg [31:0] sm_row_max_q;
	reg [31:0] sm_exp_sum_q;
	reg sm_have_vis_q;
	reg signed [15:0] sm_keep_through_q;
	reg [10:0] write_chunk_q;
	reg [1:0] gelu_part_q;
	reg [15:0] attn_k_idx_q;
	reg [127:0] gelu_i8_row_q;
	reg [127:0] gelu_row0_q;
	reg [127:0] gelu_row1_q;
	reg [127:0] gelu_row2_q;
	reg [127:0] gelu_row3_q;
	reg [31:0] scale0_q;
	reg [31:0] scale1_q;
	reg [31:0] scale2_q;
	reg [31:0] scale3_q;
	reg [31:0] row_data_q [0:1023];
	reg [31:0] attn_accum_q [0:1023];
	reg [31:0] gamma_q [0:1023];
	reg [31:0] beta_q [0:1023];
	reg [7:0] out_bytes_q [0:1023];
	reg [15:0] out_h_q [0:1023];
	reg [12:0] g2_rows_q;
	reg [31:0] g2_maxabs_q;
	reg g2_wr_phase_q;
	reg [31:0] attn_row_max_q;
	reg [31:0] attn_exp_sum_q;
	wire [14:0] dispatch_m_rows_w;
	wire [10:0] dispatch_n_tiles_w;
	wire [10:0] dispatch_k_tiles_w;
	wire [12:0] dispatch_n_chunks_i32_w;
	wire [12:0] dispatch_k_chunks_i32_w;
	wire [15:0] dispatch_n_elems_w;
	wire [15:0] dispatch_k_elems_w;
	wire [15:0] dispatch_ln_gamma_rows_w;
	wire [15:0] dispatch_ln_param_rows_w;
	wire [15:0] dispatch_src1_rows_w;
	wire [15:0] dispatch_src2_rows_w;
	wire [15:0] dispatch_dst_rows_w;
	reg dispatch_attn_context_bad_w;
	reg dispatch_unsupported_w;
	reg dispatch_sram_oob_w;
	reg [31:0] dispatch_src1_need_rows_w;
	reg [31:0] dispatch_src2_need_rows_w;
	reg [31:0] dispatch_dst_need_rows_w;
	wire [15:0] dispatch_attn_key_cols_w;
	wire [31:0] row_i8_addr_w;
	wire [31:0] row_i32_addr_w;
	wire [31:0] row_dst_addr_w;
	wire [31:0] ln_param_addr_w;
	wire [31:0] gelu_i8_addr_w;
	wire [31:0] gelu_acc_addr_w;
	wire [31:0] gelu_dst_addr_w;
	wire [31:0] attn_qkt_addr_w;
	wire [31:0] attn_v_addr_w;
	reg [127:0] row_write_data_w;
	reg [127:0] row_write_q;
	reg [127:0] gelu_i8_write_data_w;
	reg [127:0] gelu_i32_write_data_w;
	reg [127:0] attn_write_data_w;
	reg [127:0] g2_write_data_w;
	localparam signed [31:0] taccel_pkg_ABUF_ROWS = 8192;
	localparam signed [31:0] taccel_pkg_ACCUM_ROWS = 4096;
	localparam signed [31:0] taccel_pkg_WBUF_ROWS = 16384;
	function automatic signed [15:0] sv2v_cast_16_signed;
		input reg signed [15:0] inp;
		sv2v_cast_16_signed = inp;
	endfunction
	function automatic [15:0] buf_rows;
		input reg [1:0] bid;
		case (bid)
			2'b00: buf_rows = sv2v_cast_16_signed(taccel_pkg_ABUF_ROWS);
			2'b01: buf_rows = sv2v_cast_16_signed(taccel_pkg_WBUF_ROWS);
			2'b10: buf_rows = sv2v_cast_16_signed(taccel_pkg_ACCUM_ROWS);
			default: buf_rows = 16'h0000;
		endcase
	endfunction
	function automatic signed [7:0] get_i8;
		input reg [127:0] row;
		input integer idx;
		get_i8 = row[idx * 8+:8];
	endfunction
	function automatic signed [31:0] get_i32;
		input reg [127:0] row;
		input integer idx;
		get_i32 = row[idx * 32+:32];
	endfunction
	function automatic [15:0] get_u16;
		input reg [127:0] row;
		input integer idx;
		get_u16 = row[idx * 16+:16];
	endfunction
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	function automatic attn_visible;
		input reg [14:0] row_idx;
		input integer col_idx;
		integer abs_query_row;
		begin
			abs_query_row = sv2v_cast_32_signed(attn_query_row_base_q) + sv2v_cast_32_signed(row_idx);
			attn_visible = 1'b1;
			if (attn_mode_q[1])
				attn_visible = attn_visible && (col_idx <= abs_query_row);
			if (attn_mode_q[0])
				attn_visible = attn_visible && (col_idx < sv2v_cast_32_signed(attn_valid_kv_len_q));
		end
	endfunction
	assign dispatch_m_rows_w = ({5'h00, tile_m} + 15'd1) << 4;
	assign dispatch_n_tiles_w = {1'b0, tile_n} + 11'd1;
	assign dispatch_k_tiles_w = {1'b0, tile_k} + 11'd1;
	assign dispatch_n_chunks_i32_w = dispatch_n_tiles_w << 2;
	assign dispatch_k_chunks_i32_w = dispatch_k_tiles_w << 2;
	assign dispatch_n_elems_w = {1'b0, dispatch_n_tiles_w, 4'h0};
	assign dispatch_k_elems_w = {1'b0, dispatch_k_tiles_w, 4'h0};
	assign dispatch_ln_gamma_rows_w = {5'h00, dispatch_n_tiles_w} << 1;
	assign dispatch_ln_param_rows_w = {5'h00, dispatch_n_tiles_w} << 2;
	assign dispatch_src1_rows_w = buf_rows(src1_buf);
	assign dispatch_src2_rows_w = buf_rows(src2_buf);
	assign dispatch_dst_rows_w = buf_rows(dst_buf);
	assign dispatch_attn_key_cols_w = dispatch_n_elems_w;
	wire dispatch_g2_vadd_w;
	wire dispatch_g2_ln_w;
	wire dispatch_g2_gelu_w;
	wire [12:0] dispatch_g2_rows_w;
	assign dispatch_g2_rows_w = {1'b0, dispatch_n_tiles_w} + {1'b0, dispatch_n_tiles_w};
	assign dispatch_g2_vadd_w = (((opcode == 5'h19) && (src1_buf == 2'b00)) && (src2_buf == 2'b00)) && (dst_buf == 2'b00);
	assign dispatch_g2_ln_w = (((opcode == 5'h1a) && (src1_buf == 2'b00)) && (src2_buf == 2'b01)) && (dst_buf == 2'b00);
	assign dispatch_g2_gelu_w = ((opcode == 5'h1b) && (src1_buf == 2'b00)) && (dst_buf == 2'b00);
	wire dispatch_g2_dq_w;
	wire dispatch_g2_q_w;
	assign dispatch_g2_dq_w = (((opcode == 5'h17) && (src1_buf == 2'b10)) && (src2_buf == 2'b01)) && (dst_buf == 2'b00);
	assign dispatch_g2_q_w = ((opcode == 5'h18) && (src1_buf == 2'b00)) && (dst_buf == 2'b00);
	wire dispatch_g2_ms_w;
	assign dispatch_g2_ms_w = ((opcode == 5'h1d) && (src1_buf == 2'b00)) && (dst_buf == 2'b00);
	wire dispatch_g2_ds_w;
	wire dispatch_g2_mar_w;
	assign dispatch_g2_ds_w = (((opcode == 5'h1e) && (src1_buf == 2'b10)) && (src2_buf == 2'b01)) && (dst_buf == 2'b00);
	assign dispatch_g2_mar_w = ((opcode == 5'h1f) && (src1_buf == 2'b00)) && (sreg <= 4'd14);
	wire [31:0] synth_a_bits;
	wire [31:0] synth_b_bits;
	assign synth_a_bits = row_data_q[iter_idx_q[9:0]];
	assign synth_b_bits = attn_accum_q[iter_idx_q[9:0]];
	reg [31:0] synth_b_bits_eff;
	always @(*) begin
		if (_sv2v_0)
			;
		case (opcode_q)
			5'h18: synth_b_bits_eff = scale0_q;
			default: synth_b_bits_eff = synth_b_bits;
		endcase
	end
	wire [31:0] synth_add_out;
	wire [31:0] synth_mul_out;
	fp32_add u_synth_add(
		.a(synth_a_bits),
		.b(synth_b_bits),
		.y(synth_add_out)
	);
	fp32_mul u_synth_mul(
		.a(synth_a_bits),
		.b(synth_b_bits_eff),
		.y(synth_mul_out)
	);
	reg [31:0] synth_compute_out;
	wire [15:0] synth_out_bits;
	wire [31:0] synth_gelu_out;
	wire [31:0] synth_scaled_add;
	always @(*) begin
		if (_sv2v_0)
			;
		case (opcode_q)
			5'h19: synth_compute_out = synth_add_out;
			5'h17: synth_compute_out = synth_mul_out;
			5'h1e: synth_compute_out = synth_scaled_add;
			5'h1b: synth_compute_out = synth_gelu_out;
			default: synth_compute_out = 32'd0;
		endcase
	end
	fp32_to_fp16 u_synth_f2h(
		.a(synth_compute_out),
		.y(synth_out_bits)
	);
	wire signed [7:0] synth_quant_out;
	fp32_quantize_i8 u_synth_quant(
		.a(synth_mul_out),
		.y(synth_quant_out)
	);
	fp32_gelu_new u_synth_gelu(
		.a(synth_a_bits),
		.y(synth_gelu_out)
	);
	wire [31:0] synth_gamma_bits;
	wire [31:0] synth_beta_bits;
	wire [31:0] synth_scale0_bits;
	wire [31:0] synth_scaled_mul1;
	wire [31:0] synth_scaled_mul2;
	assign synth_gamma_bits = gamma_q[iter_idx_q[9:0]];
	assign synth_beta_bits = beta_q[iter_idx_q[9:0]];
	assign synth_scale0_bits = scale0_q;
	fp32_mul u_synth_scaled_mul1(
		.a(synth_a_bits),
		.b(synth_gamma_bits),
		.y(synth_scaled_mul1)
	);
	fp32_mul u_synth_scaled_mul2(
		.a(synth_scaled_mul1),
		.b(synth_scale0_bits),
		.y(synth_scaled_mul2)
	);
	fp32_add u_synth_scaled_add(
		.a(synth_scaled_mul2),
		.b(synth_beta_bits),
		.y(synth_scaled_add)
	);
	localparam [31:0] C_127_FP32 = 32'h42fe0000;
	localparam [31:0] C_CLAMP_MIN_FP32 = 32'h3b000000;
	localparam [31:0] C_CLAMP_MAX_FP32 = 32'h4a7ddc00;
	wire [31:0] synth_clamp_eps_bits;
	wire [31:0] synth_inv_eps;
	wire [31:0] synth_eps_inv127;
	wire [15:0] synth_inv_eps_fp16;
	wire [15:0] synth_eps_inv127_fp16;
	assign synth_clamp_eps_bits = (g2_maxabs_q < C_CLAMP_MIN_FP32 ? C_CLAMP_MIN_FP32 : (g2_maxabs_q > C_CLAMP_MAX_FP32 ? C_CLAMP_MAX_FP32 : g2_maxabs_q));
	fp32_div u_synth_inv_eps(
		.a(C_127_FP32),
		.b(synth_clamp_eps_bits),
		.y(synth_inv_eps)
	);
	fp32_div u_synth_eps_inv127(
		.a(synth_clamp_eps_bits),
		.b(C_127_FP32),
		.y(synth_eps_inv127)
	);
	fp32_to_fp16 u_synth_inv_eps_h(
		.a(synth_inv_eps),
		.y(synth_inv_eps_fp16)
	);
	fp32_to_fp16 u_synth_eps_inv127_h(
		.a(synth_eps_inv127),
		.y(synth_eps_inv127_fp16)
	);
	localparam [31:0] C_LN_FP32_EPS = 32'h3727c5ac;
	localparam [31:0] C_LN_EPS_G1 = 32'h358d3f3f;
	wire [31:0] ln_neg_mean;
	wire [31:0] ln_n_fp32;
	wire [31:0] ln_sum_add_w;
	wire [31:0] ln_mean_div_w;
	wire [31:0] ln_diff_w;
	wire [31:0] ln_diff_sq_w;
	wire [31:0] ln_var_add_w;
	wire [31:0] ln_var_norm_w;
	wire [31:0] ln_var_eps_w;
	wire [31:0] ln_eps_sel_w;
	wire [31:0] ln_denom_w;
	wire [31:0] ln_norm_w;
	wire [31:0] ln_norm_g_w;
	wire [31:0] ln_norm_gb_w;
	wire [15:0] ln_out_h_w;
	wire [31:0] ln_g1_scaled_w;
	wire signed [7:0] ln_g1_quant_w;
	assign ln_neg_mean = ln_mean_q ^ 32'h80000000;
	i32_to_fp32 u_ln_n_cvt(
		.a({16'h0000, n_elems_q}),
		.y(ln_n_fp32)
	);
	assign ln_eps_sel_w = (opcode_q == 5'h0f ? C_LN_EPS_G1 : C_LN_FP32_EPS);
	fp32_add u_ln_sum_add(
		.a(ln_sum_acc_q),
		.b(synth_a_bits),
		.y(ln_sum_add_w)
	);
	fp32_div u_ln_mean(
		.a(ln_sum_acc_q),
		.b(ln_n_fp32),
		.y(ln_mean_div_w)
	);
	fp32_add u_ln_diff(
		.a(synth_a_bits),
		.b(ln_neg_mean),
		.y(ln_diff_w)
	);
	fp32_mul u_ln_diff_sq(
		.a(ln_diff_w),
		.b(ln_diff_w),
		.y(ln_diff_sq_w)
	);
	fp32_add u_ln_var_add(
		.a(ln_var_acc_q),
		.b(ln_diff_sq_w),
		.y(ln_var_add_w)
	);
	fp32_div u_ln_var_norm(
		.a(ln_var_acc_q),
		.b(ln_n_fp32),
		.y(ln_var_norm_w)
	);
	fp32_add u_ln_var_eps(
		.a(ln_var_norm_w),
		.b(ln_eps_sel_w),
		.y(ln_var_eps_w)
	);
	fp32_sqrt u_ln_sqrt(
		.a(ln_var_eps_w),
		.y(ln_denom_w)
	);
	fp32_div u_ln_norm(
		.a(ln_diff_w),
		.b(ln_denom_q),
		.y(ln_norm_w)
	);
	fp32_mul u_ln_norm_g(
		.a(ln_norm_w),
		.b(synth_gamma_bits),
		.y(ln_norm_g_w)
	);
	fp32_add u_ln_norm_gb(
		.a(ln_norm_g_w),
		.b(synth_beta_bits),
		.y(ln_norm_gb_w)
	);
	fp32_to_fp16 u_ln_out_h(
		.a(ln_norm_gb_w),
		.y(ln_out_h_w)
	);
	wire [31:0] synth_scale1_bits;
	fp32_div u_ln_g1_scale(
		.a(ln_norm_gb_w),
		.b(synth_scale1_bits),
		.y(ln_g1_scaled_w)
	);
	fp32_quantize_i8 u_ln_g1_quant(
		.a(ln_g1_scaled_w),
		.y(ln_g1_quant_w)
	);
	wire [31:0] sm_neg_max;
	wire [31:0] sm_diff_w;
	wire [31:0] sm_exp_w;
	wire [31:0] sm_sum_add_w;
	wire [31:0] sm_norm_w;
	wire [15:0] sm_out_h_w;
	assign sm_neg_max = sm_row_max_q ^ 32'h80000000;
	fp32_add u_sm_diff(
		.a(synth_a_bits),
		.b(sm_neg_max),
		.y(sm_diff_w)
	);
	fp32_exp u_sm_exp(
		.a(sm_diff_w),
		.y(sm_exp_w)
	);
	fp32_add u_sm_sum_add(
		.a(sm_exp_sum_q),
		.b(sm_exp_w),
		.y(sm_sum_add_w)
	);
	fp32_div u_sm_div(
		.a(sm_exp_w),
		.b(sm_exp_sum_q),
		.y(sm_norm_w)
	);
	fp32_to_fp16 u_sm_out_h(
		.a(sm_norm_w),
		.y(sm_out_h_w)
	);
	wire sm_row_gt_max;
	assign sm_row_gt_max = (sm_diff_w[31] == 1'b0) && (sm_diff_w[30:0] != 31'd0);
	wire [31:0] sm_g1_scaled_w;
	wire signed [7:0] sm_g1_quant_w;
	assign synth_scale1_bits = scale1_q;
	fp32_div u_sm_g1_scale(
		.a(sm_norm_w),
		.b(synth_scale1_bits),
		.y(sm_g1_scaled_w)
	);
	fp32_quantize_i8 u_sm_g1_quant(
		.a(sm_g1_scaled_w),
		.y(sm_g1_quant_w)
	);
	wire signed [7:0] gelu_g1_i8_sel;
	reg [31:0] gelu_g1_i32_sel;
	wire [31:0] gelu_g1_in_fp32;
	wire [31:0] gelu_g1_x_bits;
	wire [31:0] gelu_g1_y_bits;
	wire [31:0] gelu_g1_scaled_w;
	wire signed [7:0] gelu_g1_quant_w;
	wire [31:0] gelu_g1_in_pick;
	assign gelu_g1_i8_sel = $signed(gelu_i8_row_q[8 * iter_idx_q[3:0]+:8]);
	always @(*) begin
		if (_sv2v_0)
			;
		case (iter_idx_q[1:0])
			2'd0: gelu_g1_i32_sel = gelu_row0_q[32 * iter_idx_q[3:2]+:32];
			2'd1: gelu_g1_i32_sel = gelu_row1_q[32 * iter_idx_q[3:2]+:32];
			2'd2: gelu_g1_i32_sel = gelu_row2_q[32 * iter_idx_q[3:2]+:32];
			default: gelu_g1_i32_sel = gelu_row3_q[32 * iter_idx_q[3:2]+:32];
		endcase
	end
	assign gelu_g1_in_pick = (state == 6'd41 ? gelu_g1_i32_sel : {{24 {gelu_g1_i8_sel[7]}}, gelu_g1_i8_sel});
	i32_to_fp32 u_gelu_g1_cvt(
		.a(gelu_g1_in_pick),
		.y(gelu_g1_in_fp32)
	);
	fp32_mul u_gelu_g1_x(
		.a(gelu_g1_in_fp32),
		.b(synth_scale0_bits),
		.y(gelu_g1_x_bits)
	);
	fp32_gelu_new u_gelu_g1_y(
		.a(gelu_g1_x_bits),
		.y(gelu_g1_y_bits)
	);
	fp32_div u_gelu_g1_s(
		.a(gelu_g1_y_bits),
		.b(synth_scale1_bits),
		.y(gelu_g1_scaled_w)
	);
	fp32_quantize_i8 u_gelu_g1_q(
		.a(gelu_g1_scaled_w),
		.y(gelu_g1_quant_w)
	);
	reg sm_visible_w;
	always @(iter_idx_q or row_idx_q or attn_valid_kv_len_q or attn_mode_q[0] or attn_mode_q[1] or attn_query_row_base_q or sm_keep_through_q[15:0] or iter_idx_q or iter_idx_q or row_idx_q or attn_valid_kv_len_q or attn_mode_q[0] or attn_mode_q[1] or attn_query_row_base_q or opcode_q or _sv2v_0) begin
		if (_sv2v_0)
			;
		case (opcode_q)
			5'h0e: sm_visible_w = 1'b1;
			5'h15: sm_visible_w = attn_visible(row_idx_q, sv2v_cast_32_signed(iter_idx_q));
			5'h1d: sm_visible_w = $signed({6'b000000, iter_idx_q}) <= $signed({1'b0, sm_keep_through_q[15:0]});
			5'h12: sm_visible_w = 1'b1;
			5'h16: sm_visible_w = attn_visible(row_idx_q, sv2v_cast_32_signed(iter_idx_q));
			default: sm_visible_w = 1'b0;
		endcase
	end
	reg [15:0] sm_iter_bound_w;
	always @(*) begin
		if (_sv2v_0)
			;
		case (opcode_q)
			5'h12, 5'h16: sm_iter_bound_w = k_elems_q;
			default: sm_iter_bound_w = n_elems_q;
		endcase
	end
	wire [31:0] attn_row_at_k_bits;
	wire [31:0] attn_diff_w;
	wire [31:0] attn_exp_w;
	wire [31:0] attn_weight_w;
	wire [31:0] attn_weight_eff_w;
	wire attn_vis_at_k;
	assign attn_row_at_k_bits = row_data_q[attn_k_idx_q[9:0]];
	fp32_add u_attn_diff(
		.a(attn_row_at_k_bits),
		.b(sm_neg_max),
		.y(attn_diff_w)
	);
	fp32_exp u_attn_exp(
		.a(attn_diff_w),
		.y(attn_exp_w)
	);
	fp32_div u_attn_div(
		.a(attn_exp_w),
		.b(sm_exp_sum_q),
		.y(attn_weight_w)
	);
	assign attn_vis_at_k = (opcode_q == 5'h12) || attn_visible(row_idx_q, sv2v_cast_32_signed(attn_k_idx_q));
	assign attn_weight_eff_w = (attn_vis_at_k ? attn_weight_w : 32'h00000000);
	wire [31:0] attn_v_lane_fp32 [0:15];
	wire [31:0] attn_v_weighted [0:15];
	wire [31:0] attn_v_scaled [0:15];
	reg [31:0] attn_acc_old_bits [0:15];
	wire [31:0] attn_acc_new_bits [0:15];
	genvar _gv_gv_lane_1;
	generate
		for (_gv_gv_lane_1 = 0; _gv_gv_lane_1 < 16; _gv_gv_lane_1 = _gv_gv_lane_1 + 1) begin : v_lane
			localparam gv_lane = _gv_gv_lane_1;
			wire signed [7:0] byte_sel;
			wire [31:0] byte_sx;
			assign byte_sel = $signed(sram_b_rdata[8 * gv_lane+:8]);
			assign byte_sx = {{24 {byte_sel[7]}}, byte_sel};
			i32_to_fp32 u_v_cvt(
				.a(byte_sx),
				.y(attn_v_lane_fp32[gv_lane])
			);
			fp32_mul u_v_w(
				.a(attn_weight_eff_w),
				.b(attn_v_lane_fp32[gv_lane]),
				.y(attn_v_weighted[gv_lane])
			);
			fp32_mul u_v_s(
				.a(attn_v_weighted[gv_lane]),
				.b(synth_scale1_bits),
				.y(attn_v_scaled[gv_lane])
			);
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] li;
			for (li = 0; li < 16; li = li + 1)
				begin : sv2v_autoblock_2
					reg signed [31:0] idx_li;
					idx_li = (sv2v_cast_32_signed(read_idx_q) * 16) + li;
					attn_acc_old_bits[li] = attn_accum_q[idx_li[9:0]];
				end
		end
	end
	generate
		for (_gv_gv_lane_1 = 0; _gv_gv_lane_1 < 16; _gv_gv_lane_1 = _gv_gv_lane_1 + 1) begin : v_acc
			localparam gv_lane = _gv_gv_lane_1;
			fp32_add u_v_add(
				.a(attn_acc_old_bits[gv_lane]),
				.b(attn_v_scaled[gv_lane]),
				.y(attn_acc_new_bits[gv_lane])
			);
		end
	endgenerate
	wire [31:0] synth_lat_h2f [0:7];
	genvar _gv_g_lj_1;
	generate
		for (_gv_g_lj_1 = 0; _gv_g_lj_1 < 8; _gv_g_lj_1 = _gv_g_lj_1 + 1) begin : g_synth_lat
			localparam g_lj = _gv_g_lj_1;
			fp16_to_fp32 u_h2f(
				.a(sram_b_rdata[g_lj * 16+:16]),
				.y(synth_lat_h2f[g_lj])
			);
		end
	endgenerate
	wire [31:0] mar_lane_abs [0:7];
	wire [31:0] mar_curr_bits;
	reg [31:0] mar_cand [0:7];
	reg [31:0] mar_new_max;
	wire [15:0] mar_base_idx;
	assign mar_base_idx = {3'h0, read_idx_q[12:0]} * 16'd8;
	assign mar_curr_bits = g2_maxabs_q & 32'h7fffffff;
	generate
		for (_gv_g_lj_1 = 0; _gv_g_lj_1 < 8; _gv_g_lj_1 = _gv_g_lj_1 + 1) begin : g_mar
			localparam g_lj = _gv_g_lj_1;
			assign mar_lane_abs[g_lj] = synth_lat_h2f[g_lj] & 32'h7fffffff;
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = 0; i < 8; i = i + 1)
				if ((mar_base_idx + sv2v_cast_16_signed(i)) < n_elems_q)
					mar_cand[i] = mar_lane_abs[i];
				else
					mar_cand[i] = mar_curr_bits;
		end
	end
	always @(*) begin : sv2v_autoblock_4
		reg [31:0] m;
		if (_sv2v_0)
			;
		m = mar_curr_bits;
		begin : sv2v_autoblock_5
			reg signed [31:0] i;
			for (i = 0; i < 8; i = i + 1)
				if (mar_cand[i] > m)
					m = mar_cand[i];
		end
		mar_new_max = m;
	end
	always @(*) begin
		if (_sv2v_0)
			;
		dispatch_attn_context_bad_w = 1'b0;
		if (opcode == 5'h1d) begin
			dispatch_attn_context_bad_w = (!attn_valid || (attn_mode == 2'b00)) || (attn_valid_kv_len == 12'h000);
			if (!dispatch_attn_context_bad_w) begin
				if (attn_mode == 2'b10)
					dispatch_attn_context_bad_w = {4'h0, dispatch_attn_key_cols_w} != {8'h00, attn_valid_kv_len};
				else if (attn_mode[0])
					dispatch_attn_context_bad_w = {4'h0, dispatch_attn_key_cols_w} < {8'h00, attn_valid_kv_len};
			end
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		dispatch_unsupported_w = 1'b0;
		dispatch_sram_oob_w = 1'b0;
		dispatch_src1_need_rows_w = 32'd0;
		dispatch_src2_need_rows_w = 32'd0;
		dispatch_dst_need_rows_w = 32'd0;
		case (opcode)
			5'h19: begin
				if (!dispatch_g2_vadd_w)
					dispatch_unsupported_w = 1'b1;
				if (sv2v_cast_32_signed(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
					dispatch_unsupported_w = 1'b1;
				dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
				dispatch_src2_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
				dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
			end
			5'h1a: begin
				if (!dispatch_g2_ln_w)
					dispatch_unsupported_w = 1'b1;
				if (sv2v_cast_32_signed(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
					dispatch_unsupported_w = 1'b1;
				dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
				dispatch_src2_need_rows_w = {16'h0000, dispatch_ln_param_rows_w};
				dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
			end
			5'h1b: begin
				if (!dispatch_g2_gelu_w)
					dispatch_unsupported_w = 1'b1;
				if (sv2v_cast_32_signed(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
					dispatch_unsupported_w = 1'b1;
				dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
				dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
			end
			5'h17: begin
				if (!dispatch_g2_dq_w)
					dispatch_unsupported_w = 1'b1;
				if (sv2v_cast_32_signed(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
					dispatch_unsupported_w = 1'b1;
				dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
				dispatch_src2_need_rows_w = {19'h00000, dispatch_g2_rows_w};
				dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
			end
			5'h18: begin
				if (!dispatch_g2_q_w)
					dispatch_unsupported_w = 1'b1;
				if (sv2v_cast_32_signed(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
					dispatch_unsupported_w = 1'b1;
				dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
				dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
			end
			5'h1d: begin
				if (!dispatch_g2_ms_w)
					dispatch_unsupported_w = 1'b1;
				if (sv2v_cast_32_signed(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
					dispatch_unsupported_w = 1'b1;
				dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
				dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
			end
			5'h1e: begin
				if (!dispatch_g2_ds_w)
					dispatch_unsupported_w = 1'b1;
				if (sv2v_cast_32_signed(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
					dispatch_unsupported_w = 1'b1;
				dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
				dispatch_src2_need_rows_w = {16'h0000, dispatch_ln_param_rows_w};
				dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
			end
			5'h1f: begin
				if (!dispatch_g2_mar_w)
					dispatch_unsupported_w = 1'b1;
				if (sv2v_cast_32_signed(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
					dispatch_unsupported_w = 1'b1;
				dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
			end
			default: dispatch_unsupported_w = 1'b1;
		endcase
		dispatch_sram_oob_w = ((({16'h0000, src1_off} + dispatch_src1_need_rows_w) > {16'h0000, dispatch_src1_rows_w}) || (({16'h0000, src2_off} + dispatch_src2_need_rows_w) > {16'h0000, dispatch_src2_rows_w})) || (({16'h0000, dst_off} + dispatch_dst_need_rows_w) > {16'h0000, dispatch_dst_rows_w});
	end
	assign row_i8_addr_w = ({16'h0000, src1_off_q} + ({17'h00000, row_idx_q} * {21'h000000, n_tiles_q})) + {19'h00000, read_idx_q};
	assign row_i32_addr_w = ({16'h0000, src1_off_q} + ({17'h00000, row_idx_q} * {19'h00000, n_chunks_i32_q})) + {19'h00000, read_idx_q};
	assign row_dst_addr_w = ({16'h0000, dst_off_q} + ({17'h00000, row_idx_q} * {21'h000000, n_tiles_q})) + {21'h000000, write_chunk_q};
	assign ln_param_addr_w = {16'h0000, src2_off_q} + {19'h00000, read_idx_q};
	assign gelu_i8_addr_w = ({16'h0000, src1_off_q} + ({17'h00000, row_idx_q} * {21'h000000, n_tiles_q})) + {21'h000000, write_chunk_q};
	assign gelu_acc_addr_w = (({16'h0000, src1_off_q} + ({17'h00000, row_idx_q} * {19'h00000, n_chunks_i32_q})) + ({21'h000000, write_chunk_q} << 2)) + {30'h00000000, gelu_part_q};
	assign gelu_dst_addr_w = ({16'h0000, dst_off_q} + ({17'h00000, row_idx_q} * {21'h000000, n_tiles_q})) + {21'h000000, write_chunk_q};
	assign attn_qkt_addr_w = ({16'h0000, src1_off_q} + ({17'h00000, row_idx_q} * {19'h00000, k_chunks_i32_q})) + {19'h00000, read_idx_q};
	assign attn_v_addr_w = ({16'h0000, src2_off_q} + ({16'h0000, attn_k_idx_q} * {21'h000000, n_tiles_q})) + {19'h00000, read_idx_q};
	wire [31:0] g2_s1_addr_w;
	wire [31:0] g2_s2_addr_w;
	wire [31:0] g2_lnp_addr_w;
	wire [31:0] g2_dst_addr_w;
	assign g2_s1_addr_w = ({16'h0000, src1_off_q} + ({17'h00000, row_idx_q} * {19'h00000, g2_rows_q})) + {19'h00000, read_idx_q};
	assign g2_s2_addr_w = ({16'h0000, src2_off_q} + ({17'h00000, row_idx_q} * {19'h00000, g2_rows_q})) + {19'h00000, read_idx_q};
	assign g2_lnp_addr_w = {16'h0000, src2_off_q} + {19'h00000, read_idx_q};
	assign g2_dst_addr_w = ({16'h0000, dst_off_q} + ({17'h00000, row_idx_q} * {19'h00000, g2_rows_q})) + {21'h000000, write_chunk_q};
	always @(*) begin
		if (_sv2v_0)
			;
		row_write_data_w = 128'h00000000000000000000000000000000;
		gelu_i8_write_data_w = 128'h00000000000000000000000000000000;
		gelu_i32_write_data_w = 128'h00000000000000000000000000000000;
		attn_write_data_w = 128'h00000000000000000000000000000000;
		g2_write_data_w = 128'h00000000000000000000000000000000;
		begin : sv2v_autoblock_6
			reg signed [31:0] g2l;
			for (g2l = 0; g2l < 8; g2l = g2l + 1)
				begin : sv2v_autoblock_7
					reg signed [31:0] g2idx;
					g2idx = (sv2v_cast_32_signed(write_chunk_q) * 8) + g2l;
					if (g2idx < sv2v_cast_32_signed(n_elems_q))
						g2_write_data_w[g2l * 16+:16] = out_h_q[g2idx];
				end
		end
		begin : sv2v_autoblock_8
			reg signed [31:0] lane;
			for (lane = 0; lane < 16; lane = lane + 1)
				begin : sv2v_autoblock_9
					reg signed [31:0] idx;
					idx = (sv2v_cast_32_signed(write_chunk_q) * 16) + lane;
					if (idx < sv2v_cast_32_signed(n_elems_q))
						row_write_data_w[lane * 8+:8] = out_bytes_q[idx];
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 6'd0;
			opcode_q <= 5'h00;
			src1_buf_q <= 2'b00;
			src2_buf_q <= 2'b00;
			dst_buf_q <= 2'b00;
			src1_off_q <= 16'h0000;
			src2_off_q <= 16'h0000;
			dst_off_q <= 16'h0000;
			sreg_q <= 4'h0;
			m_rows_q <= 15'h0000;
			n_tiles_q <= 11'h000;
			k_tiles_q <= 11'h000;
			n_chunks_i32_q <= 13'h0000;
			k_chunks_i32_q <= 13'h0000;
			n_elems_q <= 16'h0000;
			k_elems_q <= 16'h0000;
			ln_gamma_rows_q <= 16'h0000;
			ln_param_rows_q <= 16'h0000;
			attn_valid_q <= 1'b0;
			attn_query_row_base_q <= 12'h000;
			attn_valid_kv_len_q <= 12'h000;
			attn_mode_q <= 2'b00;
			fault_code_r <= 4'h0;
			row_idx_q <= 15'h0000;
			read_idx_q <= 13'h0000;
			iter_idx_q <= 11'h000;
			ln_sum_acc_q <= 32'h00000000;
			ln_var_acc_q <= 32'h00000000;
			ln_mean_q <= 32'h00000000;
			ln_denom_q <= 32'h00000000;
			sm_row_max_q <= 32'h00000000;
			sm_exp_sum_q <= 32'h00000000;
			sm_have_vis_q <= 1'b0;
			sm_keep_through_q <= 16'sh0000;
			write_chunk_q <= 11'h000;
			gelu_part_q <= 2'h0;
			attn_k_idx_q <= 16'h0000;
			g2_rows_q <= 13'h0000;
			g2_maxabs_q <= 0.0;
			g2_wr_phase_q <= 1'b0;
			gelu_i8_row_q <= 128'h00000000000000000000000000000000;
			gelu_row0_q <= 128'h00000000000000000000000000000000;
			gelu_row1_q <= 128'h00000000000000000000000000000000;
			gelu_row2_q <= 128'h00000000000000000000000000000000;
			gelu_row3_q <= 128'h00000000000000000000000000000000;
			row_write_q <= 128'h00000000000000000000000000000000;
			scale0_q <= 0.0;
			scale1_q <= 0.0;
			scale2_q <= 0.0;
			scale3_q <= 0.0;
			attn_row_max_q <= 0.0;
			attn_exp_sum_q <= 0.0;
			begin : sv2v_autoblock_10
				reg signed [31:0] i;
				for (i = 0; i < SFU_MAX_ROW_ELEMS; i = i + 1)
					begin
						row_data_q[i] <= 0.0;
						attn_accum_q[i] <= 0.0;
						gamma_q[i] <= 0.0;
						beta_q[i] <= 0.0;
						out_bytes_q[i] <= 8'h00;
						out_h_q[i] <= 16'h0000;
					end
			end
		end
		else
			case (state)
				6'd0:
					if (dispatch) begin
						opcode_q <= opcode;
						src1_buf_q <= src1_buf;
						src2_buf_q <= src2_buf;
						dst_buf_q <= dst_buf;
						src1_off_q <= src1_off;
						src2_off_q <= src2_off;
						dst_off_q <= dst_off;
						sreg_q <= sreg;
						m_rows_q <= dispatch_m_rows_w;
						n_tiles_q <= dispatch_n_tiles_w;
						k_tiles_q <= dispatch_k_tiles_w;
						n_chunks_i32_q <= dispatch_n_chunks_i32_w;
						k_chunks_i32_q <= dispatch_k_chunks_i32_w;
						n_elems_q <= dispatch_n_elems_w;
						k_elems_q <= dispatch_k_elems_w;
						ln_gamma_rows_q <= dispatch_ln_gamma_rows_w;
						ln_param_rows_q <= dispatch_ln_param_rows_w;
						g2_rows_q <= dispatch_g2_rows_w;
						g2_maxabs_q <= 0.0;
						g2_wr_phase_q <= 1'b0;
						attn_valid_q <= attn_valid;
						attn_query_row_base_q <= attn_query_row_base;
						attn_valid_kv_len_q <= attn_valid_kv_len;
						attn_mode_q <= attn_mode;
						scale0_q <= {16'h0000, scale0_data};
						scale1_q <= {16'h0000, scale1_data};
						scale2_q <= {16'h0000, scale2_data};
						scale3_q <= {16'h0000, scale3_data};
						row_idx_q <= 15'h0000;
						read_idx_q <= 13'h0000;
						write_chunk_q <= 11'h000;
						gelu_part_q <= 2'h0;
						attn_k_idx_q <= 16'h0000;
						if (dispatch_unsupported_w) begin
							fault_code_r <= 4'h6;
							state <= 6'd22;
						end
						else if (dispatch_attn_context_bad_w) begin
							fault_code_r <= 4'h4;
							state <= 6'd22;
						end
						else if (dispatch_sram_oob_w) begin
							fault_code_r <= 4'h3;
							state <= 6'd22;
						end
						else
							case (opcode)
								5'h19, 5'h1a, 5'h1b, 5'h17, 5'h18, 5'h1d, 5'h1e, 5'h1f: state <= 6'd23;
								default: begin
									fault_code_r <= 4'h6;
									state <= 6'd22;
								end
							endcase
					end
				6'd8: begin
					row_write_q <= row_write_data_w;
					state <= 6'd9;
				end
				6'd9:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd22;
					end
					else if ((write_chunk_q + 11'd1) < n_tiles_q) begin
						write_chunk_q <= write_chunk_q + 11'd1;
						state <= 6'd8;
					end
					else if ((row_idx_q + 15'd1) < m_rows_q) begin
						row_idx_q <= row_idx_q + 15'd1;
						read_idx_q <= 13'h0000;
						write_chunk_q <= 11'h000;
						state <= 6'd23;
					end
					else
						state <= 6'd0;
				6'd23:
					if (sram_b_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd22;
					end
					else
						state <= 6'd24;
				6'd24: begin : sv2v_autoblock_11
					integer base_idx;
					if ((opcode_q == 5'h17) || (opcode_q == 5'h1e)) begin
						base_idx = sv2v_cast_32_signed(read_idx_q) * 4;
						if ((read_idx_q + 13'd1) < n_chunks_i32_q) begin
							read_idx_q <= read_idx_q + 13'd1;
							state <= 6'd23;
						end
						else begin
							read_idx_q <= 13'h0000;
							write_chunk_q <= 11'h000;
							state <= 6'd25;
						end
					end
					else if (opcode_q == 5'h1f) begin
						base_idx = sv2v_cast_32_signed(read_idx_q) * 8;
						if (SFU_SYNTH_MODE == 1)
							g2_maxabs_q <= mar_new_max;
						if ((read_idx_q + 13'd1) < {2'h0, g2_rows_q[10:0]}) begin
							read_idx_q <= read_idx_q + 13'd1;
							state <= 6'd23;
						end
						else if ((row_idx_q + 15'd1) < m_rows_q) begin
							row_idx_q <= row_idx_q + 15'd1;
							read_idx_q <= 13'h0000;
							state <= 6'd23;
						end
						else
							state <= 6'd30;
					end
					else begin
						base_idx = sv2v_cast_32_signed(read_idx_q) * 8;
						begin : sv2v_autoblock_12
							reg signed [31:0] lane;
							for (lane = 0; lane < 8; lane = lane + 1)
								if ((base_idx + lane) < sv2v_cast_32_signed(n_elems_q)) begin
									if (SFU_SYNTH_MODE == 1)
										row_data_q[base_idx + lane] <= synth_lat_h2f[lane];
								end
						end
						if ((read_idx_q + 13'd1) < {2'h0, g2_rows_q[10:0]}) begin
							read_idx_q <= read_idx_q + 13'd1;
							state <= 6'd23;
						end
						else begin
							read_idx_q <= 13'h0000;
							write_chunk_q <= 11'h000;
							if (((opcode_q == 5'h1b) || (opcode_q == 5'h18)) || (opcode_q == 5'h1d))
								state <= 6'd27;
							else
								state <= 6'd25;
						end
					end
				end
				6'd25:
					if (sram_b_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd22;
					end
					else
						state <= 6'd26;
				6'd26: begin : sv2v_autoblock_13
					integer base_idx;
					if ((opcode_q == 5'h1a) || (opcode_q == 5'h1e)) begin
						base_idx = (sv2v_cast_32_signed(read_idx_q) < sv2v_cast_32_signed(ln_gamma_rows_q) ? sv2v_cast_32_signed(read_idx_q) * 8 : (sv2v_cast_32_signed(read_idx_q) - sv2v_cast_32_signed(ln_gamma_rows_q)) * 8);
						begin : sv2v_autoblock_14
							reg signed [31:0] lane;
							for (lane = 0; lane < 8; lane = lane + 1)
								if ((base_idx + lane) < sv2v_cast_32_signed(n_elems_q)) begin
									if (sv2v_cast_32_signed(read_idx_q) < sv2v_cast_32_signed(ln_gamma_rows_q)) begin
										if (SFU_SYNTH_MODE == 1)
											gamma_q[base_idx + lane] <= synth_lat_h2f[lane];
									end
									else if (SFU_SYNTH_MODE == 1)
										beta_q[base_idx + lane] <= synth_lat_h2f[lane];
								end
						end
						if ((read_idx_q + 13'd1) < {1'b0, ln_param_rows_q[11:0]}) begin
							read_idx_q <= read_idx_q + 13'd1;
							state <= 6'd25;
						end
						else begin
							read_idx_q <= 13'h0000;
							write_chunk_q <= 11'h000;
							state <= 6'd27;
						end
					end
					else begin
						base_idx = sv2v_cast_32_signed(read_idx_q) * 8;
						begin : sv2v_autoblock_15
							reg signed [31:0] lane;
							for (lane = 0; lane < 8; lane = lane + 1)
								if ((base_idx + lane) < sv2v_cast_32_signed(n_elems_q)) begin
									if (SFU_SYNTH_MODE == 1)
										attn_accum_q[base_idx + lane] <= synth_lat_h2f[lane];
								end
						end
						if ((read_idx_q + 13'd1) < {2'h0, g2_rows_q[10:0]}) begin
							read_idx_q <= read_idx_q + 13'd1;
							state <= 6'd25;
						end
						else begin
							read_idx_q <= 13'h0000;
							write_chunk_q <= 11'h000;
							state <= 6'd27;
						end
					end
				end
				6'd27:
					if (opcode_q == 5'h19) begin
						if (SFU_SYNTH_MODE == 1) begin
							iter_idx_q <= 11'h000;
							state <= 6'd31;
						end
					end
					else if (opcode_q == 5'h1b) begin
						if (SFU_SYNTH_MODE == 1) begin
							iter_idx_q <= 11'h000;
							state <= 6'd31;
						end
					end
					else if (opcode_q == 5'h17) begin
						if (SFU_SYNTH_MODE == 1) begin
							iter_idx_q <= 11'h000;
							state <= 6'd31;
						end
					end
					else if (opcode_q == 5'h18) begin
						if (SFU_SYNTH_MODE == 1) begin
							iter_idx_q <= 11'h000;
							state <= 6'd31;
						end
					end
					else if (opcode_q == 5'h1e) begin
						if (SFU_SYNTH_MODE == 1) begin
							iter_idx_q <= 11'h000;
							state <= 6'd31;
						end
					end
					else if (opcode_q == 5'h1d) begin
						if (SFU_SYNTH_MODE == 1) begin : sv2v_autoblock_16
							reg signed [16:0] qrow_s;
							reg signed [16:0] kt_s;
							qrow_s = $signed({5'b00000, attn_query_row_base_q}) + $signed({2'b00, row_idx_q[14:0]});
							kt_s = $signed({5'b00000, attn_valid_kv_len_q}) - 17'sd1;
							if (qrow_s < kt_s)
								sm_keep_through_q <= sv2v_cast_16_signed(qrow_s);
							else
								sm_keep_through_q <= sv2v_cast_16_signed(kt_s);
							iter_idx_q <= 11'h000;
							sm_row_max_q <= 32'h00000000;
							sm_exp_sum_q <= 32'h00000000;
							sm_have_vis_q <= 1'b0;
							state <= 6'd37;
						end
					end
					else if ((SFU_SYNTH_MODE == 1) && (opcode_q == 5'h1a)) begin
						iter_idx_q <= 11'h000;
						ln_sum_acc_q <= 32'h00000000;
						state <= 6'd32;
					end
				6'd31:
					if ({5'h00, iter_idx_q} < n_elems_q) begin
						if (opcode_q == 5'h18)
							out_bytes_q[iter_idx_q[9:0]] <= synth_quant_out;
						else
							out_h_q[iter_idx_q[9:0]] <= synth_out_bits;
						iter_idx_q <= iter_idx_q + 11'd1;
					end
					else begin
						iter_idx_q <= 11'h000;
						state <= (opcode_q == 5'h18 ? 6'd8 : 6'd28);
					end
				6'd32:
					if ({5'h00, iter_idx_q} < n_elems_q) begin
						ln_sum_acc_q <= ln_sum_add_w;
						iter_idx_q <= iter_idx_q + 11'd1;
					end
					else begin
						iter_idx_q <= 11'h000;
						state <= 6'd33;
					end
				6'd33: begin
					ln_mean_q <= ln_mean_div_w;
					ln_var_acc_q <= 32'h00000000;
					state <= 6'd34;
				end
				6'd34:
					if ({5'h00, iter_idx_q} < n_elems_q) begin
						ln_var_acc_q <= ln_var_add_w;
						iter_idx_q <= iter_idx_q + 11'd1;
					end
					else begin
						iter_idx_q <= 11'h000;
						state <= 6'd35;
					end
				6'd35: begin
					ln_denom_q <= ln_denom_w;
					state <= 6'd36;
				end
				6'd36:
					if ({5'h00, iter_idx_q} < n_elems_q) begin
						if (opcode_q == 5'h0f)
							out_bytes_q[iter_idx_q[9:0]] <= ln_g1_quant_w;
						else
							out_h_q[iter_idx_q[9:0]] <= ln_out_h_w;
						iter_idx_q <= iter_idx_q + 11'd1;
					end
					else begin
						iter_idx_q <= 11'h000;
						state <= (opcode_q == 5'h0f ? 6'd8 : 6'd28);
					end
				6'd37:
					if ({5'h00, iter_idx_q} < sm_iter_bound_w) begin
						if (sm_visible_w) begin
							if (!sm_have_vis_q || sm_row_gt_max)
								sm_row_max_q <= synth_a_bits;
							sm_have_vis_q <= 1'b1;
						end
						iter_idx_q <= iter_idx_q + 11'd1;
					end
					else begin
						iter_idx_q <= 11'h000;
						if (((((opcode_q == 5'h0e) || (opcode_q == 5'h15)) || (opcode_q == 5'h12)) || (opcode_q == 5'h16)) && !sm_have_vis_q) begin
							fault_code_r <= 4'h4;
							state <= 6'd22;
						end
						else
							state <= 6'd38;
					end
				6'd38:
					if ({5'h00, iter_idx_q} < sm_iter_bound_w) begin
						if (sm_have_vis_q && sm_visible_w)
							sm_exp_sum_q <= sm_sum_add_w;
						iter_idx_q <= iter_idx_q + 11'd1;
					end
					else begin
						iter_idx_q <= 11'h000;
						if (((((opcode_q == 5'h0e) || (opcode_q == 5'h15)) || (opcode_q == 5'h12)) || (opcode_q == 5'h16)) && (sm_exp_sum_q == 32'h00000000)) begin
							fault_code_r <= 4'h4;
							state <= 6'd22;
						end
						else if ((opcode_q == 5'h12) || (opcode_q == 5'h16)) begin
							attn_row_max_q <= sm_row_max_q;
							attn_exp_sum_q <= sm_exp_sum_q;
							begin : sv2v_autoblock_17
								reg signed [31:0] zi;
								for (zi = 0; zi < SFU_MAX_ROW_ELEMS; zi = zi + 1)
									if (zi < sv2v_cast_32_signed(n_elems_q))
										attn_accum_q[zi] <= 0.0;
							end
							attn_k_idx_q <= 16'h0000;
							read_idx_q <= 13'h0000;
							write_chunk_q <= 11'h000;
							state <= 6'd19;
						end
						else
							state <= 6'd39;
					end
				6'd39:
					if ({5'h00, iter_idx_q} < sm_iter_bound_w) begin
						if ((opcode_q == 5'h0e) || (opcode_q == 5'h15)) begin
							if ((sm_have_vis_q && sm_visible_w) && (sm_exp_sum_q != 32'h00000000))
								out_bytes_q[iter_idx_q[9:0]] <= sm_g1_quant_w;
							else
								out_bytes_q[iter_idx_q[9:0]] <= 8'h00;
						end
						else if ((sm_have_vis_q && sm_visible_w) && (sm_exp_sum_q != 32'h00000000))
							out_h_q[iter_idx_q[9:0]] <= sm_out_h_w;
						else
							out_h_q[iter_idx_q[9:0]] <= 16'h0000;
						iter_idx_q <= iter_idx_q + 11'd1;
					end
					else begin
						iter_idx_q <= 11'h000;
						if ((opcode_q == 5'h0e) || (opcode_q == 5'h15))
							state <= 6'd8;
						else
							state <= 6'd28;
					end
				6'd28: begin
					row_write_q <= g2_write_data_w;
					state <= 6'd29;
				end
				6'd29:
					if (sram_a_fault) begin
						fault_code_r <= 4'h3;
						state <= 6'd22;
					end
					else if ((write_chunk_q + 11'd1) < g2_rows_q[10:0]) begin
						write_chunk_q <= write_chunk_q + 11'd1;
						state <= 6'd28;
					end
					else if ((row_idx_q + 15'd1) < m_rows_q) begin
						row_idx_q <= row_idx_q + 15'd1;
						read_idx_q <= 13'h0000;
						write_chunk_q <= 11'h000;
						state <= 6'd23;
					end
					else
						state <= 6'd0;
				6'd30:
					if (g2_wr_phase_q == 1'b0)
						g2_wr_phase_q <= 1'b1;
					else
						state <= 6'd0;
				6'd22:
					;
				default: state <= 6'd0;
			endcase
	always @(*) begin
		if (_sv2v_0)
			;
		sfu_busy = (state != 6'd0) && (state != 6'd22);
		sfu_fault = state == 6'd22;
		sfu_fault_code = fault_code_r;
		sram_a_en = 1'b0;
		sram_a_we = 1'b0;
		sram_a_buf = dst_buf_q;
		sram_a_row = 16'h0000;
		sram_a_wdata = 128'h00000000000000000000000000000000;
		sram_b_en = 1'b0;
		sram_b_buf = src1_buf_q;
		sram_b_row = 16'h0000;
		sfu_scale_we = 1'b0;
		sfu_scale_waddr = 4'h0;
		sfu_scale_wdata = 16'h0000;
		case (state)
			6'd9: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = row_dst_addr_w[15:0];
				sram_a_wdata = row_write_q;
			end
			6'd23: begin
				sram_b_en = 1'b1;
				sram_b_buf = src1_buf_q;
				sram_b_row = ((opcode_q == 5'h17) || (opcode_q == 5'h1e) ? row_i32_addr_w[15:0] : g2_s1_addr_w[15:0]);
			end
			6'd25: begin
				sram_b_en = 1'b1;
				sram_b_buf = src2_buf_q;
				sram_b_row = (opcode_q == 5'h19 ? g2_s2_addr_w[15:0] : g2_lnp_addr_w[15:0]);
			end
			6'd29: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = g2_dst_addr_w[15:0];
				sram_a_wdata = row_write_q;
			end
			6'd30: begin
				sfu_scale_we = 1'b1;
				if (g2_wr_phase_q == 1'b0) begin
					sfu_scale_waddr = sreg_q;
					if (SFU_SYNTH_MODE == 1)
						sfu_scale_wdata = synth_inv_eps_fp16;
				end
				else begin
					sfu_scale_waddr = sreg_q + 4'd1;
					if (SFU_SYNTH_MODE == 1)
						sfu_scale_wdata = synth_eps_inv127_fp16;
				end
			end
			default:
				;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
module register_file (
	clk,
	rst_n,
	scale_we,
	scale_waddr,
	scale_wdata,
	scale_raddr0,
	scale_rdata0,
	scale_raddr1,
	scale_rdata1,
	scale_raddr2,
	scale_rdata2,
	scale_raddr3,
	scale_rdata3,
	addr_lo_we,
	addr_hi_we,
	addr_wsel,
	addr_imm28,
	addr_rsel,
	addr_rdata,
	tile_we,
	tile_m_in,
	tile_n_in,
	tile_k_in,
	attn_we,
	attn_query_row_base_in,
	attn_valid_kv_len_in,
	attn_mode_in,
	tile_m,
	tile_n,
	tile_k,
	tile_valid,
	attn_valid,
	attn_query_row_base,
	attn_valid_kv_len,
	attn_mode
);
	input wire clk;
	input wire rst_n;
	input wire scale_we;
	input wire [3:0] scale_waddr;
	input wire [15:0] scale_wdata;
	input wire [3:0] scale_raddr0;
	output wire [15:0] scale_rdata0;
	input wire [3:0] scale_raddr1;
	output wire [15:0] scale_rdata1;
	input wire [3:0] scale_raddr2;
	output wire [15:0] scale_rdata2;
	input wire [3:0] scale_raddr3;
	output wire [15:0] scale_rdata3;
	input wire addr_lo_we;
	input wire addr_hi_we;
	input wire [1:0] addr_wsel;
	input wire [27:0] addr_imm28;
	input wire [1:0] addr_rsel;
	output wire [55:0] addr_rdata;
	input wire tile_we;
	input wire [9:0] tile_m_in;
	input wire [9:0] tile_n_in;
	input wire [9:0] tile_k_in;
	input wire attn_we;
	input wire [11:0] attn_query_row_base_in;
	input wire [11:0] attn_valid_kv_len_in;
	input wire [1:0] attn_mode_in;
	output reg [9:0] tile_m;
	output reg [9:0] tile_n;
	output reg [9:0] tile_k;
	output reg tile_valid;
	output reg attn_valid;
	output reg [11:0] attn_query_row_base;
	output reg [11:0] attn_valid_kv_len;
	output reg [1:0] attn_mode;
	localparam signed [31:0] taccel_pkg_NUM_SCALE_REGS = 16;
	reg [15:0] scale_regs [0:15];
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < taccel_pkg_NUM_SCALE_REGS; i = i + 1)
				scale_regs[i] <= 16'h0000;
		end
		else if (scale_we)
			scale_regs[scale_waddr] <= scale_wdata;
	assign scale_rdata0 = scale_regs[scale_raddr0];
	assign scale_rdata1 = scale_regs[scale_raddr1];
	assign scale_rdata2 = scale_regs[scale_raddr2];
	assign scale_rdata3 = scale_regs[scale_raddr3];
	localparam signed [31:0] taccel_pkg_NUM_ADDR_REGS = 4;
	reg [55:0] addr_regs [0:3];
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < taccel_pkg_NUM_ADDR_REGS; i = i + 1)
				addr_regs[i] <= 56'h00000000000000;
		end
		else begin
			if (addr_lo_we)
				addr_regs[addr_wsel][27:0] <= addr_imm28;
			if (addr_hi_we)
				addr_regs[addr_wsel][55:28] <= addr_imm28;
		end
	assign addr_rdata = addr_regs[addr_rsel];
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			tile_m <= 10'h000;
			tile_n <= 10'h000;
			tile_k <= 10'h000;
			tile_valid <= 1'b0;
		end
		else if (tile_we) begin
			tile_m <= tile_m_in;
			tile_n <= tile_n_in;
			tile_k <= tile_k_in;
			tile_valid <= 1'b1;
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			attn_valid <= 1'b0;
			attn_query_row_base <= 12'h000;
			attn_valid_kv_len <= 12'h000;
			attn_mode <= 2'b00;
		end
		else if (attn_we) begin
			attn_valid <= 1'b1;
			attn_query_row_base <= attn_query_row_base_in;
			attn_valid_kv_len <= attn_valid_kv_len_in;
			attn_mode <= attn_mode_in;
		end
endmodule
module sram_dp_inferred (
	clk,
	a_en,
	a_we,
	a_addr,
	a_wdata,
	a_rdata,
	b_en,
	b_addr,
	b_rdata
);
	parameter signed [31:0] DATA_W = 128;
	parameter signed [31:0] DEPTH = 8192;
	input wire clk;
	input wire a_en;
	input wire a_we;
	input wire [$clog2(DEPTH) - 1:0] a_addr;
	input wire [DATA_W - 1:0] a_wdata;
	output reg [DATA_W - 1:0] a_rdata;
	input wire b_en;
	input wire [$clog2(DEPTH) - 1:0] b_addr;
	output reg [DATA_W - 1:0] b_rdata;
	(* ram_style = "block" *) reg [DATA_W - 1:0] mem [0:DEPTH - 1];
	always @(posedge clk)
		if (a_en) begin
			if (a_we)
				mem[a_addr] <= a_wdata;
			a_rdata <= (a_we ? a_wdata : mem[a_addr]);
		end
	always @(posedge clk)
		if (b_en)
			b_rdata <= mem[b_addr];
endmodule
module sram_dp (
	clk,
	a_en,
	a_we,
	a_addr,
	a_wdata,
	a_rdata,
	b_en,
	b_addr,
	b_rdata
);
	parameter signed [31:0] DATA_W = 128;
	parameter signed [31:0] DEPTH = 8192;
	input wire clk;
	input wire a_en;
	input wire a_we;
	input wire [$clog2(DEPTH) - 1:0] a_addr;
	input wire [DATA_W - 1:0] a_wdata;
	output wire [DATA_W - 1:0] a_rdata;
	input wire b_en;
	input wire [$clog2(DEPTH) - 1:0] b_addr;
	output wire [DATA_W - 1:0] b_rdata;
	sram_dp_inferred #(
		.DATA_W(DATA_W),
		.DEPTH(DEPTH)
	) u_impl(
		.clk(clk),
		.a_en(a_en),
		.a_we(a_we),
		.a_addr(a_addr),
		.a_wdata(a_wdata),
		.a_rdata(a_rdata),
		.b_en(b_en),
		.b_addr(b_addr),
		.b_rdata(b_rdata)
	);
endmodule
module sram_subsystem (
	clk,
	rst_n,
	a_en,
	a_we,
	a_buf,
	a_row,
	a_wdata,
	a_rdata,
	a_fault,
	b_en,
	b_buf,
	b_row,
	b_rdata,
	b_fault
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire a_en;
	input wire a_we;
	input wire [1:0] a_buf;
	input wire [15:0] a_row;
	input wire [127:0] a_wdata;
	output reg [127:0] a_rdata;
	output wire a_fault;
	input wire b_en;
	input wire [1:0] b_buf;
	input wire [15:0] b_row;
	output reg [127:0] b_rdata;
	output wire b_fault;
	localparam signed [31:0] taccel_pkg_ABUF_ROWS = 8192;
	localparam signed [31:0] taccel_pkg_ACCUM_ROWS = 4096;
	localparam signed [31:0] taccel_pkg_WBUF_ROWS = 16384;
	function automatic signed [15:0] sv2v_cast_16_signed;
		input reg signed [15:0] inp;
		sv2v_cast_16_signed = inp;
	endfunction
	function automatic oob_check;
		input reg [1:0] bid;
		input reg [15:0] row;
		case (bid)
			2'b00: oob_check = row >= sv2v_cast_16_signed(taccel_pkg_ABUF_ROWS);
			2'b01: oob_check = row >= sv2v_cast_16_signed(taccel_pkg_WBUF_ROWS);
			2'b10: oob_check = row >= sv2v_cast_16_signed(taccel_pkg_ACCUM_ROWS);
			default: oob_check = 1'b1;
		endcase
	endfunction
	assign a_fault = oob_check(a_buf, a_row);
	assign b_fault = oob_check(b_buf, b_row);
	wire [127:0] abuf_a_rdata;
	wire [127:0] abuf_b_rdata;
	wire abuf_a_en;
	wire abuf_b_en;
	wire abuf_a_we;
	assign abuf_a_en = (a_en && (a_buf == 2'b00)) && !a_fault;
	assign abuf_a_we = a_we;
	assign abuf_b_en = (b_en && (b_buf == 2'b00)) && !b_fault;
	sram_dp #(
		.DATA_W(128),
		.DEPTH(taccel_pkg_ABUF_ROWS)
	) u_abuf(
		.clk(clk),
		.a_en(abuf_a_en),
		.a_we(abuf_a_we),
		.a_addr(a_row[12:0]),
		.a_wdata(a_wdata),
		.a_rdata(abuf_a_rdata),
		.b_en(abuf_b_en),
		.b_addr(b_row[12:0]),
		.b_rdata(abuf_b_rdata)
	);
	wire [127:0] wbuf_a_rdata;
	wire [127:0] wbuf_b_rdata;
	wire wbuf_a_en;
	wire wbuf_b_en;
	wire wbuf_a_we;
	assign wbuf_a_en = (a_en && (a_buf == 2'b01)) && !a_fault;
	assign wbuf_a_we = a_we;
	assign wbuf_b_en = (b_en && (b_buf == 2'b01)) && !b_fault;
	sram_dp #(
		.DATA_W(128),
		.DEPTH(taccel_pkg_WBUF_ROWS)
	) u_wbuf(
		.clk(clk),
		.a_en(wbuf_a_en),
		.a_we(wbuf_a_we),
		.a_addr(a_row[13:0]),
		.a_wdata(a_wdata),
		.a_rdata(wbuf_a_rdata),
		.b_en(wbuf_b_en),
		.b_addr(b_row[13:0]),
		.b_rdata(wbuf_b_rdata)
	);
	wire [127:0] accum_a_rdata;
	wire [127:0] accum_b_rdata;
	wire accum_a_en;
	wire accum_b_en;
	wire accum_a_we;
	assign accum_a_en = (a_en && (a_buf == 2'b10)) && !a_fault;
	assign accum_a_we = a_we;
	assign accum_b_en = (b_en && (b_buf == 2'b10)) && !b_fault;
	sram_dp #(
		.DATA_W(128),
		.DEPTH(taccel_pkg_ACCUM_ROWS)
	) u_accum(
		.clk(clk),
		.a_en(accum_a_en),
		.a_we(accum_a_we),
		.a_addr(a_row[11:0]),
		.a_wdata(a_wdata),
		.a_rdata(accum_a_rdata),
		.b_en(accum_b_en),
		.b_addr(b_row[11:0]),
		.b_rdata(accum_b_rdata)
	);
	reg [1:0] a_buf_q;
	reg [1:0] b_buf_q;
	always @(posedge clk) begin
		a_buf_q <= a_buf;
		b_buf_q <= b_buf;
	end
	always @(*) begin
		if (_sv2v_0)
			;
		case (a_buf_q)
			2'b00: a_rdata = abuf_a_rdata;
			2'b01: a_rdata = wbuf_a_rdata;
			2'b10: a_rdata = accum_a_rdata;
			default: a_rdata = 1'sb0;
		endcase
	end
	always @(*) begin
		if (_sv2v_0)
			;
		case (b_buf_q)
			2'b00: b_rdata = abuf_b_rdata;
			2'b01: b_rdata = wbuf_b_rdata;
			2'b10: b_rdata = accum_b_rdata;
			default: b_rdata = 1'sb0;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
module systolic_pe (
	clk,
	rst_n,
	en,
	acc_clear,
	a_in,
	b_in,
	a_out,
	b_out,
	acc
);
	input wire clk;
	input wire rst_n;
	input wire en;
	input wire acc_clear;
	input wire [7:0] a_in;
	input wire [7:0] b_in;
	output reg [7:0] a_out;
	output reg [7:0] b_out;
	output reg [31:0] acc;
	wire signed [7:0] a_s;
	wire signed [7:0] b_s;
	wire signed [15:0] prod_s;
	assign a_s = a_in;
	assign b_s = b_in;
	assign prod_s = a_s * b_s;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			a_out <= 8'h00;
			b_out <= 8'h00;
			acc <= 32'h00000000;
		end
		else if (acc_clear) begin
			a_out <= 8'h00;
			b_out <= 8'h00;
			acc <= 32'h00000000;
		end
		else if (en) begin
			a_out <= a_in;
			b_out <= b_in;
			acc <= $signed(acc) + $signed({{16 {prod_s[15]}}, prod_s});
		end
endmodule
module systolic_array (
	clk,
	rst_n,
	step_en,
	clear_acc,
	a_row_data,
	b_row_data,
	acc_flat
);
	reg _sv2v_0;
	localparam signed [31:0] taccel_pkg_SYS_MODE_CHAINED = 1;
	localparam signed [31:0] taccel_pkg_SYS_MODE_DEFAULT = taccel_pkg_SYS_MODE_CHAINED;
	parameter signed [31:0] SYSTOLIC_ARCH_MODE = taccel_pkg_SYS_MODE_DEFAULT;
	input wire clk;
	input wire rst_n;
	input wire step_en;
	input wire clear_acc;
	localparam signed [31:0] taccel_pkg_AXI_DATA_W = 128;
	input wire [127:0] a_row_data;
	input wire [127:0] b_row_data;
	localparam signed [31:0] taccel_pkg_SYS_DIM = 16;
	output wire [((taccel_pkg_SYS_DIM * taccel_pkg_SYS_DIM) * 32) - 1:0] acc_flat;
	wire [7:0] a_vec [0:15];
	wire [7:0] b_vec [0:15];
	wire [7:0] a_edge_vec [0:15];
	wire [7:0] b_edge_vec [0:15];
	reg [1919:0] a_skew;
	reg [1919:0] b_skew;
	wire [((taccel_pkg_SYS_DIM * taccel_pkg_SYS_DIM) * 32) - 1:0] pe_acc;
	reg [((taccel_pkg_SYS_DIM * taccel_pkg_SYS_DIM) * 8) - 1:0] pe_a_in;
	reg [((taccel_pkg_SYS_DIM * taccel_pkg_SYS_DIM) * 8) - 1:0] pe_b_in;
	wire [((taccel_pkg_SYS_DIM * taccel_pkg_SYS_DIM) * 8) - 1:0] pe_a_out;
	wire [((taccel_pkg_SYS_DIM * taccel_pkg_SYS_DIM) * 8) - 1:0] pe_b_out;
	genvar _gv_i_1;
	genvar _gv_j_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < taccel_pkg_SYS_DIM; _gv_i_1 = _gv_i_1 + 1) begin : GEN_A_B
			localparam i = _gv_i_1;
			assign a_vec[i] = a_row_data[i * 8+:8];
			assign b_vec[i] = b_row_data[i * 8+:8];
			if (i == 0) begin : GEN_EDGE_NO_DELAY
				assign a_edge_vec[i] = a_vec[i];
				assign b_edge_vec[i] = b_vec[i];
			end
			else begin : GEN_EDGE_DELAYED
				assign a_edge_vec[i] = a_skew[((i * 15) + (i - 1)) * 8+:8];
				assign b_edge_vec[i] = b_skew[((i * 15) + (i - 1)) * 8+:8];
			end
		end
	endgenerate
	always @(posedge clk or negedge rst_n) begin : SKew_PIPE
		reg signed [31:0] r;
		reg signed [31:0] s;
		if (!rst_n)
			for (r = 0; r < taccel_pkg_SYS_DIM; r = r + 1)
				for (s = 0; s < 15; s = s + 1)
					begin
						a_skew[((r * 15) + s) * 8+:8] <= 8'h00;
						b_skew[((r * 15) + s) * 8+:8] <= 8'h00;
					end
		else if (clear_acc)
			for (r = 0; r < taccel_pkg_SYS_DIM; r = r + 1)
				for (s = 0; s < 15; s = s + 1)
					begin
						a_skew[((r * 15) + s) * 8+:8] <= 8'h00;
						b_skew[((r * 15) + s) * 8+:8] <= 8'h00;
					end
		else if (step_en)
			for (r = 0; r < taccel_pkg_SYS_DIM; r = r + 1)
				begin
					a_skew[((r * 15) + 0) * 8+:8] <= a_vec[r];
					b_skew[((r * 15) + 0) * 8+:8] <= b_vec[r];
					for (s = 1; s < 15; s = s + 1)
						begin
							a_skew[((r * 15) + s) * 8+:8] <= a_skew[((r * 15) + (s - 1)) * 8+:8];
							b_skew[((r * 15) + s) * 8+:8] <= b_skew[((r * 15) + (s - 1)) * 8+:8];
						end
				end
	end
	generate
		for (_gv_i_1 = 0; _gv_i_1 < taccel_pkg_SYS_DIM; _gv_i_1 = _gv_i_1 + 1) begin : GEN_ROUTE_ROW
			localparam i = _gv_i_1;
			for (_gv_j_1 = 0; _gv_j_1 < taccel_pkg_SYS_DIM; _gv_j_1 = _gv_j_1 + 1) begin : GEN_ROUTE_COL
				localparam j = _gv_j_1;
				always @(*) begin
					if (_sv2v_0)
						;
					if (SYSTOLIC_ARCH_MODE == taccel_pkg_SYS_MODE_CHAINED) begin
						pe_a_in[((i * taccel_pkg_SYS_DIM) + j) * 8+:8] = (j == 0 ? a_edge_vec[i] : pe_a_out[((i * taccel_pkg_SYS_DIM) + (j - 1)) * 8+:8]);
						pe_b_in[((i * taccel_pkg_SYS_DIM) + j) * 8+:8] = (i == 0 ? b_edge_vec[j] : pe_b_out[(((i - 1) * taccel_pkg_SYS_DIM) + j) * 8+:8]);
					end
					else begin
						pe_a_in[((i * taccel_pkg_SYS_DIM) + j) * 8+:8] = a_vec[i];
						pe_b_in[((i * taccel_pkg_SYS_DIM) + j) * 8+:8] = b_vec[j];
					end
				end
			end
		end
		for (_gv_i_1 = 0; _gv_i_1 < taccel_pkg_SYS_DIM; _gv_i_1 = _gv_i_1 + 1) begin : GEN_ROW
			localparam i = _gv_i_1;
			for (_gv_j_1 = 0; _gv_j_1 < taccel_pkg_SYS_DIM; _gv_j_1 = _gv_j_1 + 1) begin : GEN_COL
				localparam j = _gv_j_1;
				systolic_pe u_pe(
					.clk(clk),
					.rst_n(rst_n),
					.en(step_en),
					.acc_clear(clear_acc),
					.a_in(pe_a_in[((i * taccel_pkg_SYS_DIM) + j) * 8+:8]),
					.b_in(pe_b_in[((i * taccel_pkg_SYS_DIM) + j) * 8+:8]),
					.a_out(pe_a_out[((i * taccel_pkg_SYS_DIM) + j) * 8+:8]),
					.b_out(pe_b_out[((i * taccel_pkg_SYS_DIM) + j) * 8+:8]),
					.acc(pe_acc[((i * taccel_pkg_SYS_DIM) + j) * 32+:32])
				);
				localparam signed [31:0] FLAT = ((i * taccel_pkg_SYS_DIM) + j) * 32;
				assign acc_flat[FLAT+:32] = pe_acc[((i * taccel_pkg_SYS_DIM) + j) * 32+:32];
			end
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule
module systolic_controller (
	clk,
	rst_n,
	dispatch,
	tile_m,
	tile_n,
	tile_k,
	src1_buf,
	src1_off,
	src2_buf,
	src2_off,
	dst_buf,
	dst_off,
	flags_accumulate,
	sys_busy,
	sram_a_en,
	sram_a_we,
	sram_a_buf,
	sram_a_row,
	sram_a_wdata,
	sram_a_rdata,
	sram_b_en,
	sram_b_buf,
	sram_b_row,
	sram_b_rdata
);
	reg _sv2v_0;
	localparam signed [31:0] taccel_pkg_SYS_MODE_CHAINED = 1;
	localparam signed [31:0] taccel_pkg_SYS_MODE_DEFAULT = taccel_pkg_SYS_MODE_CHAINED;
	parameter signed [31:0] SYSTOLIC_ARCH_MODE = taccel_pkg_SYS_MODE_DEFAULT;
	input wire clk;
	input wire rst_n;
	input wire dispatch;
	input wire [9:0] tile_m;
	input wire [9:0] tile_n;
	input wire [9:0] tile_k;
	input wire [1:0] src1_buf;
	input wire [15:0] src1_off;
	input wire [1:0] src2_buf;
	input wire [15:0] src2_off;
	input wire [1:0] dst_buf;
	input wire [15:0] dst_off;
	input wire flags_accumulate;
	output reg sys_busy;
	output reg sram_a_en;
	output reg sram_a_we;
	output reg [1:0] sram_a_buf;
	output reg [15:0] sram_a_row;
	output reg [127:0] sram_a_wdata;
	input wire [127:0] sram_a_rdata;
	output reg sram_b_en;
	output reg [1:0] sram_b_buf;
	output reg [15:0] sram_b_row;
	input wire [127:0] sram_b_rdata;
	reg [3:0] state;
	reg [1:0] src1_buf_q;
	reg [1:0] src2_buf_q;
	reg [1:0] dst_buf_q;
	reg [15:0] src1_off_q;
	reg [15:0] src2_off_q;
	reg [15:0] dst_off_q;
	reg flags_accumulate_q;
	reg [10:0] m_tiles_q;
	reg [10:0] n_tiles_q;
	reg [10:0] k_tiles_q;
	reg [10:0] mtile_q;
	reg [10:0] ntile_q;
	reg [10:0] ktile_q;
	reg [5:0] lane_q;
	reg [4:0] a_load_row_q;
	reg [4:0] drain_row_q;
	reg [1:0] drain_grp_q;
	reg [31:0] dst_clear_row_idx_q;
	reg [31:0] dst_clear_total_rows_q;
	localparam signed [31:0] taccel_pkg_SYS_DIM = 16;
	reg [((taccel_pkg_SYS_DIM * taccel_pkg_SYS_DIM) * 8) - 1:0] a_tile_scratch;
	reg [15:0] tile_drain_base_q;
	reg [15:0] drain_row_addr_q;
	reg step_en;
	reg clear_acc;
	reg inject_zero_data;
	reg [15:0] lane_row_idx;
	reg [127:0] a_row_data_q;
	reg [127:0] b_row_data_q;
	wire [((taccel_pkg_SYS_DIM * taccel_pkg_SYS_DIM) * 32) - 1:0] acc_flat;
	localparam signed [31:0] CHAIN_FLUSH_CYCLES = 30;
	localparam signed [31:0] CHAIN_TOTAL_STEPS = taccel_pkg_SYS_DIM + CHAIN_FLUSH_CYCLES;
	systolic_array #(.SYSTOLIC_ARCH_MODE(SYSTOLIC_ARCH_MODE)) u_array(
		.clk(clk),
		.rst_n(rst_n),
		.step_en(step_en),
		.clear_acc(clear_acc),
		.a_row_data(a_row_data_q),
		.b_row_data(b_row_data_q),
		.acc_flat(acc_flat)
	);
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	function automatic [31:0] acc_at;
		input reg [4:0] r;
		input reg [4:0] c;
		reg signed [31:0] idx;
		begin
			idx = ((sv2v_cast_32_signed(r) * taccel_pkg_SYS_DIM) + sv2v_cast_32_signed(c)) * 32;
			acc_at = acc_flat[idx+:32];
		end
	endfunction
	reg [31:0] src1_row_units;
	reg [31:0] src2_row_units;
	reg [31:0] src1_logical_row;
	reg [31:0] src2_logical_row;
	reg [31:0] src1_load_row_addr;
	reg [31:0] src2_stream_row_addr;
	reg [31:0] dispatch_m_tiles_w;
	reg [31:0] dispatch_n_tiles_w;
	reg [31:0] dispatch_clear_rows_w;
	reg needs_dst_preclear_w;
	always @(*) begin
		if (_sv2v_0)
			;
		src1_row_units = {21'h000000, k_tiles_q};
		src2_row_units = {21'h000000, n_tiles_q};
		dispatch_m_tiles_w = {22'h000000, tile_m} + 32'd1;
		dispatch_n_tiles_w = {22'h000000, tile_n} + 32'd1;
		dispatch_clear_rows_w = (dispatch_m_tiles_w * dispatch_n_tiles_w) << 6;
		needs_dst_preclear_w = !flags_accumulate && (dst_buf == 2'b10);
		src1_logical_row = ({21'h000000, mtile_q} << 4) + {27'h0000000, a_load_row_q};
		src2_logical_row = ({21'h000000, ktile_q} << 4) + {26'h0000000, lane_q};
		src1_load_row_addr = ({16'h0000, src1_off_q} + (src1_logical_row * src1_row_units)) + {21'h000000, ktile_q};
		src2_stream_row_addr = ({16'h0000, src2_off_q} + (src2_logical_row * src2_row_units)) + {21'h000000, ntile_q};
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 4'd0;
			src1_buf_q <= 2'b00;
			src2_buf_q <= 2'b00;
			dst_buf_q <= 2'b00;
			src1_off_q <= 16'h0000;
			src2_off_q <= 16'h0000;
			dst_off_q <= 16'h0000;
			flags_accumulate_q <= 1'b0;
			m_tiles_q <= 11'd0;
			n_tiles_q <= 11'd0;
			k_tiles_q <= 11'd0;
			mtile_q <= 11'd0;
			ntile_q <= 11'd0;
			ktile_q <= 11'd0;
			lane_q <= 6'd0;
			a_load_row_q <= 5'd0;
			drain_row_q <= 5'd0;
			drain_grp_q <= 2'd0;
			dst_clear_row_idx_q <= 32'd0;
			dst_clear_total_rows_q <= 32'd0;
			tile_drain_base_q <= 16'h0000;
			drain_row_addr_q <= 16'h0000;
			begin : sv2v_autoblock_1
				reg signed [31:0] row_idx;
				for (row_idx = 0; row_idx < taccel_pkg_SYS_DIM; row_idx = row_idx + 1)
					begin : sv2v_autoblock_2
						reg signed [31:0] col_idx;
						for (col_idx = 0; col_idx < taccel_pkg_SYS_DIM; col_idx = col_idx + 1)
							a_tile_scratch[((row_idx * taccel_pkg_SYS_DIM) + col_idx) * 8+:8] <= 8'h00;
					end
			end
		end
		else
			case (state)
				4'd0:
					if (dispatch) begin
						src1_buf_q <= src1_buf;
						src2_buf_q <= src2_buf;
						dst_buf_q <= dst_buf;
						src1_off_q <= src1_off;
						src2_off_q <= src2_off;
						dst_off_q <= dst_off;
						flags_accumulate_q <= flags_accumulate;
						tile_drain_base_q <= dst_off;
						m_tiles_q <= {1'b0, tile_m} + 11'd1;
						n_tiles_q <= {1'b0, tile_n} + 11'd1;
						k_tiles_q <= {1'b0, tile_k} + 11'd1;
						mtile_q <= 11'd0;
						ntile_q <= 11'd0;
						ktile_q <= 11'd0;
						lane_q <= 6'd0;
						a_load_row_q <= 5'd0;
						drain_row_q <= 5'd0;
						drain_grp_q <= 2'd0;
						dst_clear_row_idx_q <= 32'd0;
						dst_clear_total_rows_q <= dispatch_clear_rows_w;
						tile_drain_base_q <= dst_off;
						drain_row_addr_q <= dst_off;
						state <= (needs_dst_preclear_w ? 4'd8 : 4'd1);
					end
				4'd8: begin
					dst_clear_row_idx_q <= 32'd0;
					if (dst_clear_total_rows_q == 32'd0)
						state <= 4'd1;
					else
						state <= 4'd9;
				end
				4'd9:
					if ((dst_clear_row_idx_q + 32'd1) >= dst_clear_total_rows_q) begin
						dst_clear_row_idx_q <= 32'd0;
						state <= 4'd1;
					end
					else
						dst_clear_row_idx_q <= dst_clear_row_idx_q + 32'd1;
				4'd1: begin
					lane_q <= 6'd0;
					a_load_row_q <= 5'd0;
					state <= 4'd6;
				end
				4'd6: state <= 4'd7;
				4'd7: begin
					begin : sv2v_autoblock_3
						reg signed [31:0] col_idx;
						for (col_idx = 0; col_idx < taccel_pkg_SYS_DIM; col_idx = col_idx + 1)
							a_tile_scratch[((a_load_row_q[3:0] * taccel_pkg_SYS_DIM) + col_idx) * 8+:8] <= sram_b_rdata[col_idx * 8+:8];
					end
					if (a_load_row_q == 5'd15) begin
						lane_q <= 6'd0;
						state <= 4'd2;
					end
					else begin
						a_load_row_q <= a_load_row_q + 5'd1;
						state <= 4'd6;
					end
				end
				4'd2: state <= 4'd3;
				4'd3:
					if (sv2v_cast_32_signed(lane_q) == (SYSTOLIC_ARCH_MODE == taccel_pkg_SYS_MODE_CHAINED ? CHAIN_TOTAL_STEPS - 1 : 15)) begin
						lane_q <= 6'd0;
						if ((ktile_q + 11'd1) < k_tiles_q) begin
							ktile_q <= ktile_q + 11'd1;
							a_load_row_q <= 5'd0;
							state <= 4'd6;
						end
						else begin
							drain_row_q <= 5'd0;
							drain_grp_q <= 2'd0;
							state <= 4'd4;
						end
					end
					else begin
						lane_q <= lane_q + 6'd1;
						state <= 4'd2;
					end
				4'd4: begin
					drain_row_addr_q <= tile_drain_base_q + ({5'h00, ntile_q} << 2);
					state <= (flags_accumulate_q ? 4'd10 : 4'd5);
				end
				4'd10: state <= 4'd5;
				4'd5:
					if (drain_grp_q == 2'd3) begin
						drain_grp_q <= 2'd0;
						if (drain_row_q == 5'd15) begin
							ktile_q <= 11'd0;
							if ((ntile_q + 11'd1) < n_tiles_q) begin
								ntile_q <= ntile_q + 11'd1;
								state <= 4'd1;
							end
							else if ((mtile_q + 11'd1) < m_tiles_q) begin
								mtile_q <= mtile_q + 11'd1;
								ntile_q <= 11'd0;
								tile_drain_base_q <= tile_drain_base_q + ({5'h00, n_tiles_q} << 6);
								state <= 4'd1;
							end
							else
								state <= 4'd0;
						end
						else begin
							drain_row_q <= drain_row_q + 5'd1;
							drain_row_addr_q <= drain_row_addr_q + ({5'h00, n_tiles_q} << 2);
							state <= (flags_accumulate_q ? 4'd10 : 4'd5);
						end
					end
					else begin
						drain_grp_q <= drain_grp_q + 2'd1;
						state <= (flags_accumulate_q ? 4'd10 : 4'd5);
					end
				default: state <= 4'd0;
			endcase
	always @(dst_clear_row_idx_q[15:0] or dst_off_q or dst_buf_q or drain_grp_q or drain_row_q or acc_flat or drain_grp_q or drain_row_q or acc_flat or drain_grp_q or drain_row_q or acc_flat or drain_grp_q or drain_row_q or acc_flat or sram_a_rdata[127:96] or drain_grp_q or drain_row_q or acc_flat or sram_a_rdata[95:64] or drain_grp_q or drain_row_q or acc_flat or sram_a_rdata[63:32] or drain_grp_q or drain_row_q or acc_flat or sram_a_rdata[31:0] or drain_grp_q or drain_row_q or acc_flat or flags_accumulate_q or drain_grp_q or drain_row_addr_q or dst_buf_q or drain_grp_q or drain_row_addr_q or dst_buf_q or src2_stream_row_addr[15:0] or src2_buf_q or inject_zero_data or src1_load_row_addr[15:0] or src1_buf_q or state or state or src1_buf_q or src2_buf_q or sram_a_rdata or inject_zero_data or lane_q[3:0] or a_tile_scratch or lane_q or inject_zero_data or lane_q[5:0] or lane_q[5:0] or state or state or lane_q or state or _sv2v_0) begin
		if (_sv2v_0)
			;
		sys_busy = state != 4'd0;
		inject_zero_data = ((SYSTOLIC_ARCH_MODE == taccel_pkg_SYS_MODE_CHAINED) && (sv2v_cast_32_signed(lane_q) >= taccel_pkg_SYS_DIM)) && ((state == 4'd2) || (state == 4'd3));
		lane_row_idx = (SYSTOLIC_ARCH_MODE == taccel_pkg_SYS_MODE_CHAINED ? {10'h000, lane_q[5:0]} : {10'h000, lane_q[5:0]});
		a_row_data_q = 128'h00000000000000000000000000000000;
		begin : sv2v_autoblock_4
			reg signed [31:0] row_idx;
			for (row_idx = 0; row_idx < taccel_pkg_SYS_DIM; row_idx = row_idx + 1)
				if (!inject_zero_data && (lane_q < 6'd16))
					a_row_data_q[row_idx * 8+:8] = a_tile_scratch[((row_idx * taccel_pkg_SYS_DIM) + lane_q[3:0]) * 8+:8];
		end
		b_row_data_q = (inject_zero_data ? 128'h00000000000000000000000000000000 : sram_a_rdata);
		sram_a_en = 1'b0;
		sram_a_we = 1'b0;
		sram_a_buf = src2_buf_q;
		sram_a_row = 16'h0000;
		sram_a_wdata = 128'h00000000000000000000000000000000;
		sram_b_en = 1'b0;
		sram_b_buf = src1_buf_q;
		sram_b_row = 16'h0000;
		step_en = 1'b0;
		clear_acc = state == 4'd1;
		case (state)
			4'd6: begin
				sram_b_en = 1'b1;
				sram_b_buf = src1_buf_q;
				sram_b_row = src1_load_row_addr[15:0];
			end
			4'd2:
				if (inject_zero_data) begin
					sram_b_en = 1'b0;
					sram_a_en = 1'b0;
				end
				else begin
					sram_a_en = 1'b1;
					sram_a_we = 1'b0;
					sram_a_buf = src2_buf_q;
					sram_a_row = src2_stream_row_addr[15:0];
				end
			4'd3: step_en = 1'b1;
			4'd10: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b0;
				sram_a_buf = dst_buf_q;
				sram_a_row = drain_row_addr_q + {14'h0000, drain_grp_q};
			end
			4'd5: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = drain_row_addr_q + {14'h0000, drain_grp_q};
				if (flags_accumulate_q) begin
					sram_a_wdata[31:0] = $signed(acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00})) + $signed(sram_a_rdata[31:0]);
					sram_a_wdata[63:32] = $signed(acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd1)) + $signed(sram_a_rdata[63:32]);
					sram_a_wdata[95:64] = $signed(acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd2)) + $signed(sram_a_rdata[95:64]);
					sram_a_wdata[127:96] = $signed(acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd3)) + $signed(sram_a_rdata[127:96]);
				end
				else begin
					sram_a_wdata[31:0] = acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00});
					sram_a_wdata[63:32] = acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd1);
					sram_a_wdata[95:64] = acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd2);
					sram_a_wdata[127:96] = acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd3);
				end
			end
			4'd9: begin
				sram_a_en = 1'b1;
				sram_a_we = 1'b1;
				sram_a_buf = dst_buf_q;
				sram_a_row = dst_off_q + dst_clear_row_idx_q[15:0];
				sram_a_wdata = 128'h00000000000000000000000000000000;
			end
			default:
				;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
module dma_engine (
	clk,
	rst_n,
	dispatch,
	is_store,
	buf_id,
	sram_off,
	xfer_len,
	base_addr,
	dram_off,
	dma_busy,
	dma_rd_busy,
	dma_fault,
	dma_fault_code,
	sram_en,
	sram_we,
	sram_buf,
	sram_row,
	sram_wdata,
	sram_rdata,
	sram_fault,
	dma_ar_addr,
	dma_ar_len,
	dma_ar_valid,
	dma_ar_ready,
	dma_r_data,
	dma_r_resp,
	dma_r_valid,
	dma_r_last,
	dma_r_ready,
	dma_aw_addr,
	dma_aw_len,
	dma_aw_valid,
	dma_aw_ready,
	dma_w_data,
	dma_w_strb,
	dma_w_valid,
	dma_w_last,
	dma_w_ready,
	dma_b_resp,
	dma_b_valid,
	dma_b_ready
);
	reg _sv2v_0;
	parameter signed [31:0] DRAM_SIZE = 16777216;
	input wire clk;
	input wire rst_n;
	input wire dispatch;
	input wire is_store;
	input wire [1:0] buf_id;
	input wire [15:0] sram_off;
	input wire [15:0] xfer_len;
	input wire [55:0] base_addr;
	input wire [15:0] dram_off;
	output reg dma_busy;
	output reg dma_rd_busy;
	output reg dma_fault;
	output reg [3:0] dma_fault_code;
	output reg sram_en;
	output reg sram_we;
	output reg [1:0] sram_buf;
	output reg [15:0] sram_row;
	output reg [127:0] sram_wdata;
	input wire [127:0] sram_rdata;
	input wire sram_fault;
	localparam signed [31:0] taccel_pkg_AXI_ADDR_W = 56;
	output reg [55:0] dma_ar_addr;
	output reg [7:0] dma_ar_len;
	output reg dma_ar_valid;
	input wire dma_ar_ready;
	localparam signed [31:0] taccel_pkg_AXI_DATA_W = 128;
	input wire [127:0] dma_r_data;
	input wire [1:0] dma_r_resp;
	input wire dma_r_valid;
	input wire dma_r_last;
	output reg dma_r_ready;
	output reg [55:0] dma_aw_addr;
	output reg [7:0] dma_aw_len;
	output reg dma_aw_valid;
	input wire dma_aw_ready;
	output reg [127:0] dma_w_data;
	output reg [15:0] dma_w_strb;
	output reg dma_w_valid;
	output reg dma_w_last;
	input wire dma_w_ready;
	input wire [1:0] dma_b_resp;
	input wire dma_b_valid;
	output reg dma_b_ready;
	reg [2:0] state;
	reg is_store_q;
	reg [1:0] buf_id_q;
	reg [15:0] curr_sram_row_q;
	reg [15:0] beats_remaining_q;
	reg [15:0] burst_beats_q;
	reg [15:0] burst_beat_idx_q;
	reg [55:0] curr_dram_addr_q;
	reg [3:0] fault_code_r;
	wire [55:0] burst_bytes_w;
	wire [15:0] remaining_after_burst_w;
	wire [15:0] next_burst_beats_w;
	wire [55:0] dram_addr_after_burst_w;
	wire [15:0] sram_row_after_burst_w;
	wire burst_last_beat_w;
	wire transfer_last_burst_w;
	wire load_beat_fault_w;
	wire load_beat_accept_w;
	wire [56:0] dispatch_dram_byte_addr_w;
	wire [56:0] dispatch_dram_end_w;
	wire [15:0] dispatch_buf_rows_w;
	wire [16:0] dispatch_sram_end_w;
	wire dispatch_dram_oob_w;
	wire dispatch_sram_oob_w;
	localparam signed [31:0] taccel_pkg_ABUF_ROWS = 8192;
	localparam signed [31:0] taccel_pkg_ACCUM_ROWS = 4096;
	localparam signed [31:0] taccel_pkg_WBUF_ROWS = 16384;
	function automatic signed [15:0] sv2v_cast_16_signed;
		input reg signed [15:0] inp;
		sv2v_cast_16_signed = inp;
	endfunction
	function automatic [15:0] buf_rows;
		input reg [1:0] bid;
		case (bid)
			2'b00: buf_rows = sv2v_cast_16_signed(taccel_pkg_ABUF_ROWS);
			2'b01: buf_rows = sv2v_cast_16_signed(taccel_pkg_WBUF_ROWS);
			2'b10: buf_rows = sv2v_cast_16_signed(taccel_pkg_ACCUM_ROWS);
			default: buf_rows = 16'h0000;
		endcase
	endfunction
	function automatic [15:0] burst_beats;
		input reg [15:0] remaining;
		if (remaining > 16'd256)
			burst_beats = 16'd256;
		else
			burst_beats = remaining;
	endfunction
	assign burst_bytes_w = {36'h000000000, burst_beats_q, 4'b0000};
	assign remaining_after_burst_w = beats_remaining_q - burst_beats_q;
	assign next_burst_beats_w = burst_beats(remaining_after_burst_w);
	assign dram_addr_after_burst_w = curr_dram_addr_q + burst_bytes_w;
	assign sram_row_after_burst_w = curr_sram_row_q + burst_beats_q;
	assign burst_last_beat_w = burst_beat_idx_q == (burst_beats_q - 16'h0001);
	assign transfer_last_burst_w = beats_remaining_q == burst_beats_q;
	assign load_beat_fault_w = (sram_fault | (dma_r_resp != 2'b00)) | (dma_r_last != burst_last_beat_w);
	assign load_beat_accept_w = (dma_r_valid & (dma_r_resp == 2'b00)) & (dma_r_last == burst_last_beat_w);
	assign dispatch_dram_byte_addr_w = {1'b0, base_addr} + {37'h0000000000, dram_off, 4'b0000};
	assign dispatch_dram_end_w = dispatch_dram_byte_addr_w + {37'h0000000000, xfer_len, 4'b0000};
	assign dispatch_buf_rows_w = buf_rows(buf_id);
	assign dispatch_sram_end_w = {1'b0, sram_off} + {1'b0, xfer_len};
	function automatic signed [56:0] sv2v_cast_57_signed;
		input reg signed [56:0] inp;
		sv2v_cast_57_signed = inp;
	endfunction
	assign dispatch_dram_oob_w = dispatch_dram_end_w > sv2v_cast_57_signed(DRAM_SIZE);
	assign dispatch_sram_oob_w = (dispatch_buf_rows_w == 16'h0000) | (xfer_len == 16'h0000 ? sram_off >= dispatch_buf_rows_w : dispatch_sram_end_w > {1'b0, dispatch_buf_rows_w});
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			state <= 3'd0;
			is_store_q <= 1'b0;
			buf_id_q <= 2'b00;
			curr_sram_row_q <= 16'h0000;
			beats_remaining_q <= 16'h0000;
			burst_beats_q <= 16'h0000;
			burst_beat_idx_q <= 16'h0000;
			curr_dram_addr_q <= 56'h00000000000000;
			fault_code_r <= 4'h0;
		end
		else
			case (state)
				3'd0:
					if (dispatch) begin
						is_store_q <= is_store;
						buf_id_q <= buf_id;
						curr_sram_row_q <= sram_off;
						beats_remaining_q <= xfer_len;
						burst_beats_q <= burst_beats(xfer_len);
						burst_beat_idx_q <= 16'h0000;
						curr_dram_addr_q <= dispatch_dram_byte_addr_w[55:0];
						if (dispatch_dram_oob_w) begin
							fault_code_r <= 4'h2;
							state <= 3'd7;
						end
						else if (dispatch_sram_oob_w) begin
							fault_code_r <= 4'h3;
							state <= 3'd7;
						end
						else if (xfer_len != 16'h0000)
							state <= (is_store ? 3'd3 : 3'd1);
					end
				3'd1:
					if (dma_ar_ready)
						state <= 3'd2;
				3'd2:
					if (dma_r_valid) begin
						if (sram_fault) begin
							fault_code_r <= 4'h3;
							state <= 3'd7;
						end
						else if (dma_r_resp != 2'b00) begin
							fault_code_r <= 4'h2;
							state <= 3'd7;
						end
						else if (dma_r_last != burst_last_beat_w) begin
							fault_code_r <= 4'h2;
							state <= 3'd7;
						end
						else if (burst_last_beat_w) begin
							if (transfer_last_burst_w)
								state <= 3'd0;
							else begin
								curr_dram_addr_q <= dram_addr_after_burst_w;
								curr_sram_row_q <= sram_row_after_burst_w;
								beats_remaining_q <= remaining_after_burst_w;
								burst_beats_q <= next_burst_beats_w;
								burst_beat_idx_q <= 16'h0000;
								state <= 3'd1;
							end
						end
						else
							burst_beat_idx_q <= burst_beat_idx_q + 16'h0001;
					end
				3'd3:
					if (dma_aw_ready) begin
						burst_beat_idx_q <= 16'h0000;
						state <= 3'd4;
					end
				3'd4:
					if (sram_fault) begin
						fault_code_r <= 4'h3;
						state <= 3'd7;
					end
					else
						state <= 3'd5;
				3'd5:
					if (dma_w_ready) begin
						if (burst_last_beat_w)
							state <= 3'd6;
						else begin
							burst_beat_idx_q <= burst_beat_idx_q + 16'h0001;
							state <= 3'd4;
						end
					end
				3'd6:
					if (dma_b_valid) begin
						if (dma_b_resp != 2'b00) begin
							fault_code_r <= 4'h2;
							state <= 3'd7;
						end
						else if (transfer_last_burst_w)
							state <= 3'd0;
						else begin
							curr_dram_addr_q <= dram_addr_after_burst_w;
							curr_sram_row_q <= sram_row_after_burst_w;
							beats_remaining_q <= remaining_after_burst_w;
							burst_beats_q <= next_burst_beats_w;
							burst_beat_idx_q <= 16'h0000;
							state <= 3'd3;
						end
					end
				3'd7:
					;
				default: state <= 3'd0;
			endcase
	function automatic [7:0] sv2v_cast_8;
		input reg [7:0] inp;
		sv2v_cast_8 = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		dma_busy = state != 3'd0;
		dma_rd_busy = (state == 3'd1) || (state == 3'd2);
		dma_fault = state == 3'd7;
		dma_fault_code = fault_code_r;
		dma_ar_addr = curr_dram_addr_q;
		dma_ar_len = (burst_beats_q == 16'h0000 ? 8'h00 : sv2v_cast_8(burst_beats_q - 16'h0001));
		dma_ar_valid = 1'b0;
		dma_r_ready = 1'b0;
		dma_aw_addr = curr_dram_addr_q;
		dma_aw_len = (burst_beats_q == 16'h0000 ? 8'h00 : sv2v_cast_8(burst_beats_q - 16'h0001));
		dma_aw_valid = 1'b0;
		dma_w_data = 128'h00000000000000000000000000000000;
		dma_w_strb = 16'hffff;
		dma_w_valid = 1'b0;
		dma_w_last = 1'b0;
		dma_b_ready = 1'b0;
		sram_en = 1'b0;
		sram_we = 1'b0;
		sram_buf = buf_id_q;
		sram_row = curr_sram_row_q + burst_beat_idx_q;
		sram_wdata = dma_r_data;
		case (state)
			3'd1: dma_ar_valid = 1'b1;
			3'd2: begin
				dma_r_ready = 1'b1;
				if (load_beat_accept_w) begin
					sram_en = 1'b1;
					sram_we = 1'b1;
					sram_wdata = dma_r_data;
				end
			end
			3'd3: dma_aw_valid = 1'b1;
			3'd4: begin
				sram_en = 1'b1;
				sram_we = 1'b0;
			end
			3'd5: begin
				dma_w_valid = 1'b1;
				dma_w_data = sram_rdata;
				dma_w_last = burst_last_beat_w;
			end
			3'd6: dma_b_ready = 1'b1;
			default:
				;
		endcase
	end
	initial _sv2v_0 = 0;
endmodule
module taccel_top (
	clk,
	rst_n,
	start,
	done,
	fault,
	fault_code,
	m_axi_ar_addr,
	m_axi_ar_valid,
	m_axi_ar_len,
	m_axi_ar_size,
	m_axi_ar_burst,
	m_axi_ar_ready,
	m_axi_r_data,
	m_axi_r_resp,
	m_axi_r_valid,
	m_axi_r_last,
	m_axi_r_ready,
	m_axi_aw_addr,
	m_axi_aw_len,
	m_axi_aw_size,
	m_axi_aw_burst,
	m_axi_aw_valid,
	m_axi_aw_ready,
	m_axi_w_data,
	m_axi_w_strb,
	m_axi_w_valid,
	m_axi_w_last,
	m_axi_w_ready,
	m_axi_b_resp,
	m_axi_b_valid,
	m_axi_b_ready
);
	reg _sv2v_0;
	localparam signed [31:0] taccel_pkg_SYS_MODE_CHAINED = 1;
	localparam signed [31:0] taccel_pkg_SYS_MODE_DEFAULT = taccel_pkg_SYS_MODE_CHAINED;
	parameter signed [31:0] SYSTOLIC_ARCH_MODE = taccel_pkg_SYS_MODE_DEFAULT;
	parameter signed [31:0] DRAM_SIZE = 16777216;
	parameter signed [31:0] SFU_SYNTH_MODE = 0;
	parameter signed [31:0] HELPER_SYNTH_MODE = 0;
	input wire clk;
	input wire rst_n;
	input wire start;
	output wire done;
	output wire fault;
	output wire [3:0] fault_code;
	localparam signed [31:0] taccel_pkg_AXI_ADDR_W = 56;
	output wire [55:0] m_axi_ar_addr;
	output wire m_axi_ar_valid;
	output wire [7:0] m_axi_ar_len;
	output wire [2:0] m_axi_ar_size;
	output wire [1:0] m_axi_ar_burst;
	input wire m_axi_ar_ready;
	localparam signed [31:0] taccel_pkg_AXI_DATA_W = 128;
	input wire [127:0] m_axi_r_data;
	input wire [1:0] m_axi_r_resp;
	input wire m_axi_r_valid;
	input wire m_axi_r_last;
	output wire m_axi_r_ready;
	output wire [55:0] m_axi_aw_addr;
	output wire [7:0] m_axi_aw_len;
	output wire [2:0] m_axi_aw_size;
	output wire [1:0] m_axi_aw_burst;
	output wire m_axi_aw_valid;
	input wire m_axi_aw_ready;
	output wire [127:0] m_axi_w_data;
	output wire [15:0] m_axi_w_strb;
	output wire m_axi_w_valid;
	output wire m_axi_w_last;
	input wire m_axi_w_ready;
	input wire [1:0] m_axi_b_resp;
	input wire m_axi_b_valid;
	output wire m_axi_b_ready;
	wire [55:0] fetch_ar_addr;
	wire fetch_ar_valid;
	wire fetch_ar_ready;
	wire fetch_r_valid;
	wire fetch_r_ready;
	wire [55:0] dma_ar_addr;
	wire [7:0] dma_ar_len;
	wire dma_ar_valid;
	wire dma_ar_ready;
	wire dma_r_valid;
	wire dma_r_ready;
	reg dma_r_owner_q;
	reg rd_inflight_q;
	reg prefer_fetch_after_dma_q;
	reg select_dma_ar_w;
	reg select_fetch_ar_w;
	wire dma_rd_busy;
	always @(*) begin
		if (_sv2v_0)
			;
		select_dma_ar_w = 1'b0;
		select_fetch_ar_w = 1'b0;
		if (!rd_inflight_q) begin
			if (dma_ar_valid && fetch_ar_valid) begin
				if (prefer_fetch_after_dma_q)
					select_fetch_ar_w = 1'b1;
				else
					select_dma_ar_w = 1'b1;
			end
			else if (dma_ar_valid)
				select_dma_ar_w = 1'b1;
			else if (fetch_ar_valid)
				select_fetch_ar_w = 1'b1;
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			dma_r_owner_q <= 1'b0;
			rd_inflight_q <= 1'b0;
			prefer_fetch_after_dma_q <= 1'b0;
		end
		else begin
			if (m_axi_ar_valid && m_axi_ar_ready) begin
				dma_r_owner_q <= select_dma_ar_w;
				rd_inflight_q <= 1'b1;
				if (select_fetch_ar_w)
					prefer_fetch_after_dma_q <= 1'b0;
			end
			if (((rd_inflight_q && m_axi_r_valid) && m_axi_r_ready) && m_axi_r_last) begin
				rd_inflight_q <= 1'b0;
				if (dma_r_owner_q)
					prefer_fetch_after_dma_q <= 1'b1;
			end
		end
	assign m_axi_ar_addr = (select_dma_ar_w ? dma_ar_addr : fetch_ar_addr);
	assign m_axi_ar_len = (select_dma_ar_w ? dma_ar_len : 8'h00);
	assign m_axi_ar_size = 3'b100;
	assign m_axi_ar_burst = 2'b01;
	assign m_axi_ar_valid = select_dma_ar_w | select_fetch_ar_w;
	assign dma_ar_ready = (m_axi_ar_ready & ~rd_inflight_q) & (~fetch_ar_valid | ~prefer_fetch_after_dma_q);
	assign fetch_ar_ready = (m_axi_ar_ready & ~rd_inflight_q) & (~dma_ar_valid | prefer_fetch_after_dma_q);
	assign dma_r_valid = (m_axi_r_valid & rd_inflight_q) & dma_r_owner_q;
	assign fetch_r_valid = (m_axi_r_valid & rd_inflight_q) & ~dma_r_owner_q;
	assign m_axi_r_ready = (rd_inflight_q ? (dma_r_owner_q ? dma_r_ready : fetch_r_ready) : 1'b0);
	assign m_axi_aw_size = 3'b100;
	assign m_axi_aw_burst = 2'b01;
	wire [55:0] pc;
	wire fetch_req;
	wire insn_valid_w;
	wire [63:0] insn_data_w;
	reg [63:0] insn_data_q;
	wire [286:0] insn;
	wire fetch_fault_w;
	wire [3:0] fetch_fault_code_w;
	wire scale_we;
	wire [3:0] scale_waddr;
	wire [15:0] scale_wdata;
	wire addr_lo_we;
	wire addr_hi_we;
	wire [1:0] addr_wsel;
	wire [27:0] addr_imm28;
	wire tile_we;
	wire [9:0] tile_m_in;
	wire [9:0] tile_n_in;
	wire [9:0] tile_k_in;
	wire [9:0] tile_m;
	wire [9:0] tile_n;
	wire [9:0] tile_k;
	wire tile_valid;
	wire attn_we;
	wire [11:0] attn_query_row_base_in;
	wire [11:0] attn_valid_kv_len_in;
	wire [1:0] attn_mode_in;
	wire attn_valid;
	wire [11:0] attn_query_row_base;
	wire [11:0] attn_valid_kv_len;
	wire [1:0] attn_mode;
	wire [15:0] scale_rdata0;
	wire [15:0] scale_rdata1;
	wire [15:0] scale_rdata2;
	wire [15:0] scale_rdata3;
	wire [55:0] addr_rdata;
	wire [1:0] helper_src1_buf_w;
	wire [1:0] helper_src2_buf_w;
	wire [1:0] helper_dst_buf_w;
	wire [15:0] helper_src1_off_w;
	wire [15:0] helper_src2_off_w;
	wire [15:0] helper_dst_off_w;
	wire dma_dispatch;
	wire sys_dispatch;
	wire sfu_dispatch;
	wire helper_dispatch;
	wire dma_is_store;
	assign dma_is_store = insn_data_q[63:59] == 5'h08;
	assign helper_src1_buf_w = (insn[286-:5] == 5'h09 ? insn[169-:2] : insn[280-:2]);
	assign helper_src1_off_w = (insn[286-:5] == 5'h09 ? insn[167-:16] : insn[278-:16]);
	assign helper_src2_buf_w = (insn[286-:5] == 5'h09 ? 2'b00 : insn[262-:2]);
	assign helper_src2_off_w = (insn[286-:5] == 5'h09 ? 16'h0000 : insn[260-:16]);
	assign helper_dst_buf_w = (insn[286-:5] == 5'h09 ? insn[151-:2] : insn[244-:2]);
	assign helper_dst_off_w = (insn[286-:5] == 5'h09 ? insn[149-:16] : insn[242-:16]);
	wire dma_busy;
	wire dma_fault_w;
	wire [3:0] dma_fault_code_w;
	wire dma_sram_fault_w;
	wire ext_fault_w;
	reg [3:0] ext_fault_code_w;
	wire helper_fault_w;
	wire [3:0] helper_fault_code_w;
	wire sfu_fault_w;
	wire [3:0] sfu_fault_code_w;
	wire sys_sram_fault_now;
	reg sys_sram_fault_latched;
	wire sys_busy;
	wire sfu_busy;
	wire helper_busy;
	wire obs_retire_pulse_w;
	wire [55:0] obs_retire_pc_w;
	wire [4:0] obs_retire_opcode_w;
	wire obs_ctrl_fault_pulse_w;
	wire [3:0] obs_ctrl_fault_code_w;
	wire [55:0] obs_ctrl_fault_pc_w;
	wire [4:0] obs_ctrl_fault_opcode_w;
	wire obs_sync_wait_dma_w;
	wire obs_sync_wait_sys_w;
	wire obs_sync_wait_sfu_w;
	reg obs_run_active_q;
	reg [63:0] obs_cycle_count_q;
	reg [63:0] obs_retired_insn_count_q;
	reg [63:0] obs_sync_wait_dma_cycles_q;
	reg [63:0] obs_sync_wait_sys_cycles_q;
	reg [63:0] obs_sync_wait_sfu_cycles_q;
	reg [63:0] obs_dma_burst_count_q;
	reg [63:0] obs_dma_beat_count_q;
	reg [63:0] obs_helper_busy_cycles_q;
	reg [63:0] obs_sfu_busy_cycles_q;
	reg [63:0] obs_sys_busy_cycles_q;
	reg obs_fault_valid_q;
	reg [55:0] obs_fault_pc_q;
	reg [4:0] obs_fault_opcode_q;
	reg obs_fault_opcode_valid_q;
	reg [2:0] obs_fault_source_q;
	reg [3:0] obs_fault_code_q;
	reg obs_forbidden_overlap_violation_q;
	reg [55:0] obs_dma_issue_pc_q;
	reg [4:0] obs_dma_issue_opcode_q;
	reg [55:0] obs_sys_issue_pc_q;
	reg [4:0] obs_sys_issue_opcode_q;
	reg [55:0] obs_helper_issue_pc_q;
	reg [4:0] obs_helper_issue_opcode_q;
	reg [55:0] obs_sfu_issue_pc_q;
	reg [4:0] obs_sfu_issue_opcode_q;
	wire obs_dma_burst_fire_w;
	wire obs_dma_beat_fire_w;
	wire obs_terminal_event_w;
	wire helper_sram_a_en;
	wire helper_sram_a_we;
	wire [1:0] helper_sram_a_buf;
	wire [15:0] helper_sram_a_row;
	wire [127:0] helper_sram_a_wdata;
	wire [127:0] helper_sram_a_rdata;
	wire helper_sram_b_en;
	wire [1:0] helper_sram_b_buf;
	wire [15:0] helper_sram_b_row;
	wire [127:0] helper_sram_b_rdata;
	wire sfu_sram_a_en;
	wire sfu_sram_a_we;
	wire [1:0] sfu_sram_a_buf;
	wire [15:0] sfu_sram_a_row;
	wire [127:0] sfu_sram_a_wdata;
	wire sfu_scale_we_w;
	wire [3:0] sfu_scale_waddr_w;
	wire [15:0] sfu_scale_wdata_w;
	wire sfu_sram_b_en;
	wire [1:0] sfu_sram_b_buf;
	wire [15:0] sfu_sram_b_row;
	wire [127:0] sfu_sram_b_rdata;
	wire dma_sram_en;
	wire dma_sram_we;
	wire [1:0] dma_sram_buf;
	wire [15:0] dma_sram_row;
	wire [127:0] dma_sram_wdata;
	wire [127:0] dma_sram_rdata;
	wire sys_sram_a_en;
	wire sys_sram_a_we;
	wire [1:0] sys_sram_a_buf;
	wire [15:0] sys_sram_a_row;
	wire [127:0] sys_sram_a_wdata;
	wire [127:0] sys_sram_a_rdata;
	wire sys_sram_b_en;
	wire [1:0] sys_sram_b_buf;
	wire [15:0] sys_sram_b_row;
	wire [127:0] sys_sram_b_rdata;
	wire sram_a_en;
	wire sram_a_we;
	wire [1:0] sram_a_buf;
	wire [15:0] sram_a_row;
	wire [127:0] sram_a_wdata;
	wire [127:0] sram_a_rdata;
	wire sram_a_fault;
	wire sram_b_en;
	wire [1:0] sram_b_buf;
	wire [15:0] sram_b_row;
	wire [127:0] sram_b_rdata;
	wire sram_b_fault;
	assign sram_a_en = (helper_sram_a_en ? helper_sram_a_en : (sfu_sram_a_en ? sfu_sram_a_en : (dma_sram_en ? dma_sram_en : sys_sram_a_en)));
	assign sram_a_we = (helper_sram_a_en ? helper_sram_a_we : (sfu_sram_a_en ? sfu_sram_a_we : (dma_sram_en ? dma_sram_we : sys_sram_a_we)));
	assign sram_a_buf = (helper_sram_a_en ? helper_sram_a_buf : (sfu_sram_a_en ? sfu_sram_a_buf : (dma_sram_en ? dma_sram_buf : sys_sram_a_buf)));
	assign sram_a_row = (helper_sram_a_en ? helper_sram_a_row : (sfu_sram_a_en ? sfu_sram_a_row : (dma_sram_en ? dma_sram_row : sys_sram_a_row)));
	assign sram_a_wdata = (helper_sram_a_en ? helper_sram_a_wdata : (sfu_sram_a_en ? sfu_sram_a_wdata : (dma_sram_en ? dma_sram_wdata : sys_sram_a_wdata)));
	assign sram_b_en = (helper_sram_b_en ? helper_sram_b_en : (sfu_sram_b_en ? sfu_sram_b_en : sys_sram_b_en));
	assign sram_b_buf = (helper_sram_b_en ? helper_sram_b_buf : (sfu_sram_b_en ? sfu_sram_b_buf : sys_sram_b_buf));
	assign sram_b_row = (helper_sram_b_en ? helper_sram_b_row : (sfu_sram_b_en ? sfu_sram_b_row : sys_sram_b_row));
	assign helper_sram_a_rdata = sram_a_rdata;
	assign dma_sram_rdata = sram_a_rdata;
	assign sys_sram_a_rdata = sram_a_rdata;
	assign helper_sram_b_rdata = sram_b_rdata;
	assign sfu_sram_b_rdata = sram_b_rdata;
	assign sys_sram_b_rdata = sram_b_rdata;
	assign dma_sram_fault_w = dma_sram_en & sram_a_fault;
	assign sys_sram_fault_now = ((sys_sram_b_en & ~helper_sram_b_en) & sram_b_fault) | ((((sys_sram_a_en & ~helper_sram_a_en) & ~sfu_sram_a_en) & ~dma_sram_en) & sram_a_fault);
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			sys_sram_fault_latched <= 1'b0;
		else if (sys_sram_fault_now)
			sys_sram_fault_latched <= 1'b1;
	assign ext_fault_w = ((((fetch_fault_w | dma_fault_w) | helper_fault_w) | sfu_fault_w) | sys_sram_fault_now) | sys_sram_fault_latched;
	always @(*) begin
		if (_sv2v_0)
			;
		if (fetch_fault_w)
			ext_fault_code_w = fetch_fault_code_w;
		else if (dma_fault_w)
			ext_fault_code_w = dma_fault_code_w;
		else if (helper_fault_w)
			ext_fault_code_w = helper_fault_code_w;
		else if (sfu_fault_w)
			ext_fault_code_w = sfu_fault_code_w;
		else if (sys_sram_fault_now || sys_sram_fault_latched)
			ext_fault_code_w = 4'h3;
		else
			ext_fault_code_w = 4'h0;
	end
	assign obs_dma_burst_fire_w = (dma_ar_valid && dma_ar_ready) || (m_axi_aw_valid && m_axi_aw_ready);
	assign obs_dma_beat_fire_w = (dma_r_valid && dma_r_ready) || (m_axi_w_valid && m_axi_w_ready);
	assign obs_terminal_event_w = ((((((obs_ctrl_fault_pulse_w | fetch_fault_w) | dma_fault_w) | helper_fault_w) | sfu_fault_w) | sys_sram_fault_now) | sys_sram_fault_latched) | (obs_retire_pulse_w && (obs_retire_opcode_w == 5'h01));
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			obs_run_active_q <= 1'b0;
			obs_cycle_count_q <= 64'h0000000000000000;
			obs_retired_insn_count_q <= 64'h0000000000000000;
			obs_sync_wait_dma_cycles_q <= 64'h0000000000000000;
			obs_sync_wait_sys_cycles_q <= 64'h0000000000000000;
			obs_sync_wait_sfu_cycles_q <= 64'h0000000000000000;
			obs_dma_burst_count_q <= 64'h0000000000000000;
			obs_dma_beat_count_q <= 64'h0000000000000000;
			obs_helper_busy_cycles_q <= 64'h0000000000000000;
			obs_sfu_busy_cycles_q <= 64'h0000000000000000;
			obs_sys_busy_cycles_q <= 64'h0000000000000000;
			obs_fault_valid_q <= 1'b0;
			obs_fault_pc_q <= 56'h00000000000000;
			obs_fault_opcode_q <= 5'h00;
			obs_fault_opcode_valid_q <= 1'b0;
			obs_fault_source_q <= 3'd0;
			obs_fault_code_q <= 4'h0;
			obs_forbidden_overlap_violation_q <= 1'b0;
			obs_dma_issue_pc_q <= 56'h00000000000000;
			obs_dma_issue_opcode_q <= 5'h00;
			obs_sys_issue_pc_q <= 56'h00000000000000;
			obs_sys_issue_opcode_q <= 5'h00;
			obs_helper_issue_pc_q <= 56'h00000000000000;
			obs_helper_issue_opcode_q <= 5'h00;
			obs_sfu_issue_pc_q <= 56'h00000000000000;
			obs_sfu_issue_opcode_q <= 5'h00;
		end
		else if (((start && !obs_run_active_q) && !done) && !fault) begin
			obs_run_active_q <= 1'b1;
			obs_cycle_count_q <= 64'h0000000000000000;
			obs_retired_insn_count_q <= 64'h0000000000000000;
			obs_sync_wait_dma_cycles_q <= 64'h0000000000000000;
			obs_sync_wait_sys_cycles_q <= 64'h0000000000000000;
			obs_sync_wait_sfu_cycles_q <= 64'h0000000000000000;
			obs_dma_burst_count_q <= 64'h0000000000000000;
			obs_dma_beat_count_q <= 64'h0000000000000000;
			obs_helper_busy_cycles_q <= 64'h0000000000000000;
			obs_sfu_busy_cycles_q <= 64'h0000000000000000;
			obs_sys_busy_cycles_q <= 64'h0000000000000000;
			obs_fault_valid_q <= 1'b0;
			obs_fault_pc_q <= 56'h00000000000000;
			obs_fault_opcode_q <= 5'h00;
			obs_fault_opcode_valid_q <= 1'b0;
			obs_fault_source_q <= 3'd0;
			obs_fault_code_q <= 4'h0;
			obs_forbidden_overlap_violation_q <= 1'b0;
			obs_dma_issue_pc_q <= 56'h00000000000000;
			obs_dma_issue_opcode_q <= 5'h00;
			obs_sys_issue_pc_q <= 56'h00000000000000;
			obs_sys_issue_opcode_q <= 5'h00;
			obs_helper_issue_pc_q <= 56'h00000000000000;
			obs_helper_issue_opcode_q <= 5'h00;
			obs_sfu_issue_pc_q <= 56'h00000000000000;
			obs_sfu_issue_opcode_q <= 5'h00;
		end
		else begin
			if (obs_run_active_q) begin
				obs_cycle_count_q <= obs_cycle_count_q + 64'd1;
				if (obs_sync_wait_dma_w)
					obs_sync_wait_dma_cycles_q <= obs_sync_wait_dma_cycles_q + 64'd1;
				if (obs_sync_wait_sys_w)
					obs_sync_wait_sys_cycles_q <= obs_sync_wait_sys_cycles_q + 64'd1;
				if (obs_sync_wait_sfu_w)
					obs_sync_wait_sfu_cycles_q <= obs_sync_wait_sfu_cycles_q + 64'd1;
				if (helper_busy)
					obs_helper_busy_cycles_q <= obs_helper_busy_cycles_q + 64'd1;
				if (sfu_busy)
					obs_sfu_busy_cycles_q <= obs_sfu_busy_cycles_q + 64'd1;
				if (sys_busy)
					obs_sys_busy_cycles_q <= obs_sys_busy_cycles_q + 64'd1;
			end
			if (obs_retire_pulse_w)
				obs_retired_insn_count_q <= obs_retired_insn_count_q + 64'd1;
			if (obs_dma_burst_fire_w)
				obs_dma_burst_count_q <= obs_dma_burst_count_q + 64'd1;
			if (obs_dma_beat_fire_w)
				obs_dma_beat_count_q <= obs_dma_beat_count_q + 64'd1;
			if (dma_dispatch) begin
				obs_dma_issue_pc_q <= pc;
				obs_dma_issue_opcode_q <= insn[286-:5];
			end
			if (sys_dispatch) begin
				obs_sys_issue_pc_q <= pc;
				obs_sys_issue_opcode_q <= insn[286-:5];
			end
			if (helper_dispatch) begin
				obs_helper_issue_pc_q <= pc;
				obs_helper_issue_opcode_q <= insn[286-:5];
			end
			if (sfu_dispatch) begin
				obs_sfu_issue_pc_q <= pc;
				obs_sfu_issue_opcode_q <= insn[286-:5];
			end
			if (helper_busy && ((dma_busy || sys_busy) || sfu_busy))
				obs_forbidden_overlap_violation_q <= 1'b1;
			if (sfu_busy && ((dma_busy || sys_busy) || helper_busy))
				obs_forbidden_overlap_violation_q <= 1'b1;
			if (scale_we && sfu_scale_we_w)
				obs_forbidden_overlap_violation_q <= 1'b1;
			if (!obs_fault_valid_q) begin
				if (obs_ctrl_fault_pulse_w) begin
					obs_fault_valid_q <= 1'b1;
					obs_fault_pc_q <= obs_ctrl_fault_pc_w;
					obs_fault_opcode_q <= obs_ctrl_fault_opcode_w;
					obs_fault_opcode_valid_q <= 1'b1;
					obs_fault_source_q <= 3'd6;
					obs_fault_code_q <= obs_ctrl_fault_code_w;
				end
				else if (fetch_fault_w) begin
					obs_fault_valid_q <= 1'b1;
					obs_fault_pc_q <= pc;
					obs_fault_opcode_q <= 5'h00;
					obs_fault_opcode_valid_q <= 1'b0;
					obs_fault_source_q <= 3'd1;
					obs_fault_code_q <= fetch_fault_code_w;
				end
				else if (dma_fault_w) begin
					obs_fault_valid_q <= 1'b1;
					obs_fault_pc_q <= obs_dma_issue_pc_q;
					obs_fault_opcode_q <= obs_dma_issue_opcode_q;
					obs_fault_opcode_valid_q <= 1'b1;
					obs_fault_source_q <= 3'd2;
					obs_fault_code_q <= dma_fault_code_w;
				end
				else if (helper_fault_w) begin
					obs_fault_valid_q <= 1'b1;
					obs_fault_pc_q <= obs_helper_issue_pc_q;
					obs_fault_opcode_q <= obs_helper_issue_opcode_q;
					obs_fault_opcode_valid_q <= 1'b1;
					obs_fault_source_q <= 3'd3;
					obs_fault_code_q <= helper_fault_code_w;
				end
				else if (sfu_fault_w) begin
					obs_fault_valid_q <= 1'b1;
					obs_fault_pc_q <= obs_sfu_issue_pc_q;
					obs_fault_opcode_q <= obs_sfu_issue_opcode_q;
					obs_fault_opcode_valid_q <= 1'b1;
					obs_fault_source_q <= 3'd4;
					obs_fault_code_q <= sfu_fault_code_w;
				end
				else if (sys_sram_fault_now || sys_sram_fault_latched) begin
					obs_fault_valid_q <= 1'b1;
					obs_fault_pc_q <= obs_sys_issue_pc_q;
					obs_fault_opcode_q <= obs_sys_issue_opcode_q;
					obs_fault_opcode_valid_q <= 1'b1;
					obs_fault_source_q <= 3'd5;
					obs_fault_code_q <= 4'h3;
				end
			end
			if (obs_run_active_q && obs_terminal_event_w)
				obs_run_active_q <= 1'b0;
		end
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			insn_data_q <= 64'h0000000000000000;
		else if (insn_valid_w)
			insn_data_q <= insn_data_w;
	fetch_unit u_fetch(
		.clk(clk),
		.rst_n(rst_n),
		.pc(pc),
		.fetch_req(fetch_req),
		.insn_valid(insn_valid_w),
		.insn_data(insn_data_w),
		.fetch_fault(fetch_fault_w),
		.fetch_fault_code(fetch_fault_code_w),
		.m_axi_ar_addr(fetch_ar_addr),
		.m_axi_ar_valid(fetch_ar_valid),
		.m_axi_ar_len(),
		.m_axi_ar_size(),
		.m_axi_ar_burst(),
		.m_axi_ar_ready(fetch_ar_ready),
		.m_axi_r_data(m_axi_r_data),
		.m_axi_r_resp(m_axi_r_resp),
		.m_axi_r_valid(fetch_r_valid),
		.m_axi_r_last(m_axi_r_last),
		.m_axi_r_ready(fetch_r_ready)
	);
	decode_unit u_decode(
		.insn_data(insn_data_q),
		.insn(insn)
	);
	control_unit u_ctrl(
		.clk(clk),
		.rst_n(rst_n),
		.start(start),
		.pc(pc),
		.fetch_req(fetch_req),
		.insn_valid(insn_valid_w),
		.insn(insn),
		.scale_we(scale_we),
		.scale_waddr(scale_waddr),
		.scale_wdata(scale_wdata),
		.addr_lo_we(addr_lo_we),
		.addr_hi_we(addr_hi_we),
		.addr_wsel(addr_wsel),
		.addr_imm28(addr_imm28),
		.tile_we(tile_we),
		.tile_m_in(tile_m_in),
		.tile_n_in(tile_n_in),
		.tile_k_in(tile_k_in),
		.attn_we(attn_we),
		.attn_query_row_base_in(attn_query_row_base_in),
		.attn_valid_kv_len_in(attn_valid_kv_len_in),
		.attn_mode_in(attn_mode_in),
		.tile_valid(tile_valid),
		.tile_n(tile_n),
		.tile_k(tile_k),
		.attn_valid(attn_valid),
		.attn_valid_kv_len(attn_valid_kv_len),
		.attn_mode(attn_mode),
		.dma_dispatch(dma_dispatch),
		.sys_dispatch(sys_dispatch),
		.sfu_dispatch(sfu_dispatch),
		.helper_dispatch(helper_dispatch),
		.dma_busy(dma_busy),
		.sys_busy(sys_busy),
		.sfu_busy(sfu_busy),
		.helper_busy(helper_busy),
		.ext_fault(ext_fault_w),
		.ext_fault_code(ext_fault_code_w),
		.done(done),
		.fault(fault),
		.fault_code(fault_code),
		.obs_retire_pulse(obs_retire_pulse_w),
		.obs_retire_pc(obs_retire_pc_w),
		.obs_retire_opcode(obs_retire_opcode_w),
		.obs_ctrl_fault_pulse(obs_ctrl_fault_pulse_w),
		.obs_ctrl_fault_code(obs_ctrl_fault_code_w),
		.obs_ctrl_fault_pc(obs_ctrl_fault_pc_w),
		.obs_ctrl_fault_opcode(obs_ctrl_fault_opcode_w),
		.obs_sync_wait_dma(obs_sync_wait_dma_w),
		.obs_sync_wait_sys(obs_sync_wait_sys_w),
		.obs_sync_wait_sfu(obs_sync_wait_sfu_w)
	);
	register_file u_regfile(
		.clk(clk),
		.rst_n(rst_n),
		.scale_we((sfu_scale_we_w ? 1'b1 : scale_we)),
		.scale_waddr((sfu_scale_we_w ? sfu_scale_waddr_w : scale_waddr)),
		.scale_wdata((sfu_scale_we_w ? sfu_scale_wdata_w : scale_wdata)),
		.scale_raddr0(insn[226-:4]),
		.scale_rdata0(scale_rdata0),
		.scale_raddr1(insn[226-:4] + 4'd1),
		.scale_rdata1(scale_rdata1),
		.scale_raddr2(insn[226-:4] + 4'd2),
		.scale_rdata2(scale_rdata2),
		.scale_raddr3(insn[226-:4] + 4'd3),
		.scale_rdata3(scale_rdata3),
		.addr_lo_we(addr_lo_we),
		.addr_hi_we(addr_hi_we),
		.addr_wsel(addr_wsel),
		.addr_imm28(addr_imm28),
		.addr_rsel(insn[187-:2]),
		.addr_rdata(addr_rdata),
		.tile_we(tile_we),
		.tile_m_in(tile_m_in),
		.tile_n_in(tile_n_in),
		.tile_k_in(tile_k_in),
		.attn_we(attn_we),
		.attn_query_row_base_in(attn_query_row_base_in),
		.attn_valid_kv_len_in(attn_valid_kv_len_in),
		.attn_mode_in(attn_mode_in),
		.tile_m(tile_m),
		.tile_n(tile_n),
		.tile_k(tile_k),
		.tile_valid(tile_valid),
		.attn_valid(attn_valid),
		.attn_query_row_base(attn_query_row_base),
		.attn_valid_kv_len(attn_valid_kv_len),
		.attn_mode(attn_mode)
	);
	blocking_helper_engine #(.HELPER_SYNTH_MODE(HELPER_SYNTH_MODE)) u_helper(
		.clk(clk),
		.rst_n(rst_n),
		.dispatch(helper_dispatch),
		.opcode(insn[286-:5]),
		.src1_buf(helper_src1_buf_w),
		.src1_off(helper_src1_off_w),
		.src2_buf(helper_src2_buf_w),
		.src2_off(helper_src2_off_w),
		.dst_buf(helper_dst_buf_w),
		.dst_off(helper_dst_off_w),
		.sreg(insn[226-:4]),
		.b_length(insn[133-:16]),
		.b_src_rows(insn[117-:6]),
		.b_transpose(insn[111]),
		.tile_m(tile_m),
		.tile_n(tile_n),
		.scale0_data(scale_rdata0),
		.scale1_data(scale_rdata1),
		.helper_busy(helper_busy),
		.helper_fault(helper_fault_w),
		.helper_fault_code(helper_fault_code_w),
		.sram_a_en(helper_sram_a_en),
		.sram_a_we(helper_sram_a_we),
		.sram_a_buf(helper_sram_a_buf),
		.sram_a_row(helper_sram_a_row),
		.sram_a_wdata(helper_sram_a_wdata),
		.sram_a_rdata(helper_sram_a_rdata),
		.sram_a_fault(helper_sram_a_en & sram_a_fault),
		.sram_b_en(helper_sram_b_en),
		.sram_b_buf(helper_sram_b_buf),
		.sram_b_row(helper_sram_b_row),
		.sram_b_rdata(helper_sram_b_rdata),
		.sram_b_fault(helper_sram_b_en & sram_b_fault)
	);
	sfu_engine #(.SFU_SYNTH_MODE(SFU_SYNTH_MODE)) u_sfu(
		.clk(clk),
		.rst_n(rst_n),
		.dispatch(sfu_dispatch),
		.opcode(insn[286-:5]),
		.src1_buf(insn[280-:2]),
		.src1_off(insn[278-:16]),
		.src2_buf(insn[262-:2]),
		.src2_off(insn[260-:16]),
		.dst_buf(insn[244-:2]),
		.dst_off(insn[242-:16]),
		.sreg(insn[226-:4]),
		.tile_m(tile_m),
		.tile_n(tile_n),
		.tile_k(tile_k),
		.attn_valid(attn_valid),
		.attn_query_row_base(attn_query_row_base),
		.attn_valid_kv_len(attn_valid_kv_len),
		.attn_mode(attn_mode),
		.scale0_data(scale_rdata0),
		.scale1_data(scale_rdata1),
		.scale2_data(scale_rdata2),
		.scale3_data(scale_rdata3),
		.sfu_busy(sfu_busy),
		.sfu_fault(sfu_fault_w),
		.sfu_fault_code(sfu_fault_code_w),
		.sram_a_en(sfu_sram_a_en),
		.sram_a_we(sfu_sram_a_we),
		.sram_a_buf(sfu_sram_a_buf),
		.sram_a_row(sfu_sram_a_row),
		.sram_a_wdata(sfu_sram_a_wdata),
		.sram_a_fault(sfu_sram_a_en & sram_a_fault),
		.sram_b_en(sfu_sram_b_en),
		.sram_b_buf(sfu_sram_b_buf),
		.sram_b_row(sfu_sram_b_row),
		.sram_b_rdata(sfu_sram_b_rdata),
		.sram_b_fault(sfu_sram_b_en & sram_b_fault),
		.sfu_scale_we(sfu_scale_we_w),
		.sfu_scale_waddr(sfu_scale_waddr_w),
		.sfu_scale_wdata(sfu_scale_wdata_w)
	);
	dma_engine #(.DRAM_SIZE(DRAM_SIZE)) u_dma(
		.clk(clk),
		.rst_n(rst_n),
		.dispatch(dma_dispatch),
		.is_store(dma_is_store),
		.buf_id(insn[221-:2]),
		.sram_off(insn[219-:16]),
		.xfer_len(insn[203-:16]),
		.base_addr(addr_rdata),
		.dram_off(insn[185-:16]),
		.dma_busy(dma_busy),
		.dma_rd_busy(dma_rd_busy),
		.dma_fault(dma_fault_w),
		.dma_fault_code(dma_fault_code_w),
		.sram_en(dma_sram_en),
		.sram_we(dma_sram_we),
		.sram_buf(dma_sram_buf),
		.sram_row(dma_sram_row),
		.sram_wdata(dma_sram_wdata),
		.sram_rdata(dma_sram_rdata),
		.sram_fault(dma_sram_fault_w),
		.dma_ar_addr(dma_ar_addr),
		.dma_ar_len(dma_ar_len),
		.dma_ar_valid(dma_ar_valid),
		.dma_ar_ready(dma_ar_ready),
		.dma_r_data(m_axi_r_data),
		.dma_r_resp(m_axi_r_resp),
		.dma_r_valid(dma_r_valid),
		.dma_r_last(m_axi_r_last),
		.dma_r_ready(dma_r_ready),
		.dma_aw_addr(m_axi_aw_addr),
		.dma_aw_len(m_axi_aw_len),
		.dma_aw_valid(m_axi_aw_valid),
		.dma_aw_ready(m_axi_aw_ready),
		.dma_w_data(m_axi_w_data),
		.dma_w_strb(m_axi_w_strb),
		.dma_w_valid(m_axi_w_valid),
		.dma_w_last(m_axi_w_last),
		.dma_w_ready(m_axi_w_ready),
		.dma_b_resp(m_axi_b_resp),
		.dma_b_valid(m_axi_b_valid),
		.dma_b_ready(m_axi_b_ready)
	);
	systolic_controller #(.SYSTOLIC_ARCH_MODE(SYSTOLIC_ARCH_MODE)) u_systolic(
		.clk(clk),
		.rst_n(rst_n),
		.dispatch(sys_dispatch),
		.tile_m(tile_m),
		.tile_n(tile_n),
		.tile_k(tile_k),
		.src1_buf(insn[280-:2]),
		.src1_off(insn[278-:16]),
		.src2_buf(insn[262-:2]),
		.src2_off(insn[260-:16]),
		.dst_buf(insn[244-:2]),
		.dst_off(insn[242-:16]),
		.flags_accumulate(insn[222]),
		.sys_busy(sys_busy),
		.sram_a_en(sys_sram_a_en),
		.sram_a_we(sys_sram_a_we),
		.sram_a_buf(sys_sram_a_buf),
		.sram_a_row(sys_sram_a_row),
		.sram_a_wdata(sys_sram_a_wdata),
		.sram_a_rdata(sys_sram_a_rdata),
		.sram_b_en(sys_sram_b_en),
		.sram_b_buf(sys_sram_b_buf),
		.sram_b_row(sys_sram_b_row),
		.sram_b_rdata(sys_sram_b_rdata)
	);
	sram_subsystem u_sram(
		.clk(clk),
		.rst_n(rst_n),
		.a_en(sram_a_en),
		.a_we(sram_a_we),
		.a_buf(sram_a_buf),
		.a_row(sram_a_row),
		.a_wdata(sram_a_wdata),
		.a_rdata(sram_a_rdata),
		.a_fault(sram_a_fault),
		.b_en(sram_b_en),
		.b_buf(sram_b_buf),
		.b_row(sram_b_row),
		.b_rdata(sram_b_rdata),
		.b_fault(sram_b_fault)
	);
	initial _sv2v_0 = 0;
endmodule
module pll_stub (
	clk_in,
	clk_out
);
	input wire clk_in;
	output wire clk_out;
	assign clk_out = clk_in;
endmodule
module iobuf_stub (
	pin,
	clk,
	out
);
	input wire pin;
	input wire clk;
	output wire out;
	reg [1:0] sync_q;
	always @(posedge clk) sync_q <= {sync_q[0], pin};
	assign out = sync_q[1];
endmodule
module ddr_axi_stub (
	clk,
	rst_n,
	m_axi_ar_addr,
	m_axi_ar_valid,
	m_axi_ar_len,
	m_axi_ar_size,
	m_axi_ar_burst,
	m_axi_ar_ready,
	m_axi_r_data,
	m_axi_r_resp,
	m_axi_r_valid,
	m_axi_r_last,
	m_axi_r_ready,
	m_axi_aw_addr,
	m_axi_aw_len,
	m_axi_aw_size,
	m_axi_aw_burst,
	m_axi_aw_valid,
	m_axi_aw_ready,
	m_axi_w_data,
	m_axi_w_strb,
	m_axi_w_valid,
	m_axi_w_last,
	m_axi_w_ready,
	m_axi_b_resp,
	m_axi_b_valid,
	m_axi_b_ready
);
	parameter signed [31:0] DRAM_SIZE = 16777216;
	input wire clk;
	input wire rst_n;
	localparam signed [31:0] taccel_pkg_AXI_ADDR_W = 56;
	input wire [55:0] m_axi_ar_addr;
	input wire m_axi_ar_valid;
	input wire [7:0] m_axi_ar_len;
	input wire [2:0] m_axi_ar_size;
	input wire [1:0] m_axi_ar_burst;
	output wire m_axi_ar_ready;
	localparam signed [31:0] taccel_pkg_AXI_DATA_W = 128;
	output wire [127:0] m_axi_r_data;
	output wire [1:0] m_axi_r_resp;
	output wire m_axi_r_valid;
	output wire m_axi_r_last;
	input wire m_axi_r_ready;
	input wire [55:0] m_axi_aw_addr;
	input wire [7:0] m_axi_aw_len;
	input wire [2:0] m_axi_aw_size;
	input wire [1:0] m_axi_aw_burst;
	input wire m_axi_aw_valid;
	output wire m_axi_aw_ready;
	input wire [127:0] m_axi_w_data;
	input wire [15:0] m_axi_w_strb;
	input wire m_axi_w_valid;
	input wire m_axi_w_last;
	output wire m_axi_w_ready;
	output wire [1:0] m_axi_b_resp;
	output wire m_axi_b_valid;
	input wire m_axi_b_ready;
	assign m_axi_ar_ready = 1'b1;
	assign m_axi_aw_ready = 1'b1;
	assign m_axi_w_ready = 1'b1;
	assign m_axi_r_data = 1'sb0;
	assign m_axi_r_resp = 2'b00;
	assign m_axi_r_valid = 1'b0;
	assign m_axi_r_last = 1'b0;
	assign m_axi_b_resp = 2'b00;
	assign m_axi_b_valid = 1'b0;
	wire _unused;
	assign _unused = &{1'b0, clk, rst_n, m_axi_ar_addr, m_axi_ar_valid, m_axi_ar_len, m_axi_ar_size, m_axi_ar_burst, m_axi_r_ready, m_axi_aw_addr, m_axi_aw_len, m_axi_aw_size, m_axi_aw_burst, m_axi_aw_valid, m_axi_w_data, m_axi_w_strb, m_axi_w_valid, m_axi_w_last, m_axi_b_ready};
endmodule
module taccel_top_fpga (
	clk_pin,
	rst_n_pin,
	start,
	done,
	fault,
	fault_code
);
	localparam signed [31:0] taccel_pkg_SYS_MODE_CHAINED = 1;
	localparam signed [31:0] taccel_pkg_SYS_MODE_DEFAULT = taccel_pkg_SYS_MODE_CHAINED;
	parameter signed [31:0] SYSTOLIC_ARCH_MODE = taccel_pkg_SYS_MODE_DEFAULT;
	parameter signed [31:0] DRAM_SIZE = 16777216;
	parameter signed [31:0] SFU_SYNTH_MODE = 1;
	parameter signed [31:0] HELPER_SYNTH_MODE = 1;
	input wire clk_pin;
	input wire rst_n_pin;
	input wire start;
	output wire done;
	output wire fault;
	output wire [3:0] fault_code;
	wire clk;
	wire rst_n;
	pll_stub u_pll(
		.clk_in(clk_pin),
		.clk_out(clk)
	);
	iobuf_stub u_rst_sync(
		.pin(rst_n_pin),
		.clk(clk),
		.out(rst_n)
	);
	localparam signed [31:0] taccel_pkg_AXI_ADDR_W = 56;
	wire [55:0] m_axi_ar_addr;
	wire m_axi_ar_valid;
	wire [7:0] m_axi_ar_len;
	wire [2:0] m_axi_ar_size;
	wire [1:0] m_axi_ar_burst;
	wire m_axi_ar_ready;
	localparam signed [31:0] taccel_pkg_AXI_DATA_W = 128;
	wire [127:0] m_axi_r_data;
	wire [1:0] m_axi_r_resp;
	wire m_axi_r_valid;
	wire m_axi_r_last;
	wire m_axi_r_ready;
	wire [55:0] m_axi_aw_addr;
	wire [7:0] m_axi_aw_len;
	wire [2:0] m_axi_aw_size;
	wire [1:0] m_axi_aw_burst;
	wire m_axi_aw_valid;
	wire m_axi_aw_ready;
	wire [127:0] m_axi_w_data;
	wire [15:0] m_axi_w_strb;
	wire m_axi_w_valid;
	wire m_axi_w_last;
	wire m_axi_w_ready;
	wire [1:0] m_axi_b_resp;
	wire m_axi_b_valid;
	wire m_axi_b_ready;
	ddr_axi_stub #(.DRAM_SIZE(DRAM_SIZE)) u_ddr(
		.clk(clk),
		.rst_n(rst_n),
		.m_axi_ar_addr(m_axi_ar_addr),
		.m_axi_ar_valid(m_axi_ar_valid),
		.m_axi_ar_len(m_axi_ar_len),
		.m_axi_ar_size(m_axi_ar_size),
		.m_axi_ar_burst(m_axi_ar_burst),
		.m_axi_ar_ready(m_axi_ar_ready),
		.m_axi_r_data(m_axi_r_data),
		.m_axi_r_resp(m_axi_r_resp),
		.m_axi_r_valid(m_axi_r_valid),
		.m_axi_r_last(m_axi_r_last),
		.m_axi_r_ready(m_axi_r_ready),
		.m_axi_aw_addr(m_axi_aw_addr),
		.m_axi_aw_len(m_axi_aw_len),
		.m_axi_aw_size(m_axi_aw_size),
		.m_axi_aw_burst(m_axi_aw_burst),
		.m_axi_aw_valid(m_axi_aw_valid),
		.m_axi_aw_ready(m_axi_aw_ready),
		.m_axi_w_data(m_axi_w_data),
		.m_axi_w_strb(m_axi_w_strb),
		.m_axi_w_valid(m_axi_w_valid),
		.m_axi_w_last(m_axi_w_last),
		.m_axi_w_ready(m_axi_w_ready),
		.m_axi_b_resp(m_axi_b_resp),
		.m_axi_b_valid(m_axi_b_valid),
		.m_axi_b_ready(m_axi_b_ready)
	);
	taccel_top #(
		.SYSTOLIC_ARCH_MODE(SYSTOLIC_ARCH_MODE),
		.DRAM_SIZE(DRAM_SIZE),
		.SFU_SYNTH_MODE(SFU_SYNTH_MODE),
		.HELPER_SYNTH_MODE(HELPER_SYNTH_MODE)
	) u_core(
		.clk(clk),
		.rst_n(rst_n),
		.start(start),
		.done(done),
		.fault(fault),
		.fault_code(fault_code),
		.m_axi_ar_addr(m_axi_ar_addr),
		.m_axi_ar_valid(m_axi_ar_valid),
		.m_axi_ar_len(m_axi_ar_len),
		.m_axi_ar_size(m_axi_ar_size),
		.m_axi_ar_burst(m_axi_ar_burst),
		.m_axi_ar_ready(m_axi_ar_ready),
		.m_axi_r_data(m_axi_r_data),
		.m_axi_r_resp(m_axi_r_resp),
		.m_axi_r_valid(m_axi_r_valid),
		.m_axi_r_last(m_axi_r_last),
		.m_axi_r_ready(m_axi_r_ready),
		.m_axi_aw_addr(m_axi_aw_addr),
		.m_axi_aw_len(m_axi_aw_len),
		.m_axi_aw_size(m_axi_aw_size),
		.m_axi_aw_burst(m_axi_aw_burst),
		.m_axi_aw_valid(m_axi_aw_valid),
		.m_axi_aw_ready(m_axi_aw_ready),
		.m_axi_w_data(m_axi_w_data),
		.m_axi_w_strb(m_axi_w_strb),
		.m_axi_w_valid(m_axi_w_valid),
		.m_axi_w_last(m_axi_w_last),
		.m_axi_w_ready(m_axi_w_ready),
		.m_axi_b_resp(m_axi_b_resp),
		.m_axi_b_valid(m_axi_b_valid),
		.m_axi_b_ready(m_axi_b_ready)
	);
endmodule