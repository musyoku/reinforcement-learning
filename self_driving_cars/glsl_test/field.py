# -*- coding: utf-8 -*-
import math, time
import numpy as np
from pprint import pprint
from vispy import app, gloo, visuals

# width, height
screen_size = (1180, 800)

# グリッド数
n_grid_h = 6
n_grid_w = 8


# 色
color_black = np.asarray((14.0 / 255.0, 14.0 / 255.0, 14.0 / 255.0, 1.0))
color_field_bg_base = np.asarray((27.0 / 255.0, 59.0 / 255.0, 70.0 / 255.0, 1.0))
color_field_bg = 0.0 * color_black + 1.0 * color_field_bg_base
color_field_grid_base = np.asarray((232.0 / 255.0, 250.0 / 255.0, 174.0 / 255.0, 1.0))
color_field_point = 0.6 * color_field_bg_base + 0.4 * color_field_grid_base
color_field_subdiv_point_base = np.asarray((134.0 / 255.0, 214.0 / 255.0, 247.0 / 255.0, 1.0))
color_field_subdiv_point = 0.3 * color_field_bg_base + 0.7 * color_field_subdiv_point_base
color_field_wall = np.asarray((241.0 / 255.0, 30.0 / 255.0, 30.0 / 255.0, 1.0))
color_gui_text_highlighted = np.asarray((170.0 / 255.0, 248.0 / 255.0, 230.0 / 255.0, 1.0))
color_gui_text = np.asarray((107.0 / 255.0, 189.0 / 255.0, 205.0 / 255.0, 1.0))
color_gui_grid_base = np.asarray((155.0 / 255.0, 234.0 / 255.0, 247.0 / 255.0, 1.0))
color_gui_grid_highlighted = 0.2 * color_black + 0.8 * color_gui_grid_base
color_gui_grid = 0.4 * color_black + 0.5 * color_gui_grid_base
color_gui_sensor_red = np.asarray((247.0 / 255.0, 121.0 / 255.0, 71.0 / 255.0, 1.0))
color_gui_sensor_yellow = np.asarray((212.0 / 255.0, 219.0 / 255.0, 185.0 / 255.0, 1.0))
color_gui_sensor_blue = np.asarray((107.0 / 255.0, 189.0 / 255.0, 205.0 / 255.0, 1.0))
color_gui_sensor_line = color_gui_grid_highlighted
color_gui_sensor_line_highlight = np.asarray((39.0 / 255.0, 68.0 / 255.0, 74.0 / 255.0, 1.0))


# シェーダ
field_point_vertex = """
attribute vec2 position;
uniform float point_size;

void main() {
	gl_Position = vec4(position, 0.0, 1.0);
	gl_PointSize = point_size;
}
"""

field_point_fragment = """
uniform vec4 color;

void main() {
	gl_FragColor = color;
}
"""

field_bg_vertex = """
attribute vec2 position;
attribute vec4 bg_color;
attribute vec4 wall_color;
attribute vec4 grid_color;
attribute float is_wall;
attribute float grid_enabled;
varying vec4 v_bg_color;
varying vec4 v_wall_color;
varying vec4 v_grid_color;
varying vec2 v_position;
varying float v_is_wall;
varying float v_grid_enabled;

void main() {
	v_bg_color = bg_color;
	v_wall_color = wall_color;
	v_grid_color = grid_color;
	v_position = position;
	v_grid_enabled = grid_enabled;
	v_is_wall = is_wall;
	gl_Position = vec4(position, 0.0, 1.0);
}
"""

field_bg_fragment = """
uniform vec2 screen_size;
varying vec4 v_bg_color;
varying vec4 v_wall_color;
varying vec4 v_grid_color;
varying vec2 v_position;
varying float v_is_wall;
varying float v_grid_enabled;

void main() {
	vec4 bg_color = v_bg_color;
	
	// Wall
	const float M_PI = 3.14159265358979323846;
	const float NUM_LINES = 100.0;
	float theta = M_PI / 6.0;
	float cos_theta = cos(theta);
	float sin_theta = sin(theta);
	vec2 coord = gl_FragCoord.xy / screen_size;
	float x = cos_theta * coord.x - sin_theta * coord.y;
	float f = fract(x * NUM_LINES);
	vec4 wall_highlighted_color = 0.5 * v_bg_color + 0.5 * v_wall_color;
	vec4 wall_color = 0.8 * v_bg_color + 0.2 * v_wall_color;
	vec4 result = mix(wall_highlighted_color, wall_color, float(f < 0.4));
	gl_FragColor = mix(result, bg_color, float(v_is_wall == 0));
}
"""

gui_grid_vertex = """
attribute vec2 position;
attribute float highlighted;
varying float v_highlighted;

void main() {
	v_highlighted = highlighted;
	gl_Position = vec4(position, 0.0, 1.0);
}
"""

gui_grid_fragment = """
varying float v_highlighted;
uniform vec4 color;
uniform vec4 highlighted_color;

void main() {
	gl_FragColor = mix(highlighted_color, color, float(v_highlighted == 0));
}
"""

gui_sensor_vertex = """
attribute vec2 position;

void main() {
	gl_Position = vec4(position, 0.0, 1.0);
}
"""

gui_sensor_fragment = """
uniform vec2 screen_size;
uniform vec2 u_center;
uniform vec2 u_size;
uniform float near[8];
uniform float mid[16];
uniform float far[24];
uniform vec4 near_color;
uniform vec4 mid_color;
uniform vec4 far_color;
uniform vec4 bg_color;
uniform vec4 line_color;
uniform vec4 line_highlighted_color;

const float M_PI = 3.14159265358979323846;

float atan2(in float y, in float x)
{
	float result = atan(y, x) + M_PI;
	return result / M_PI / 2.0;
	//bool s = (abs(x) > abs(y));
	//float result = mix(M_PI / 2.0 - atan(x, y), atan(y, x) + M_PI / 2.0, float(s));
	//return result;
}

void main() {
	vec2 coord = gl_FragCoord.xy;
	const float OUTER_RADIUS_RATIO = 0.8;
	const float COVRE_RADIUS_RATIO = 0.7;
	float outer_radius = u_size.x / 2.0 * OUTER_RADIUS_RATIO;
	const float OUTER_LINE_WIDTH = 1.5;

	// Cover
	float d = distance(coord, u_center);
	float diff = d - outer_radius;
	vec2 local = coord - u_center;
	if(diff <= OUTER_LINE_WIDTH && diff >= -OUTER_LINE_WIDTH){
		diff /= OUTER_LINE_WIDTH;
		float rad = atan2(local.y, local.x);
		//float alpha = (1.0 - abs(0.5 - fract((rad + M_PI / 8.0) * 4.0)));
		//vec4 color = alpha * line_color + (1.0 - alpha) * bg_color;
		gl_FragColor = mix(vec4(line_color.rgb, fract(1 + diff)), vec4(line_color.rgb, 1.0 - fract(diff)), float(diff > 0));
		return;
	}
}
"""

class Field:
	def __init__(self):
		self.enable_grid = True

		# グリッド数
		self.n_grid_h = 6
		self.n_grid_w = 8

		# パディング
		self.px = 80
		self.py = 120

		self.grid_subdiv_bg, self.grid_subdiv_wall = self.load()

		self.program_grid_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.program_grid_point["color"] = color_field_point
		self.program_grid_point["point_size"] = 3.0
		self.program_subdiv_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.program_subdiv_point["color"] = color_field_subdiv_point
		self.program_subdiv_point["point_size"] = 1.0
		self.program_bg = gloo.Program(field_bg_vertex, field_bg_fragment)
		self.program_bg["screen_size"] = canvas.size

	def get_wall_array(self):
		return self.grid_subdiv_wall

	def surrounding_wal_indicis(self, array_x, array_y, radius=1):
		start_xi = 0 if array_x - radius < 0 else array_x - radius
		start_yi = 0 if array_y - radius < 0 else array_y - radius
		end_xi = self.grid_subdiv_wall.shape[1] if array_x + radius + 1 > self.grid_subdiv_wall.shape[1] else array_x + radius + 1
		end_yi = self.grid_subdiv_wall.shape[0] if array_y + radius + 1 > self.grid_subdiv_wall.shape[0] else array_y + radius + 1
		return np.argwhere(self.grid_subdiv_wall[start_yi:end_yi, start_xi:end_xi] == 1)


	def load(self):
		# 背景
		grid_subdiv_bg = np.ones((self.n_grid_h * 4 + 4, self.n_grid_w * 4 + 4), dtype=np.uint8)
		skip = int(np.random.uniform(low=0.3) * 10)
		for x in xrange(grid_subdiv_bg.shape[1]):
			if skip == 0:
				skip = int(np.random.uniform(low=0.3) * 10)
				grid_subdiv_bg[0, x] = 0
				continue
			skip -= 1
		skip = int(np.random.uniform(low=0.3) * 10)
		for x in xrange(grid_subdiv_bg.shape[1]):
			if skip == 0:
				skip = int(np.random.uniform(low=0.3) * 10)
				grid_subdiv_bg[-1, x] = 0
				continue
			skip -= 1
		skip = int(np.random.uniform(low=0.3) * 10)
		for y in xrange(grid_subdiv_bg.shape[0]):
			if skip == 0:
				skip = int(np.random.uniform(low=0.3) * 10)
				grid_subdiv_bg[y, 0] = 0
				continue
			skip -= 1
		skip = int(np.random.uniform(low=0.3) * 10)
		for y in xrange(grid_subdiv_bg.shape[0]):
			if skip == 0:
				skip = int(np.random.uniform(low=0.3) * 10)
				grid_subdiv_bg[y, -1] = 0
				continue
			skip -= 1

		# 壁
		grid_subdiv_wall = np.zeros((self.n_grid_h * 4 + 4, self.n_grid_w * 4 + 4), dtype=np.uint8)
		for x in xrange(grid_subdiv_wall.shape[1]):
			grid_subdiv_wall[0, x] = 1 if grid_subdiv_bg[0, x] == 1 else 0
			grid_subdiv_wall[-1, x] = 1 if grid_subdiv_bg[-1, x] == 1 else 0
			grid_subdiv_wall[1, x] = 1 if grid_subdiv_bg[1, x] == 1 else 0
			grid_subdiv_wall[-2, x] = 1 if grid_subdiv_bg[-2, x] == 1 else 0
		for y in xrange(grid_subdiv_wall.shape[0]):
			grid_subdiv_wall[y, 0] = 1 if grid_subdiv_bg[y, 0] == 1 else 0
			grid_subdiv_wall[y, -1] = 1 if grid_subdiv_bg[y, -1] == 1 else 0
			grid_subdiv_wall[y, 1] = 1 if grid_subdiv_bg[y, 1] == 1 else 0
			grid_subdiv_wall[y, -2] = 1 if grid_subdiv_bg[y, -2] == 1 else 0

		return grid_subdiv_bg, grid_subdiv_wall

	def is_screen_position_inside_field(self, pixel_x, pixel_y, grid_width=None, grid_height=None):
		if grid_width is None or grid_height is None:
			grid_width, grid_height = self.comput_grid_size()
		subdivision_width = grid_width / float(self.n_grid_w) / 4.0
		subdivision_height = grid_height / float(self.n_grid_h) / 4.0
		if pixel_x < self.px - subdivision_width * 2:
			return False
		if pixel_x > self.px + grid_width + subdivision_width * 2:
			return False
		if pixel_y < self.py - subdivision_height * 2:
			return False
		if pixel_y > self.py + grid_height + subdivision_height * 2:
			return False
		return True

	def compute_array_index_from_position(self, pixel_x, pixel_y, grid_width=None, grid_height=None):
		grid_width, grid_height = self.comput_grid_size()
		if self.is_screen_position_inside_field(pixel_x, pixel_y, grid_width=grid_width, grid_height=grid_height) is False:
			return -1, -1
		subdivision_width = grid_width / float(self.n_grid_w) / 4.0
		subdivision_height = grid_height / float(self.n_grid_h) / 4.0
		x = pixel_x - self.px + subdivision_width * 2
		y = pixel_y - self.py + subdivision_height * 2
		return int(x / subdivision_width), self.grid_subdiv_wall.shape[0] - int(y / subdivision_height) - 1

	def is_wall(self, array_x, array_y):
		if array_x < 0:
			return False
		if array_y < 0:
			return False
		if array_x >= self.grid_subdiv_wall.shape[1]:
			return False
		if array_y >= self.grid_subdiv_wall.shape[0]:
			return False
		return True if self.grid_subdiv_wall[array_y, array_x] == 1 else False


	def subdivision_exists(self, array_x, array_y):
		if array_x < 0:
			return False
		if array_y < 0:
			return False
		if array_x >= self.grid_subdiv_bg.shape[1]:
			return False
		if array_y >= self.grid_subdiv_bg.shape[0]:
			return False
		return True if self.grid_subdiv_bg[array_y, array_x] == 1 else False

	def comput_grid_size(self):
		sw, sh = canvas.size
		ratio = self.n_grid_h / float(self.n_grid_w)
		if sw >= sh:
			lh = sh - self.py * 2
			lw = lh / ratio
			# フィールドは画面の左半分
			if lw > sw / 1.3 - self.px * 2:
				lw = sw / 1.3 - self.px * 2
				lh = lw * ratio
			if lh > sh - self.py * 2:
				lh = sh - self.py * 2
				lw = lh / ratio
		else:
			lw = sw / 1.3 - self.px * 2
			lh = lw * ratio
		return lw, lh

	def construct_wall_at_index(self, array_x, array_y):
		if array_x < 0:
			return
		if array_y < 0:
			return
		if array_x >= self.grid_subdiv_wall.shape[1]:
			return
		if array_y >= self.grid_subdiv_wall.shape[0]:
			return
		self.grid_subdiv_wall[array_y, array_x] = 1

	def destroy_wall_at_index(self, array_x, array_y):
		if array_x < 2:
			return
		if array_y < 2:
			return
		if array_x >= self.grid_subdiv_wall.shape[1] - 2:
			return
		if array_y >= self.grid_subdiv_wall.shape[0] - 2:
			return
		self.grid_subdiv_wall[array_y, array_x] = 0

	def set_positions(self):
		np.random.seed(0)
		# スクリーンサイズ
		sw, sh = canvas.size
		# 枠線サイズ
		lw ,lh = self.comput_grid_size()

		# 小さいグリッド
		sgw = lw / float(self.n_grid_w) / 4.0 / float(sw) * 2.0
		sgh = lh / float(self.n_grid_h) / 4.0 / float(sh) * 2.0

		point_positions = []
		for nw in xrange(self.n_grid_w + 1):
			x = lw / float(self.n_grid_w) * nw + self.px
			x /= float(sw)
			for nh in xrange(self.n_grid_h + 1):
				y = lh / float(self.n_grid_h) * nh + self.py
				y /= float(sh)
				point_positions.append((2.0 * x - 1.0, 2.0 * y - 1.0))

		self.program_grid_point["position"] = point_positions

		subdiv_point_positions = []
		# 大きいグリッドの交差点
		for nw in xrange(self.n_grid_w + 1):
			x = lw / float(self.n_grid_w) * nw + self.px
			x = 2.0 * x / float(sw) - 1.0
			for nh in xrange(self.n_grid_h + 1):
				y = lh / float(self.n_grid_h) * nh + self.py
				y = 2.0 * y / float(sh) - 1.0
				# 小さいグリッド
				for sub_y in xrange(5):
					_y = y + sgh * sub_y
					for sub_x in xrange(5):
						xi = nw * 4 + sub_x
						yi = nh * 4 + sub_y

						if self.subdivision_exists(xi, yi) or self.subdivision_exists(xi - 1, yi) or self.subdivision_exists(xi, yi - 1) or self.subdivision_exists(xi - 1, yi - 1):
							_x = x + sgw * sub_x
							# x, yそれぞれ2マス分ずらす
							subdiv_point_positions.append((_x - sgw * 2.0, _y - sgh * 2.0))

		self.program_subdiv_point["position"] = subdiv_point_positions

		bg_positions = []
		bg_colors = []
		wall_colors = []
		is_wall = []
		# x, yそれぞれ2マス分ずらす
		x_start = 2.0 * self.px / float(sw) - 1.0 - sgw * 2.0
		y_start = 2.0 * self.py / float(sh) - 1.0 - sgh * 2.0
		for h in xrange(self.grid_subdiv_bg.shape[0]):
			for w in xrange(self.grid_subdiv_bg.shape[1]):
				if self.grid_subdiv_bg[h, w] == 1:
					bg_positions.append((x_start + sgw * w, y_start + sgh * h))
					bg_positions.append((x_start + sgw * (w + 1), y_start + sgh * h))
					bg_positions.append((x_start + sgw * w, y_start + sgh * (h + 1)))

					bg_positions.append((x_start + sgw * (w + 1), y_start + sgh * h))
					bg_positions.append((x_start + sgw * w, y_start + sgh * (h + 1)))
					bg_positions.append((x_start + sgw * (w + 1), y_start + sgh * (h + 1)))

					distance = (abs(w - self.grid_subdiv_bg.shape[1] / 2.0) / float(self.grid_subdiv_bg.shape[1]), abs(h - self.grid_subdiv_bg.shape[0] / 2.0) / float(self.grid_subdiv_bg.shape[0]))
					weight = 1.0 - math.sqrt(distance[0] ** 2 + distance[1] ** 2)
					weight = weight / 2.0 + 0.5
					opacity = np.random.uniform(0.6 * math.sqrt(weight), 0.7) * weight
					bg_color = opacity * color_field_bg_base + (1.0 - opacity) * color_black
					for i in xrange(6):
						bg_colors.append(bg_color)
						wall_colors.append(color_field_wall)
						iw = 1.0 if self.grid_subdiv_wall[h, w] == 1 else 0.0
						is_wall.append(iw)

		np.random.seed(int(time.time()))
		self.program_bg["position"] = bg_positions
		self.program_bg["bg_color"] = bg_colors
		self.program_bg["wall_color"] = wall_colors
		self.program_bg["is_wall"] = np.asarray(is_wall, dtype=np.float32)

	def draw(self):
		self.set_positions()
		self.program_bg.draw("triangles")
		if self.enable_grid:
			self.program_subdiv_point.draw("points")
			self.program_grid_point.draw("points")

	def draw_wall(self):

		self.program_wall.draw("triangles")

class Gui():
	def __init__(self):
		self.program_grid = gloo.Program(gui_grid_vertex, gui_grid_fragment)
		self.program_grid["color"] = color_gui_grid
		self.program_grid["highlighted_color"] = color_gui_grid_highlighted

		self.program_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.program_point["point_size"] = 3.0
		self.program_point["color"] = color_gui_grid_highlighted

		self.program_bg_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.program_bg_point["point_size"] = 1.0
		self.program_bg_point["color"] = color_gui_grid

		self.program_sensor = gloo.Program(gui_sensor_vertex, gui_sensor_fragment)
		self.program_sensor["near_color"] = color_gui_sensor_red
		self.program_sensor["mid_color"] = color_gui_sensor_yellow
		self.program_sensor["far_color"] = color_gui_sensor_blue
		self.program_sensor["bg_color"] = color_black
		self.program_sensor["line_color"] = color_gui_sensor_line
		self.program_sensor["line_highlighted_color"] = color_gui_sensor_line_highlight
		self.program_sensor["screen_size"] = canvas.size

		self.color_hex_str_text = "#6bbdcd"
		self.color_hex_str_text_highlighted = "#a0c6c3"

		self.text_title_field = visuals.TextVisual("SELF-DRIVING", color=self.color_hex_str_text_highlighted, bold=True, anchor_x="left", anchor_y="top")
		self.text_title_field.font_size = 16

		self.text_title_data = visuals.TextVisual("DATA", color=self.color_hex_str_text, bold=True, anchor_x="left", anchor_y="top")
		self.text_title_data.font_size = 16

		self.text_title_sensor = visuals.TextVisual("SENSOR", color=self.color_hex_str_text, bold=True, anchor_x="left", anchor_y="top")
		self.text_title_sensor.font_size = 16

	def set_positions(self):
		sw, sh = canvas.size
		sw = float(sw)
		sh = float(sh)
		lw ,lh = field.comput_grid_size()
		sgw = lw / float(field.n_grid_w) / 4.0
		sgh = lh / float(field.n_grid_h) / 4.0
		positions = []
		highlighted = []
		points = []

		# Background
		bg_points = []
		base_x = 2.0 * (field.px - sgw * 5) / sw - 1
		base_y = 2.0 * (field.py - sgh * 5) / sh - 1
		step_x = sgw / sw * 4.0
		step_y = sgh / sh * 4.0
		for yn in xrange(20):
			for xn in xrange(30):
				bg_points.append((base_x + step_x * xn, base_y + step_y * yn))
		self.program_bg_point["position"] = bg_points

		def register(grid_segments, point_segments, highlights, base_x, base_y, length):
			for i, (f, t) in enumerate(zip(grid_segments[:-1], grid_segments[1:])):
				positions.append((base_x + length * f, base_y))
				positions.append((base_x + length * t, base_y))
				highlighted.append(highlights[i])
				highlighted.append(highlights[i])
			for r in point_segments:
				points.append((base_x + length * r, base_y))

		# Field
		## Top
		register([0.0, 0.1, 0.3, 0.8, 1.0], [0.0, 0.1, 0.3, 0.8, 1.0], [0, 1, 0, 1, 0], 2.0 * (field.px - sgw * 2.0) / sw - 1, 2.0 * (lh + field.py + sgh * 4.0) / sh - 1, (lw + sgw * 4.0) / sw * 2.0)
		register([0.3, 0.8, 1.0], [0.0, 0.3, 0.8, 1.0], [1, 0, 1], 2.0 * (field.px - sgw * 2.0) / sw - 1, 2.0 * (lh + field.py + sgh * 3.75) / sh - 1, (lw + sgw * 4.0) / sw * 2.0)
		## Bottom
		register([0.0, 0.3, 0.8, 1.0], [0.0, 0.3, 0.8, 1.0], [1, 0, 0], 2.0 * (field.px - sgw * 2.0) / sw - 1, 2.0 * (field.py - sgh * 4.0) / sh - 1, (lw + sgw * 4.0) / sw * 2.0)
		register([0.0, 0.3, 0.8, 1.0], [0.0, 0.3, 0.8, 1.0], [0, 1, 1], 2.0 * (field.px - sgw * 2.0) / sw - 1, 2.0 * (field.py - sgh * 3.75) / sh - 1, (lw + sgw * 4.0) / sw * 2.0)

		# Right Column
		## Top
		register([0.0, 0.6, 1.0], [0.0, 0.6, 1.0], [1, 0], 2.0 * (field.px + lw + sgw * 3.0) / sw - 1, 2.0 * (field.py + lh + sgh * 4.0) / sh - 1, sgw * 10 / sw * 2.0)
		register([0.6, 1.0], [0.0, 0.6, 1.0], [1], 2.0 * (field.px + lw + sgw * 3.0) / sw - 1, 2.0 * (field.py + lh + sgh * 3.75) / sh - 1, sgw * 10 / sw * 2.0)
		## Bottom
		register([0.0, 1.0], [0.0, 1.0], [1], 2.0 * (field.px + lw + sgw * 3.0) / sw - 1, 2.0 * (field.py - sgh * 4.0) / sh - 1, sgw * 10 / sw * 2.0)
		register([0.0, 1.0], [0.0, 1.0], [0], 2.0 * (field.px + lw + sgw * 3.0) / sw - 1, 2.0 * (field.py - sgh * 3.75) / sh - 1, sgw * 10 / sw * 2.0)
		## Status
		register([0.0, 0.6, 1.0], [0.0, 0.6, 1.0], [1, 0], 2.0 * (field.px + lw + sgw * 3.0) / sw - 1, 2.0 * (field.py + sgh * 9.0) / sh - 1, sgw * 10 / sw * 2.0)
		register([0.6, 1.0], [0.0, 0.6, 1.0], [1], 2.0 * (field.px + lw + sgw * 3.0) / sw - 1, 2.0 * (field.py + sgh * 8.75) / sh - 1, sgw * 10 / sw * 2.0)

		self.program_grid["position"] = positions
		self.program_grid["highlighted"] = np.asarray(highlighted, dtype=np.float32)
		self.program_point["position"] = points

		# Text
		self.text_title_field.pos = field.px - sgw * 1.5, sh - lh - field.py - sgh * 3.5
		self.text_title_data.pos = field.px + lw + sgw * 3.5, sh - lh - field.py - sgh * 3.5
		self.text_title_sensor.pos = field.px + lw + sgw * 3.5, sh - field.py - sgh * 8.5

	def configure(self, canvas, viewport):
		self.text_title_field.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_title_data.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_title_sensor.transforms.configure(canvas=canvas, viewport=viewport)
		
	def draw(self):
		self.set_positions()
		self.program_bg_point.draw("points")
		self.program_grid.draw("lines")
		self.program_point.draw("points")
		self.text_title_field.draw()
		self.text_title_data.draw()
		self.text_title_sensor.draw()
		self.draw_sensor()

	def draw_sensor(self):
		car = cars.get_car_at_index(0)
		sensor_value = car.get_sensor_value()
		# HACK
		for i in xrange(8):
			self.program_sensor["near[%d]" % i] = sensor_value[i]
		for i in xrange(16):
			self.program_sensor["mid[%d]" % i] = sensor_value[i + 8]
		for i in xrange(24):
			self.program_sensor["far[%d]" % i] = sensor_value[i + 24]

		sw, sh = canvas.size
		sw = float(sw)
		sh = float(sh)
		lw ,lh = field.comput_grid_size()
		sgw = lw / float(field.n_grid_w) / 4.0
		sgh = lh / float(field.n_grid_h) / 4.0
		base_x = 2.0 * (field.px + lw + sgw * 3.0) / sw - 1
		base_y = 2.0 * (field.py - sgh * 2.0) / sh - 1
		width = sgw * 10 / sw * 2.0
		height = sgh * 10 / sh * 2.0
		center = (field.px + lw + sgw * 8.0, field.py + sgh * 3.0)
		self.program_sensor["u_center"] = center
		self.program_sensor["u_size"] = sgw * 10, sgh * 10
		positions = []
		positions.append((base_x, base_y))
		positions.append((base_x + width, base_y))
		positions.append((base_x, base_y + height))
		positions.append((base_x + width, base_y + height))
		positions.append(positions[1])
		positions.append(positions[2])
		positions.append(positions[3])
		self.program_sensor["position"] = positions

		self.program_sensor.draw("triangles")

class CarManager:
	def __init__(self, initial_num_car=10):
		self.cars = []
		for i in xrange(initial_num_car):
			self.cars.append(Car())

	def get_car_at_index(self, index=0):
		if index < len(self.cars):
			return self.cars[index]
		return None

class Car:
	near_lookup = np.array([[5, 4, 3], [6, -1, 2], [7, 0, 1]])
	mid_lookup = np.array([[10, 9, 8, 7, 6], [11, -1, -1, -1, 5], [12, -1, -1, -1, 4], [13, -1, -1, -1, 3], [14, 15, 0, 1, 2]])
	far_lookup = np.array([[15, 14, 13, 12, 11, 10, 9], [16, -1, -1, -1, -1, -1, 8], [17, -1, -1, -1, -1, -1, 7], [18, -1, -1, -1, -1, -1, 6], [19, -1, -1, -1, -1, -1, 5], [20, -1, -1, -1, -1, -1, 4], [21, 22, 23, 0, 1, 2, 3]])
	def __init__(self):
		self.speed = 0
		self.steering = 0
		self.steering_unit = math.pi / 30.0
		self.pos = (0, 0)

	def respawn(self):
		pass

	def get_sensor_value(self):
		xi, yi = field.compute_array_index_from_position(self.pos[0], self.pos[1])
		values = np.zeros((48,), dtype=np.float32)

		# 近距離
		blocks = field.surrounding_wal_indicis(xi, yi, 1)
		for block in blocks:
			i = Car.near_lookup[block[0]][block[1]]
			if i != -1:
				values[i] = 1.0

		# 中距離
		blocks = field.surrounding_wal_indicis(xi, yi, 2)
		for block in blocks:
			i = Car.mid_lookup[block[0]][block[1]]
			if i != -1:
				values[i + 8] = 1.0

		# 遠距離
		blocks = field.surrounding_wal_indicis(xi, yi, 3)
		for block in blocks:
			i = Car.far_lookup[block[0]][block[1]]
			if i != -1:
				values[i + 24] = 1.0

		# 車体の向きに合わせる
		area = int(self.steering / math.pi * 4.0)
		ratio = self.steering % (math.pi / 4.0)
		mix = np.roll(values[0:8], -(area + 1)) * ratio + np.roll(values[0:8], -area) * (1.0 - ratio)
		values[0:8] = mix

		area = int(self.steering / math.pi * 8.0)
		ratio = self.steering % (math.pi / 8.0)
		mix = np.roll(values[8:24], -(area + 1)) * ratio + np.roll(values[8:24], -area) * (1.0 - ratio)
		values[8:24] = mix

		area = int(self.steering / math.pi * 12.0)
		ratio = self.steering % (math.pi / 12.0)
		mix = np.roll(values[24:48], -(area + 1)) * ratio + np.roll(values[24:48], -area) * (1.0 - ratio)
		values[24:48] = mix

		return values

	# アクセル
	def action_throttle(self):
		self.speed += 1

	# ブレーキ
	def action_brake(self):
		self.speed -= 1

	# ハンドル
	def action_steer_right(self):
		self.steering = (self.steering + self.steering_unit) % (math.pi * 2.0)

	def action_steer_left(self):
		self.steering = (self.steering - self.steering_unit) % (math.pi * 2.0)

class Canvas(app.Canvas):
	def __init__(self):
		app.Canvas.__init__(self, size=screen_size, title="self-driving", keys="interactive")

		self.is_mouse_pressed = False
		self.is_key_shift_pressed = False

	def step(self):
		pass

	def on_draw(self, event):
		gloo.clear(color="#0e0e0e")
		gloo.set_viewport(0, 0, *self.physical_size)
		field.draw()
		gui.draw()

	def on_resize(self, event):
		self.activate_zoom()
		print "#on_resize()", (self.width, self.height)

	def on_mouse_press(self, event):
		self.is_mouse_pressed = True
		self.toggle_wall(event.pos)

	def on_mouse_release(self, event):
		self.is_mouse_pressed = False

	def on_mouse_move(self, event):
		self.toggle_wall(event.pos)
		car.pos = event.pos[0], event.pos[1]
		gui.draw_sensor()

	def on_mouse_wheel(self, event):
		car.action_steer_right()
		gui.draw_sensor()

	def toggle_wall(self, pos):
		if self.is_mouse_pressed:
			if field.is_screen_position_inside_field(pos[0], pos[1]):
				x, y = field.compute_array_index_from_position(pos[0], pos[1])
				if self.is_key_shift_pressed:
					field.destroy_wall_at_index(x, y)
				else:
					field.construct_wall_at_index(x, y)
				self.update()


	def on_key_press(self, event):
		if event.key == "Shift":
			self.is_key_shift_pressed = True

	def on_key_release(self, event):
		if event.key == "Shift":
			self.is_key_shift_pressed = False

	def activate_zoom(self):
		self.width, self.height = self.size
		gloo.set_viewport(0, 0, *self.physical_size)
		vp = (0, 0, self.physical_size[0], self.physical_size[1])
		gui.configure(canvas=self, viewport=vp)

if __name__ == "__main__":
	canvas = Canvas()
	gui = Gui()
	field = Field()
	cars = CarManager()
	car = Car()
	canvas.activate_zoom()
	canvas.show()
	app.run()
