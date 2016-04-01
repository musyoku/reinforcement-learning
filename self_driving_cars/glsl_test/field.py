# -*- coding: utf-8 -*-
import math, time
import numpy as np
from pprint import pprint
from vispy import app, gloo, visuals

# width, height
screen_size = (1180, 800)

initial_num_car = 20

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
color_gui_sensor_red = np.asarray((247.0 / 255.0, 101.0 / 255.0, 51.0 / 255.0, 1.0))
color_gui_sensor_yellow = np.asarray((212.0 / 255.0, 219.0 / 255.0, 185.0 / 255.0, 1.0))
color_gui_sensor_blue = np.asarray((107.0 / 255.0, 189.0 / 255.0, 205.0 / 255.0, 1.0))
color_gui_sensor_line = 0.4 * color_black + 0.6 * color_gui_grid_base
color_gui_sensor_line_highlight = np.asarray((39.0 / 255.0, 68.0 / 255.0, 74.0 / 255.0, 1.0))
color_car_normal = color_gui_sensor_yellow
color_car_crashed = np.asarray((147.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0, 1.0))
color_car_reward = color_gui_sensor_blue


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
	vec4 wall_highlighted_color = vec4(v_wall_color.rgb, 0.3);
	vec4 wall_color = vec4(v_wall_color.rgb, 0.7);
	vec4 result = mix(wall_color, wall_highlighted_color, float(f < 0.4));
	gl_FragColor = mix(result, bg_color, float(v_is_wall == 0));
}
"""


car_vertex = """
attribute vec2 position;
attribute vec4 color;
varying vec4 v_color;

void main() {
	v_color = color;
	gl_Position = vec4(position, 0.0, 1.0);
}
"""

car_fragment = """
varying vec4 v_color;

void main() {
	gl_FragColor = v_color;
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
	return 1.0 - mod(result + M_PI / 2.0, M_PI * 2.0) / M_PI / 2.0;
	//bool s = (abs(x) > abs(y));
	//float result = mix(M_PI / 2.0 - atan(x, y), atan(y, x) + M_PI / 2.0, float(s));
	//return result;
}

void main() {
	vec2 coord = gl_FragCoord.xy;
	float d = distance(coord, u_center);
	vec2 local = coord - u_center;
	float rad = atan2(local.y, local.x);

	// #1
	float radius = u_size.x / 2.0 * 0.8;
	float diff = d - radius;
	float line_width = 1.5;
	if(abs(diff) <= line_width){
		diff /= line_width;
		vec4 frag_color = mix(vec4(line_color.rgb, fract(1 + diff)), vec4(line_color.rgb, 1.0 - fract(diff)), float(diff > 0));
		frag_color.a = mix(0.0, frag_color.a, float(rad > 0.5));
		gl_FragColor = frag_color;
		return;
	}

	// #2
	radius = u_size.x / 2.0 * 0.7;
	diff = d - radius;
	if(abs(diff) <= line_width){
		diff /= line_width;
		gl_FragColor = mix(vec4(line_color.rgb, fract(1 + diff)), vec4(line_color.rgb, 1.0 - fract(diff)), float(diff > 0));
		return;
	}

	// far
	radius = u_size.x / 2.0 * 0.6;
	diff = d - radius;
	line_width = 6;
	float segments = 24.0;
	if(abs(diff) <= line_width / 2.0){
		vec4 result;
		if(diff >= 0){
			diff -= (line_width / 2.0 - 1.0);
			result = mix(vec4(far_color.rgb, 1.0 - fract(diff)), far_color, float(diff < 0));
		}else{
			diff += line_width / 2.0;
			result = mix(vec4(far_color.rgb, fract(1 + diff)), far_color, float(diff >= 1));
		}
		int index = int(fract(rad + 1.0 / segments / 2.0) * segments);
		float rat = far[index];
		gl_FragColor = mix(vec4(line_highlighted_color.rgb, result.a), result, rat);
		return;
	}

	// mid
	radius = u_size.x / 2.0 * 0.5;
	diff = d - radius;
	line_width = 6;
	segments = 16.0;
	if(abs(diff) <= line_width / 2.0){
		vec4 result;
		if(diff >= 0){
			diff -= (line_width / 2.0 - 1.0);
			result = mix(vec4(mid_color.rgb, 1.0 - fract(diff)), mid_color, float(diff < 0));
		}else{
			diff += line_width / 2.0;
			result = mix(vec4(mid_color.rgb, fract(1 + diff)), mid_color, float(diff >= 1));
		}
		int index = int(fract(rad + 1.0 / segments / 2.0) * segments);
		float rat = mid[index];
		gl_FragColor = mix(vec4(line_highlighted_color.rgb, result.a), result, rat);
		return;
	}

	// near
	radius = u_size.x / 2.0 * 0.4;
	diff = d - radius;
	line_width = 6;
	segments = 8.0;
	if(abs(diff) <= line_width / 2.0){
		vec4 result;
		if(diff >= 0){
			diff -= (line_width / 2.0 - 1.0);
			result = mix(vec4(near_color.rgb, 1.0 - fract(diff)), near_color, float(diff < 0));
		}else{
			diff += line_width / 2.0;
			result = mix(vec4(near_color.rgb, fract(1 + diff)), near_color, float(diff >= 1));
		}
		int index = int(fract(rad + 1.0 / segments / 2.0) * segments);
		float rat = near[index];
		gl_FragColor = mix(vec4(line_highlighted_color.rgb, result.a), result, rat);
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

		self.needs_display = True
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
		zeros = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
		extract = self.grid_subdiv_wall[start_yi:end_yi, start_xi:end_xi]
		y_shift = max(radius - array_y, 0)
		x_shift = max(radius - array_x, 0)
		zeros[y_shift:y_shift + extract.shape[0], x_shift:x_shift + extract.shape[1]] = extract
		return  np.argwhere(zeros == 1)

	def set_needs_display(self):
		self.needs_display = True

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
		_, screen_height = canvas.size
		subdivision_width = grid_width / float(self.n_grid_w) / 4.0
		subdivision_height = grid_height / float(self.n_grid_h) / 4.0
		if pixel_x < self.px - subdivision_width * 2:
			return False
		if pixel_x > self.px + grid_width + subdivision_width * 2:
			return False
		if pixel_y < screen_height - self.py - grid_height - subdivision_height * 2:
			return False
		if pixel_y > screen_height - self.py + subdivision_height * 2:
			return False
		return True

	def compute_array_index_from_position(self, pixel_x, pixel_y, grid_width=None, grid_height=None):
		grid_width, grid_height = self.comput_grid_size()
		if self.is_screen_position_inside_field(pixel_x, pixel_y, grid_width=grid_width, grid_height=grid_height) is False:
			return -1, -1
		_, screen_height = canvas.size
		subdivision_width = grid_width / float(self.n_grid_w) / 4.0
		subdivision_height = grid_height / float(self.n_grid_h) / 4.0
		x = pixel_x - self.px + subdivision_width * 2
		y = pixel_y - (screen_height - self.py - grid_height - subdivision_height * 2)
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
		self.set_needs_display()

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
		self.set_needs_display()

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
		if self.needs_display:
			self.needs_display = False
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
	def __init__(self):
		self.cars = []
		self.lookup = np.zeros((field.n_grid_h * 4 + 4, field.n_grid_w * 4 + 4, initial_num_car), dtype=np.uint8)
		self.program = gloo.Program(car_vertex, car_fragment)
		self.textvisuals = []
		for i in xrange(initial_num_car):
			self.cars.append(Car(self, index=i))
			text = visuals.TextVisual("car %d" % i, color="white", anchor_x="left", anchor_y="top")
			text.font_size = 8
			self.textvisuals.append(text)

	def configure(self, canvas, viewport):
		for text in self.textvisuals:
			text.transforms.configure(canvas=canvas, viewport=viewport)

	def draw(self):
		positions = []
		colors = []
		for car in self.cars:
			p, c = car.compute_gl_attributes()
			positions.extend(p)
			colors.extend(c)
		self.program["position"] = positions
		self.program["color"] = colors
		self.program.draw("lines")
		for text in self.textvisuals:
			text.draw()

	def step(self):
		for car in self.cars:
			a = np.random.randint(4)
			if a == 0:
				car.action_throttle()
			elif a == 1:
				car.action_brake()
			elif a == 2:
				car.action_steer_right()
			else:
				car.action_steer_left()
			car.move()
			text = self.textvisuals[car.index]
			text.pos = car.pos[0] + 10, car.pos[1] - 10

	def find_near_cars(self, array_x, array_y, radius=1):
		start_xi = 0 if array_x - radius < 0 else array_x - radius
		start_yi = 0 if array_y - radius < 0 else array_y - radius
		end_xi = self.lookup.shape[1] if array_x + radius + 1 > self.lookup.shape[1] else array_x + radius + 1
		end_yi = self.lookup.shape[0] if array_y + radius + 1 > self.lookup.shape[0] else array_y + radius + 1
		return np.argwhere(self.lookup[start_yi:end_yi, start_xi:end_xi, :] == 1)

	def get_car_at_index(self, index=0):
		if index < len(self.cars):
			return self.cars[index]
		return None

class Car:
	lookup = np.array([[39, 38, 37, 36, 35, 34, 33], [40, 18, 17, 16, 15, 14, 32], [41, 19, 5, 4, 3, 13, 31], [42, 20, 6, -1, 2, 12, 30], [43, 21, 7, 0, 1, 11, 29], [44, 22, 23, 8, 9, 10, 28], [45, 46, 47, 24, 25, 26, 27]])
	car_width = 12.0
	car_height = 20.0
	STATE_NORMAL = 0
	STATE_REWARD = 1
	STATE_CRASHED = 2
	shape = [(-car_width/2.0, car_height/2.0), (car_width/2.0, car_height/2.0),
				(car_width/2.0, car_height/2.0), (car_width/2.0, -car_height/2.0),
				(car_width/2.0, -car_height/2.0), (-car_width/2.0, -car_height/2.0),
				(-car_width/2.0, -car_height/2.0), (-car_width/2.0, car_height/2.0),
				(0.0, car_height/2.0+1.0), (0.0, 1.0)]
	def __init__(self, manager, index=0):
		self.index = index
		self.manager = manager
		self.speed = 0
		self.steering = 0
		self.steering_unit = math.pi / 30.0
		self.state = Car.STATE_NORMAL
		self.pos = (canvas.size[0] / 2.0 + np.random.randint(400) - 200, canvas.size[1] / 2.0 + np.random.randint(400) - 200)
		self.prev_lookup_xi, self.prev_lookup_yi = field.compute_array_index_from_position(self.pos[0], self.pos[1])
		self.manager.lookup[self.prev_lookup_yi, self.prev_lookup_xi, self.index] = 1
		self.over = False

	def compute_gl_attributes(self):
		xi, yi = field.compute_array_index_from_position(self.pos[0], self.pos[1])
		sw, sh = canvas.size
		cos = math.cos(-self.steering)
		sin = math.sin(-self.steering)
		positions = []
		colors = []
		for x, y in Car.shape:
			_x = 2.0 * (x * cos - y * sin + self.pos[0]) / sw - 1
			_y = 2.0 * (x * sin + y * cos + (sh - self.pos[1])) / sh - 1
			positions.append((_x, _y))
			if self.state == Car.STATE_CRASHED:
				colors.append(color_car_crashed)
			elif self.state == Car.STATE_REWARD:
				colors.append(color_car_reward)
			else:
				colors.append(color_car_normal)
		return positions, colors

	def respawn(self):
		self.pos = (canvas.size[0] / 2.0, canvas.size[1] / 2.0)

	def get_sensor_value(self):
		# 衝突判定
		sw, sh = canvas.size
		xi, yi = field.compute_array_index_from_position(self.pos[0], self.pos[1])
		values = np.zeros((48,), dtype=np.float32)
		self.over == False

		# 壁
		blocks = field.surrounding_wal_indicis(xi, yi, 3)
		for block in blocks:
			i = Car.lookup[block[0]][block[1]]
			if i != -1:
				values[i] = 1.0

		# 他の車
		near_cars = self.manager.find_near_cars(xi, yi, 3)
		for _, __, car_index in near_cars:
			if car_index == self.index:
				continue
			target_car = self.manager.get_car_at_index(car_index)
			if target_car is None:
				continue

			direction = target_car.pos[0] - self.pos[0], target_car.pos[1] - self.pos[1]
			distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2) / float(sw)
			theta = (math.atan2(direction[1], direction[0]) + math.pi / 2.0) % (math.pi * 2.0)
			ds = 0.018
			dsi = int(distance / ds)
			if dsi <= 1:
				if dsi == 0:
					self.over == True
				values[int(theta / math.pi * 4.0)] = 1
			elif dsi == 2:
				values[int(theta / math.pi * 8.0) + 8] = 1
			elif dsi == 3:
				values[int(theta / math.pi * 12.0) + 24] = 1

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

	def move(self):
		cos = math.cos(-self.steering)
		sin = math.sin(-self.steering)
		x = -sin * self.speed		
		y = cos * self.speed
		sensors = self.get_sensor_value()
		self.state = Car.STATE_NORMAL
		if sensors[0] > 0 and self.speed > 0:
			self.speed = 0
			self.state = Car.STATE_CRASHED
			return
		if sensors[4] > 0 and self.speed < 0:
			self.speed = 0
			self.state = Car.STATE_CRASHED
			return
		if self.speed > 0:
			self.state = Car.STATE_REWARD
		if self.over:
			self.state = Car.STATE_CRASHED
		self.pos = (self.pos[0] + x, self.pos[1] - y)
		if field.is_screen_position_inside_field(self.pos[0], self.pos[1]) is False:
			self.respawn()

		xi, yi = field.compute_array_index_from_position(self.pos[0], self.pos[1])
		if xi == self.prev_lookup_xi and yi == self.prev_lookup_yi:
			return
		self.manager.lookup[self.prev_lookup_yi,self.prev_lookup_xi,self.index] = 0
		self.manager.lookup[yi, xi, self.index] = 1
		self.prev_lookup_xi = xi
		self.prev_lookup_yi = yi

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

		self._timer = app.Timer(1.0 / 20.0, connect=self.on_timer, start=True)

	def step(self):
		pass

	def on_draw(self, event):
		gloo.clear(color="#0e0e0e")
		gloo.set_viewport(0, 0, *self.physical_size)
		field.draw()
		gui.draw()
		cars.draw()

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
		# car = cars.get_car_at_index(0)
		# car.pos = event.pos[0], event.pos[1]

	def on_mouse_wheel(self, event):
		# car = cars.get_car_at_index(0)
		# car.action_steer_right()
		pass

	def toggle_wall(self, pos):
		if self.is_mouse_pressed:
			if field.is_screen_position_inside_field(pos[0], pos[1]):
				x, y = field.compute_array_index_from_position(pos[0], pos[1])
				if self.is_key_shift_pressed:
					field.destroy_wall_at_index(x, y)
				else:
					field.construct_wall_at_index(x, y)

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
		cars.configure(canvas=self, viewport=vp)
		field.set_needs_display()
		
	def on_timer(self, event):
		cars.step()
		self.update()

if __name__ == "__main__":
	canvas = Canvas()
	gui = Gui()
	field = Field()
	cars = CarManager()
	canvas.activate_zoom()
	canvas.show()
	# canvas.measure_fps()
	app.run()
