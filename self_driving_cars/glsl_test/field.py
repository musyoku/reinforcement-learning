# -*- coding: utf-8 -*-
import math, time
import numpy as np
from vispy import app
from vispy import gloo

# width, height
screen_size = (1000, 600)

# グリッド数
n_grid_h = 6
n_grid_w = 8


# 色
color_black = np.asarray((0.0, 0.0, 0.0, 1.0))
color_field_bg_base = np.asarray((27.0 / 255.0, 59.0 / 255.0, 70.0 / 255.0, 1.0))
color_field_bg = 0.0 * color_black + 1.0 * color_field_bg_base
color_field_grid_base = np.asarray((232.0 / 255.0, 250.0 / 255.0, 174.0 / 255.0, 1.0))
color_field_grid = 0.8 * color_field_bg_base + 0.2 * color_field_grid_base
color_field_point = 0.5 * color_field_bg_base + 0.5 * color_field_grid_base
color_field_subdiv_base = np.asarray((66.0 / 255.0, 115.0 / 255.0, 129.0 / 255.0, 1.0))
color_field_subdiv_line = 0.8 * color_field_bg_base + 0.2 * color_field_subdiv_base
color_field_subdiv_point_base = np.asarray((134.0 / 255.0, 214.0 / 255.0, 247.0 / 255.0, 1.0))
color_field_subdiv_point = 0.2 * color_field_bg_base + 0.8 * color_field_subdiv_point_base
color_field_wall = np.asarray((241.0 / 255.0, 70.0 / 255.0, 57.0 / 255.0, 1.0))


# シェーダ
field_line_vertex = """
attribute vec2 position;

void main() {
	gl_Position = vec4(position, 0.0, 1.0);
}
"""

field_line_fragment = """
uniform vec4 color;

void main() {
	gl_FragColor = color;
}
"""

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
    const float M_PI = 3.14159265358979323846;
    const float NUM_LINES = 100.0;
	if(v_is_wall == 0.0){
		gl_FragColor = v_bg_color;
	}else{
	    float theta = M_PI / 6.0;
	    float cos_theta = cos(theta);
	    float sin_theta = sin(theta);
	    vec2 coord = gl_FragCoord.xy / screen_size;
	    float x = cos_theta * coord.x - sin_theta * coord.y;
	    float f = fract(x * NUM_LINES);
	    if (f > 0.4){
			gl_FragColor = 0.5 * v_bg_color + 0.5 * v_wall_color;
	    }else{
			gl_FragColor = 0.8 * v_bg_color + 0.2 * v_wall_color;
	    }
	}
}
"""

field_wall_vertex = """
attribute vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

field_wall_fragment = """
uniform vec4 color;
uniform vec4 bg_color;
uniform vec2 screen_size;

void main() {
    const float M_PI = 3.14159265358979323846;
    const float NUM_LINES = 100.0;
    float theta = M_PI / 6.0;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    vec2 coord = gl_FragCoord.xy / screen_size;
    float x = cos_theta * coord.x - sin_theta * coord.y;
    float f = fract(x * NUM_LINES);
    if (f > 0.4){
		gl_FragColor = color;
    }else{
		gl_FragColor = 0.3 * color + 0.7 * bg_color;
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
		self.py = 80

		self.grid_subdiv_bg, self.grid_subdiv_wall = self.load()

		self.program_grid_line = gloo.Program(field_line_vertex, field_line_fragment)
		self.program_grid_line["color"] = color_field_grid
		self.program_grid_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.program_grid_point["color"] = color_field_point
		self.program_grid_point["point_size"] = 3.0
		self.program_subdiv_line = gloo.Program(field_line_vertex, field_line_fragment)
		self.program_subdiv_line["color"] = color_field_subdiv_line
		self.program_subdiv_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.program_subdiv_point["color"] = color_field_subdiv_point
		self.program_subdiv_point["point_size"] = 1.0
		self.program_bg = gloo.Program(field_bg_vertex, field_bg_fragment)
		self.program_bg["screen_size"] = canvas.size
		self.program_wall = gloo.Program(field_wall_vertex, field_wall_fragment)
		self.program_wall["color"] = color_field_wall
		self.program_wall["bg_color"] = 0.6 * color_black + 0.4 * color_field_bg_base
		self.program_wall["screen_size"] = canvas.size

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

	def subdivision_exists(self, x, y):
		if x < 0:
			return False
		if y < 0:
			return False
		if x >= self.grid_subdiv_bg.shape[1]:
			return False
		if y >= self.grid_subdiv_bg.shape[0]:
			return False
		return True if self.grid_subdiv_bg[y, x] == 1 else False

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
			lh = lw / ratio
		return lw, lh

	def construct_wall_at_index(self, array_x, array_y):
		if array_x < 0:
			raise Exception()
		if array_y < 0:
			raise Exception()
		if array_x >= self.grid_subdiv_wall.shape[1]:
			raise Exception()
		if array_y >= self.grid_subdiv_wall.shape[0]:
			raise Exception()
		self.grid_subdiv_wall[array_y, array_x] = 1

	def destroy_wall_at_index(self, array_x, array_y):
		if array_x < 0:
			raise Exception()
		if array_y < 0:
			raise Exception()
		if array_x >= self.grid_subdiv_wall.shape[1]:
			raise Exception()
		if array_y >= self.grid_subdiv_wall.shape[0]:
			raise Exception()
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

		line_positions = []
		subdiv_line_positions = []
		for m in xrange(self.n_grid_w + 1):
			x1 = lw / float(self.n_grid_w) * m + self.px
			x1 = 2.0 * x1 / float(sw) - 1.0
			y1 = self.py
			y1 = 2.0 * y1 / float(sh) - 1.0
			line_positions.append((x1, y1))
			x2 = x1
			y2 = self.py + lh
			y2 = 2.0 * y2 / float(sh) - 1.0
			line_positions.append((x2, y2))
			# 小さいグリッド
			if m < self.n_grid_w:
				for sub_x in xrange(1, 4):
					subdiv_line_positions.append((x1 + sgw * sub_x, y1))
					subdiv_line_positions.append((x2 + sgw * sub_x, y2))

		for m in xrange(self.n_grid_h + 1):
			x1 = self.px
			x1 = 2.0 * x1 / float(sw) - 1.0
			y1 = lh / float(self.n_grid_h) * m + self.py
			y1 = 2.0 * y1 / float(sh) - 1.0
			line_positions.append((x1, y1))
			x2 = self.px + lw
			x2 = 2.0 * x2 / float(sw) - 1.0
			y2 = y1
			line_positions.append((x2, y2))
			# 小さいグリッド
			if m < self.n_grid_h:
				for sub_y in xrange(1, 4):
					subdiv_line_positions.append((x1, y1 + sgh * sub_y))
					subdiv_line_positions.append((x2, y2 + sgh * sub_y))

		self.program_grid_line["position"] = line_positions
		self.program_subdiv_line["position"] = subdiv_line_positions

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
		wall_positions = []
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

				if self.grid_subdiv_wall[h, w] == 1:
					wall_positions.append((x_start + sgw * w, y_start + sgh * h))
					wall_positions.append((x_start + sgw * (w + 1), y_start + sgh * h))
					wall_positions.append((x_start + sgw * w, y_start + sgh * (h + 1)))

					wall_positions.append((x_start + sgw * (w + 1), y_start + sgh * h))
					wall_positions.append((x_start + sgw * w, y_start + sgh * (h + 1)))
					wall_positions.append((x_start + sgw * (w + 1), y_start + sgh * (h + 1)))


		np.random.seed(int(time.time()))
		self.program_bg["position"] = bg_positions
		self.program_bg["bg_color"] = bg_colors
		self.program_bg["wall_color"] = wall_colors
		self.program_bg["is_wall"] = np.asarray(is_wall, dtype=np.float32)

		self.program_wall["position"] = wall_positions

	def draw(self):
		self.set_positions()
		self.program_bg.draw("triangles")
		# if self.enable_grid:
		# 	self.program_subdiv_line.draw("lines")
		# # self.program_wall.draw("triangles")
		if self.enable_grid:
			self.program_subdiv_point.draw("points")
			# self.program_grid_line.draw("lines")
			self.program_grid_point.draw("points")

	def draw_wall(self):
		self.set_positions()
		self.program_wall.draw("triangles")

class Canvas(app.Canvas):
	def __init__(self):
		app.Canvas.__init__(self, size=screen_size, title="self-driving", keys="interactive")

		self.activate_zoom()
		self.show()

		self.is_mouse_pressed = False

	def on_draw(self, event):
		gloo.clear()
		field.draw()

	def on_resize(self, event):
		self.activate_zoom()
		print "#on_resize()", (self.width, self.height)

	def on_mouse_press(self, event):
		self.is_mouse_pressed = True

	def on_mouse_release(self, event):
		self.is_mouse_pressed = False

	def on_mouse_move(self, event):
		if self.is_mouse_pressed:
			if field.is_screen_position_inside_field(event.pos[0], event.pos[1]):
				x, y = field.compute_array_index_from_position(event.pos[0], event.pos[1])
				field.construct_wall_at_index(x, y)
				self.update()

	def activate_zoom(self):
		self.width, self.height = self.size
		gloo.set_viewport(0, 0, *self.physical_size)


if __name__ == "__main__":
	canvas = Canvas()
	field = Field()
	app.run()
