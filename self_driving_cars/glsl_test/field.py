# -*- coding: utf-8 -*-
import math
import numpy as np
from vispy import app
from vispy import gloo

# width, height
screen_size = (1000, 600)

# グリッド数
n_grid_h = 6
n_grid_w = 8

# 背景
grid_subdiv_bg = np.ones((n_grid_h * 4 + 4, n_grid_w * 4 + 4), dtype=np.uint8)
# 壁
grid_subdiv_wall = np.zeros((n_grid_h * 4 + 4, n_grid_w * 4 + 4), dtype=np.uint8)

# 色
color_black = np.asarray((0.0, 0.0, 0.0, 1.0))
color_field_bg_base = np.asarray((27.0 / 255.0, 59.0 / 255.0, 70.0 / 255.0, 1.0))
color_field_bg = 0.0 * color_black + 1.0 * color_field_bg_base
color_field_line_base = np.asarray((232.0 / 255.0, 250.0 / 255.0, 174.0 / 255.0, 1.0))
color_field_line = 0.6 * color_field_bg_base + 0.4 * color_field_line_base
color_field_point = 0.4 * color_field_bg_base + 0.6 * color_field_line_base
color_field_subdiv_base = np.asarray((66.0 / 255.0, 115.0 / 255.0, 129.0 / 255.0, 1.0))
color_field_subdiv_line = 0.8 * color_field_bg_base + 0.2 * color_field_subdiv_base
color_field_subdiv_point = np.asarray((134.0 / 255.0, 214.0 / 255.0, 247.0 / 255.0, 1.0))

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
attribute vec4 color;
varying vec4 v_color;

void main() {
	v_color = color;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

field_bg_fragment = """
varying vec4 v_color;

void main() {
	gl_FragColor = v_color;
}
"""

class Field:
	def __init__(self, canvas):
		self.canvas = canvas

		# Settings
		self.enable_grid = True

		self.program_grid_line = gloo.Program(field_line_vertex, field_line_fragment)
		self.program_grid_line["color"] = color_field_line
		self.program_grid_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.program_grid_point["color"] = color_field_point
		self.program_grid_point["point_size"] = 3.0
		self.program_subdiv_line = gloo.Program(field_line_vertex, field_line_fragment)
		self.program_subdiv_line["color"] = color_field_subdiv_line
		self.program_subdiv_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.program_subdiv_point["color"] = color_field_subdiv_point
		self.program_subdiv_point["point_size"] = 1.0
		self.program_bg = gloo.Program(field_bg_vertex, field_bg_fragment)
		# self.program_bg["color"] = color_field_bg

	def set_positions(self):
		# スクリーンサイズ
		sw = self.canvas.width
		sh = self.canvas.height
		# パディング
		px = 80
		py = 80
		# 枠線サイズ
		lw = 0
		lh = 0
		ratio = n_grid_h / float(n_grid_w)
		if sw >= sh:
			lh = sh - py * 2
			lw = lh / ratio
			# フィールドは画面の左半分
			if lw > sw / 1.3 - px * 2:
				lw = sw / 1.3 - px * 2
				lh = lw * ratio
			if lh > sh - py * 2:
				lh = sh - py * 2
				lw = lh / ratio
		else:
			lw = sw / 1.3 - px * 2
			lh = lw / ratio

		# 小さいグリッド
		sgw = lw / float(n_grid_w) / 4.0 / float(sw) * 2.0
		sgh = lh / float(n_grid_h) / 4.0 / float(sh) * 2.0

		line_positions = []
		subdiv_line_positions = []
		for m in xrange(n_grid_w + 1):
			x1 = lw / float(n_grid_w) * m + px
			x1 = 2.0 * x1 / float(sw) - 1.0
			y1 = py
			y1 = 2.0 * y1 / float(sh) - 1.0
			line_positions.append((x1, y1))
			x2 = x1
			y2 = py + lh
			y2 = 2.0 * y2 / float(sh) - 1.0
			line_positions.append((x2, y2))
			# 小さいグリッド
			if m < n_grid_w:
				for sub_x in xrange(1, 4):
					subdiv_line_positions.append((x1 + sgw * sub_x, y1))
					subdiv_line_positions.append((x2 + sgw * sub_x, y2))

		for m in xrange(n_grid_h + 1):
			x1 = px
			x1 = 2.0 * x1 / float(sw) - 1.0
			y1 = lh / float(n_grid_h) * m + py
			y1 = 2.0 * y1 / float(sh) - 1.0
			line_positions.append((x1, y1))
			x2 = px + lw
			x2 = 2.0 * x2 / float(sw) - 1.0
			y2 = y1
			line_positions.append((x2, y2))
			# 小さいグリッド
			if m < n_grid_h:
				for sub_y in xrange(1, 4):
					subdiv_line_positions.append((x1, y1 + sgh * sub_y))
					subdiv_line_positions.append((x2, y2 + sgh * sub_y))

		self.program_grid_line["position"] = line_positions
		self.program_subdiv_line["position"] = subdiv_line_positions

		point_positions = []
		for nh in xrange(n_grid_w + 1):
			x = lw / float(n_grid_w) * nh + px
			x /= float(sw)
			for nv in xrange(n_grid_h + 1):
				y = lh / float(n_grid_h) * nv + py
				y /= float(sh)
				point_positions.append((2.0 * x - 1.0, 2.0 * y - 1.0))

		self.program_grid_point["position"] = point_positions

		subdiv_point_positions = []
		# 大きいグリッドの交差点
		for nh in xrange(n_grid_w + 1):
			x = lw / float(n_grid_w) * nh + px
			x = 2.0 * x / float(sw) - 1.0
			for nv in xrange(n_grid_h + 1):
				y = lh / float(n_grid_h) * nv + py
				y = 2.0 * y / float(sh) - 1.0
				# 小さいグリッド
				for sub_y in xrange(5):
					_y = y + sgh * sub_y
					for sub_x in xrange(5):
						_x = x + sgw * sub_x
						# x, yそれぞれ2マス分ずらす
						subdiv_point_positions.append((_x - sgw * 2.0, _y - sgh * 2.0))
		self.program_subdiv_point["position"] = subdiv_point_positions

		bg_positions = []
		bg_opacity = []
		# x, yそれぞれ2マス分ずらす
		x_start = 2.0 * px / float(sw) - 1.0 - sgw * 2.0
		y_start = 2.0 * py / float(sh) - 1.0 - sgh * 2.0
		for h in xrange(grid_subdiv_bg.shape[0]):
			for w in xrange(grid_subdiv_bg.shape[1]):
				if grid_subdiv_bg[h, w] == 1:
					bg_positions.append((x_start + sgw * w, y_start + sgh * h))
					bg_positions.append((x_start + sgw * (w + 1), y_start + sgh * h))
					bg_positions.append((x_start + sgw * w, y_start + sgh * (h + 1)))

					bg_positions.append((x_start + sgw * (w + 1), y_start + sgh * h))
					bg_positions.append((x_start + sgw * w, y_start + sgh * (h + 1)))
					bg_positions.append((x_start + sgw * (w + 1), y_start + sgh * (h + 1)))

					distance = (abs(w - grid_subdiv_bg.shape[1] / 2.0) / float(grid_subdiv_bg.shape[1]), abs(h - grid_subdiv_bg.shape[0] / 2.0) / float(grid_subdiv_bg.shape[0]))
					weight = 1.0 - math.sqrt(distance[0] ** 2 + distance[1] ** 2)
					weight = weight / 2.0 + 0.5
					opacity = np.random.uniform(0.9 * math.sqrt(weight), 1.0) * weight
					color = opacity * color_field_bg_base + (1.0 - opacity) * color_black
					bg_opacity.append(color)
					bg_opacity.append(color)
					bg_opacity.append(color)
					bg_opacity.append(color)
					bg_opacity.append(color)
					bg_opacity.append(color)

		self.program_bg["position"] = bg_positions
		self.program_bg["color"] = bg_opacity

	def draw(self):
		self.set_positions()
		self.program_bg.draw("triangles")
		if self.enable_grid:
			self.program_subdiv_line.draw("lines")
			self.program_subdiv_point.draw("points")
			self.program_grid_line.draw("lines")
			self.program_grid_point.draw("points")


class Canvas(app.Canvas):
	def __init__(self):
		app.Canvas.__init__(self, size=screen_size, title="self-driving", keys="interactive")

		# Init
		self.field = Field(self)

		self.activate_zoom()
		self.show()

	def on_draw(self, event):
		gloo.clear()
		self.field.draw()

	def on_resize(self, event):
		self.activate_zoom()
		print "#on_resize()", (self.width, self.height)

	def activate_zoom(self):
		self.width, self.height = self.size
		gloo.set_viewport(0, 0, *self.physical_size)


if __name__ == "__main__":
	c = Canvas()
	app.run()
