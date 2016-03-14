# -*- coding: utf-8 -*-
import numpy as np
from vispy import app
from vispy import gloo

# width, height
screen_size = (1000, 500)

# Color
color_black = np.asarray((0.0, 0.0, 0.0, 1.0))
color_field_line_base = np.asarray((231.0 / 255.0, 237.0 / 255.0, 203.0 / 255.0, 1.0))
color_field_line = 0.8 * color_black + 0.2 * color_field_line_base

# Shaders
field_line_vertex = """
attribute vec2 position;

void main() {
	gl_Position = vec4(position, 0., 1.);
}
"""

field_line_fragment = """
uniform vec4 color;

void main() {
	gl_FragColor = color;
}
"""

class Field:
	def __init__(self):
		self.program_line = gloo.Program(field_line_vertex, field_line_fragment)
		self.program_line["color"] = color_field_line

	def set_line_positions(self, screen_size):
		# スクリーンサイズ
		sw = screen_size[0]
		sh = screen_size[1]
		# パディング
		px = 40
		py = 40
		# 枠線サイズ
		lw = 0
		lh = 0
		# 伸縮比率
		rx = 1.0
		ry = 1.0
		if sw >= sh:
			lh = sh - py * 2
			lw = lh
			# フィールドは画面の左半分
			if lw > sw / 2 - px * 2:
				lw = sw / 2 - px * 2
				lh = lw
			if lh > sh - py * 2:
				lh = sh - py * 2
				lw = lh
		else:
			lw = sw / 2 - px * 2
			lh = lw

		positions = []
		n_grids = 6
		for vertical_i in xrange(n_grids + 1):
			x = lw / float(n_grids) * vertical_i + px
			x /= float(sw)
			y = py
			y /= float(sh)
			positions.append((2.0 * x - 1.0, 2.0 * y - 1.0))
			y = py + lh
			y /= float(sh)
			positions.append((2.0 * x - 1.0, 2.0 * y - 1.0))

		for horizontal_i in xrange(n_grids + 1):
			y = lh / float(n_grids) * horizontal_i + py
			y /= float(sh)
			x = px
			x /= float(sw)
			positions.append((2.0 * x - 1.0, 2.0 * y - 1.0))
			x = px + lw
			x /= float(sw)
			positions.append((2.0 * x - 1.0, 2.0 * y - 1.0))

		self.program_line["position"] = positions

	def draw(self, screen_size):
		self.set_line_positions(screen_size)
		self.program_line.draw("lines")


class Canvas(app.Canvas):
	def __init__(self):
		app.Canvas.__init__(self, size=screen_size, title="self-driving", keys="interactive")

		# Init
		self.field = Field()

		self.activate_zoom()
		self.show()

	def on_draw(self, event):
		gloo.clear()
		self.field.draw((self.width, self.height))

	def on_resize(self, event):
		self.activate_zoom()

	def activate_zoom(self):
		self.width, self.height = self.size
		gloo.set_viewport(0, 0, *self.physical_size)


if __name__ == "__main__":
	c = Canvas()
	app.run()
