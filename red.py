import math
import random
import tkinter as tk
from dataclasses import dataclass


WIDTH = 940
HEIGHT = 560
FIELD = (28, 42, 912, 532)
GOAL_TOP = 238
GOAL_BOTTOM = 336
POPULATION_SIZE = 20
ELITE_COUNT = 3
EPISODE_LENGTH = 18.0
NETWORK_INPUTS = 12
NETWORK_HIDDEN = 8
MATCH_FIELDS = (
	"blue_players",
	"red_players",
	"ball_x",
	"ball_y",
	"ball_vx",
	"ball_vy",
	"blue_score",
	"red_score",
	"time_left",
	"progress_score",
	"previous_ball_x",
	"touches",
	"kicks",
	"goal_flash",
)

COLORS = {
	"background": "#101813",
	"panel": "#17221b",
	"panel_light": "#202e25",
	"text": "#f2f5ed",
	"muted": "#a4b1a5",
	"green": "#32754c",
	"green_dark": "#2c6b44",
	"line": "#deead9",
	"blue": "#56b9f2",
	"blue_dark": "#267bb4",
	"red": "#f07868",
	"gold": "#f2c65d",
	"accent": "#b8e986",
}


def new_genome():
	return [
		random.gauss(0, 0.48)
		for _ in range((NETWORK_INPUTS + 1) * NETWORK_HIDDEN + (NETWORK_HIDDEN + 1) * 3)
	]


def neural_move(genome, inputs):
	cursor = 0
	hidden = []
	for _ in range(NETWORK_HIDDEN):
		value = genome[cursor + NETWORK_INPUTS]
		for index in range(NETWORK_INPUTS):
			value += genome[cursor + index] * inputs[index]
		hidden.append(math.tanh(value))
		cursor += NETWORK_INPUTS + 1

	outputs = []
	for _ in range(3):
		value = genome[cursor + NETWORK_HIDDEN]
		for index in range(NETWORK_HIDDEN):
			value += genome[cursor + index] * hidden[index]
		outputs.append(math.tanh(value))
		cursor += NETWORK_HIDDEN + 1
	return outputs


@dataclass
class Player:
	x: float
	y: float
	team: str
	keeper: bool = False
	vx: float = 0.0
	vy: float = 0.0
	kick_wait: float = 0.0


class EvolutionFootball:
	def __init__(self, root):
		self.root = root
		self.root.title("EVOLVE | Laboratorio de fútbol")
		self.root.configure(bg=COLORS["background"])
		self.root.resizable(False, False)
		self.root.bind_all("<KeyPress>", self.on_key_down)
		self.root.bind_all("<KeyRelease>", self.on_key_up)

		self.population = [new_genome() for _ in range(POPULATION_SIZE)]
		self.generation = 1
		self.candidate = 0
		self.matches = []
		self.fitnesses = []
		self.best_fitness = float("-inf")
		self.history = []
		self.games_played = 0
		self.speed_options = [1, 2, 4, 8]
		self.speed_index = 1
		self.paused = False
		self.manual = False
		self.keys = set()
		self.last_time = None

		self.build_ui()
		self.reset_matches()
		self.update_status()
		self.root.after(20, self.tick)

	def build_ui(self):
		header = tk.Frame(self.root, bg=COLORS["background"], padx=22, pady=15)
		header.pack(fill="x")
		tk.Label(
			header,
			text="EVOLVE",
			bg=COLORS["background"],
			fg=COLORS["accent"],
			font=("Segoe UI", 19, "bold"),
		).pack(side="left")
		tk.Label(
			header,
			text="FÚTBOL · APRENDIZAJE POR EVOLUCIÓN",
			bg=COLORS["background"],
			fg=COLORS["muted"],
			font=("Segoe UI", 10, "bold"),
		).pack(side="left", padx=(14, 0), pady=(5, 0))
		self.status_label = tk.Label(
			header,
			text="",
			bg=COLORS["background"],
			fg=COLORS["accent"],
			font=("Segoe UI", 10, "bold"),
		)
		self.status_label.pack(side="right", padx=(0, 4))

		body = tk.Frame(self.root, bg=COLORS["background"], padx=22, pady=0)
		body.pack()
		self.canvas = tk.Canvas(
			body,
			width=WIDTH,
			height=HEIGHT,
			bg=COLORS["green"],
			highlightthickness=0,
		)
		self.canvas.grid(row=0, column=0, sticky="n")

		sidebar = tk.Frame(body, bg=COLORS["background"], width=288)
		sidebar.grid(row=0, column=1, sticky="ns", padx=(20, 0))
		sidebar.grid_propagate(False)

		self.score_label = tk.Label(
			sidebar,
			text="0  :  0",
			bg=COLORS["background"],
			fg=COLORS["text"],
			font=("Segoe UI", 30, "bold"),
		)
		self.score_label.pack(anchor="w", pady=(0, 0))
		tk.Label(
			sidebar,
			text="AZUL  /  RIVAL",
			bg=COLORS["background"],
			fg=COLORS["muted"],
			font=("Segoe UI", 9, "bold"),
		).pack(anchor="w", pady=(0, 17))

		self.generation_label = self.add_stat(sidebar, "GENERACIÓN", "1")
		self.candidate_label = self.add_stat(sidebar, "PARTIDO A COLOR", "01 / 20")
		self.fitness_label = self.add_stat(sidebar, "MEJOR RESULTADO", "—")
		self.games_label = self.add_stat(sidebar, "PARTIDOS EVALUADOS", "0")

		tk.Label(
			sidebar,
			text="PROGRESO EVOLUTIVO",
			bg=COLORS["background"],
			fg=COLORS["muted"],
			font=("Segoe UI", 9, "bold"),
		).pack(anchor="w", pady=(19, 7))
		self.chart = tk.Canvas(
			sidebar,
			width=276,
			height=94,
			bg=COLORS["panel"],
			highlightthickness=0,
		)
		self.chart.pack(anchor="w")

		controls = tk.Frame(sidebar, bg=COLORS["background"])
		controls.pack(fill="x", pady=(17, 0))
		self.pause_button = self.make_button(controls, "Ⅱ  PAUSAR", self.toggle_pause)
		self.pause_button.pack(fill="x", pady=(0, 7))
		self.manual_button = self.make_button(controls, "⌨  JUGAR TÚ · WASD", self.toggle_manual)
		self.manual_button.pack(fill="x", pady=(0, 7))
		self.view_button = self.make_button(controls, "◉  SIGUIENTE PARTIDO", self.next_match)
		self.view_button.pack(fill="x", pady=(0, 7))
		self.speed_button = self.make_button(controls, "VELOCIDAD  ·  ×2", self.cycle_speed)
		self.speed_button.pack(fill="x", pady=(0, 7))
		self.reset_button = self.make_button(controls, "↻  NUEVA EVOLUCIÓN", self.restart)
		self.reset_button.pack(fill="x")

		tk.Label(
			sidebar,
			text="Los 20 cerebros juegan partidos propios en paralelo. Goles y avances deciden qué genes sobreviven; el partido resaltado se ve a color.",
			bg=COLORS["background"],
			fg=COLORS["muted"],
			font=("Segoe UI", 9),
			justify="left",
			wraplength=274,
		).pack(anchor="w", pady=(15, 0))

	def add_stat(self, parent, title, value):
		row = tk.Frame(parent, bg=COLORS["panel"], padx=12, pady=8)
		row.pack(fill="x", pady=(0, 5))
		tk.Label(
			row,
			text=title,
			bg=COLORS["panel"],
			fg=COLORS["muted"],
			font=("Segoe UI", 8, "bold"),
		).pack(anchor="w")
		value_label = tk.Label(
			row,
			text=value,
			bg=COLORS["panel"],
			fg=COLORS["text"],
			font=("Segoe UI", 13, "bold"),
		)
		value_label.pack(anchor="w", pady=(2, 0))
		return value_label

	def make_button(self, parent, text, command):
		return tk.Button(
			parent,
			text=text,
			command=command,
			anchor="w",
			padx=12,
			pady=9,
			bg=COLORS["panel_light"],
			fg=COLORS["text"],
			activebackground=COLORS["green"],
			activeforeground=COLORS["text"],
			relief="flat",
			bd=0,
			cursor="hand2",
			font=("Segoe UI", 9, "bold"),
		)

	def reset_match(self):
		left, right = FIELD[0], FIELD[2]
		self.blue_players = [
			Player(left + 260, 235, "blue"),
			Player(left + 260, 345, "blue"),
			Player(left + 35, HEIGHT / 2, "blue", keeper=True),
		]
		self.red_players = [
			Player(right - 260, 235, "red"),
			Player(right - 260, 345, "red"),
			Player(right - 35, HEIGHT / 2, "red", keeper=True),
		]
		self.ball_x = WIDTH / 2
		self.ball_y = HEIGHT / 2
		self.ball_vx = 0.0
		self.ball_vy = 0.0
		self.blue_score = 0
		self.red_score = 0
		self.time_left = EPISODE_LENGTH
		self.progress_score = 0.0
		self.previous_ball_x = self.ball_x
		self.touches = 0
		self.kicks = 0
		self.goal_flash = 0.0

	def reset_matches(self):
		self.matches = []
		for _ in range(POPULATION_SIZE):
			self.reset_match()
			self.matches.append({field: getattr(self, field) for field in MATCH_FIELDS})
		self.load_match(self.candidate)

	def load_match(self, index):
		for field, value in self.matches[index].items():
			setattr(self, field, value)

	def save_match(self, index):
		self.matches[index] = {field: getattr(self, field) for field in MATCH_FIELDS}

	def field_inputs(self, player, teammates, opponents):
		left, top, right, bottom = FIELD
		nearest_opponent = min(opponents, key=lambda other: math.hypot(other.x - player.x, other.y - player.y))
		nearest_teammate = min(
			(other for other in teammates if other is not player),
			key=lambda other: math.hypot(other.x - player.x, other.y - player.y),
		)
		return [
			(player.x - (left + right) / 2) / (right - left),
			(player.y - (top + bottom) / 2) / (bottom - top),
			(self.ball_x - player.x) / (right - left),
			(self.ball_y - player.y) / (bottom - top),
			self.ball_vx / 260,
			self.ball_vy / 260,
			(right - player.x) / (right - left),
			(HEIGHT / 2 - player.y) / (bottom - top),
			(nearest_opponent.x - player.x) / (right - left),
			(nearest_opponent.y - player.y) / (bottom - top),
			(nearest_teammate.x - player.x) / (right - left),
			(nearest_teammate.y - player.y) / (bottom - top),
		]

	def steer_toward(self, player, target_x, target_y, speed, dt):
		dx = target_x - player.x
		dy = target_y - player.y
		distance = math.hypot(dx, dy)
		if distance > 1:
			player.vx = dx / distance * speed
			player.vy = dy / distance * speed
		else:
			player.vx = 0.0
			player.vy = 0.0
		self.move_player(player, dt)

	def move_player(self, player, dt):
		left, top, right, bottom = FIELD
		player.x += player.vx * dt
		player.y += player.vy * dt
		player.kick_wait = max(0.0, player.kick_wait - dt)
		if player.keeper:
			player.x = max(left + 20, min(right - 20, player.x))
		else:
			player.x = max(left + 20, min(right - 20, player.x))
		player.y = max(top + 20, min(bottom - 20, player.y))

	def move_keeper(self, player, dt):
		target_x = FIELD[0] + 37 if player.team == "blue" else FIELD[2] - 37
		target_y = max(GOAL_TOP + 18, min(GOAL_BOTTOM - 18, self.ball_y))
		self.steer_toward(player, target_x, target_y, 155, dt)

	def move_opponent(self, player, dt):
		if player.keeper:
			self.move_keeper(player, dt)
			return
		offset = -48 if player.y < HEIGHT / 2 else 48
		target_x = self.ball_x + 18
		target_y = self.ball_y + offset
		if self.ball_x < WIDTH * 0.36:
			target_x = WIDTH * 0.58
		self.steer_toward(player, target_x, target_y, 144, dt)

	def kick_ball(self, player, direction):
		distance = math.hypot(self.ball_x - player.x, self.ball_y - player.y)
		if distance > 33 or player.kick_wait > 0:
			return False
		aim_x = 1 if direction == "blue" else -1
		aim_y = (HEIGHT / 2 - self.ball_y) / 200
		aim_length = math.hypot(aim_x, aim_y) or 1
		self.ball_vx += aim_x / aim_length * 240 + player.vx * 0.12
		self.ball_vy += aim_y / aim_length * 180 + player.vy * 0.1
		self.ball_vx = max(-310, min(310, self.ball_vx))
		self.ball_vy = max(-300, min(300, self.ball_vy))
		player.kick_wait = 0.55
		if direction == "blue":
			self.kicks += 1
		return True

	def update_players(self, dt, genome_index=None):
		blue_field = self.blue_players[:2]
		if genome_index is None:
			genome_index = self.candidate
		for player in self.blue_players:
			if player.keeper:
				self.move_keeper(player, dt)
				continue
			if self.manual and player is blue_field[0]:
				horizontal = int("d" in self.keys) - int("a" in self.keys)
				vertical = int("s" in self.keys) - int("w" in self.keys)
				length = math.hypot(horizontal, vertical) or 1
				player.vx = horizontal / length * 185
				player.vy = vertical / length * 185
				self.move_player(player, dt)
				if "space" in self.keys:
					self.kick_ball(player, "blue")
			else:
				outputs = neural_move(
					self.population[genome_index],
					self.field_inputs(player, self.blue_players, self.red_players),
				)
				player.vx = outputs[0] * 175
				player.vy = outputs[1] * 175
				self.move_player(player, dt)
				if outputs[2] > 0.12:
					self.kick_ball(player, "blue")

		for player in self.red_players:
			self.move_opponent(player, dt)
			if not player.keeper:
				self.kick_ball(player, "red")

	def update_ball(self, dt):
		left, top, right, bottom = FIELD
		self.ball_x += self.ball_vx * dt
		self.ball_y += self.ball_vy * dt
		drag = max(0.0, 1.0 - 0.8 * dt)
		self.ball_vx *= drag
		self.ball_vy *= drag

		if self.ball_y < top + 9 or self.ball_y > bottom - 9:
			self.ball_y = max(top + 9, min(bottom - 9, self.ball_y))
			self.ball_vy *= -0.78

		if self.ball_x < left + 8 or self.ball_x > right - 8:
			if GOAL_TOP < self.ball_y < GOAL_BOTTOM:
				if self.ball_x < left + 8:
					self.red_score += 1
				else:
					self.blue_score += 1
				self.goal_flash = 0.8
				self.ball_x, self.ball_y = WIDTH / 2, HEIGHT / 2
				self.ball_vx, self.ball_vy = 0.0, 0.0
				self.place_players_for_kickoff()
			else:
				self.ball_x = max(left + 8, min(right - 8, self.ball_x))
				self.ball_vx *= -0.76

		self.progress_score += (self.ball_x - self.previous_ball_x) * 0.035
		self.previous_ball_x = self.ball_x
		for player in self.blue_players:
			distance = math.hypot(self.ball_x - player.x, self.ball_y - player.y)
			if distance < 28:
				self.touches += dt * 8
				self.ball_vx += (self.ball_x - player.x) * dt * 0.85
				self.ball_vy += (self.ball_y - player.y) * dt * 0.85
		for player in self.red_players:
			distance = math.hypot(self.ball_x - player.x, self.ball_y - player.y)
			if distance < 28:
				self.ball_vx += (self.ball_x - player.x) * dt * 0.8
				self.ball_vy += (self.ball_y - player.y) * dt * 0.8

	def place_players_for_kickoff(self):
		for index, player in enumerate(self.blue_players[:2]):
			player.x = FIELD[0] + 260
			player.y = HEIGHT / 2 + (index * 2 - 1) * 56
		for index, player in enumerate(self.red_players[:2]):
			player.x = FIELD[2] - 260
			player.y = HEIGHT / 2 + (index * 2 - 1) * 56
		for player in self.blue_players + self.red_players:
			player.vx = player.vy = 0.0

	def finish_generation(self):
		self.fitnesses = [
			match["blue_score"] * 30
			- match["red_score"] * 24
			+ match["progress_score"]
			+ match["touches"] * 0.08
			+ match["kicks"] * 0.16
			for match in self.matches
		]
		self.games_played += POPULATION_SIZE
		self.evolve()
		self.reset_matches()

	def evolve(self):
		ranked = sorted(zip(self.fitnesses, self.population), key=lambda item: item[0], reverse=True)
		top_score = ranked[0][0]
		self.best_fitness = max(self.best_fitness, top_score)
		self.history.append(sum(self.fitnesses) / len(self.fitnesses))
		self.history = self.history[-18:]

		next_population = [genome[:] for _, genome in ranked[:ELITE_COUNT]]
		parent_pool = ranked[: max(ELITE_COUNT + 1, POPULATION_SIZE // 2)]
		while len(next_population) < POPULATION_SIZE:
			parent = random.choice(parent_pool)[1]
			child = parent[:]
			for index in range(len(child)):
				if random.random() < 0.16:
					child[index] += random.gauss(0, 0.38)
				if random.random() < 0.006:
					child[index] = random.gauss(0, 0.5)
			next_population.append(child)

		self.population = next_population
		self.generation += 1
		self.candidate = 0
		self.fitnesses = []
		self.update_chart()
		self.update_status()

	def tick(self):
		now = self.root.tk.call("clock", "milliseconds")
		if self.last_time is None:
			self.last_time = now
		elapsed = min(0.06, max(0.0, (now - self.last_time) / 1000))
		self.last_time = now
		if not self.paused:
			dt = elapsed * self.speed_options[self.speed_index]
			if self.manual:
				self.load_match(self.candidate)
				self.goal_flash = max(0.0, self.goal_flash - elapsed)
				self.update_players(dt, self.candidate)
				self.update_ball(dt)
				self.save_match(self.candidate)
			else:
				for index in range(POPULATION_SIZE):
					self.load_match(index)
					self.goal_flash = max(0.0, self.goal_flash - elapsed)
					self.update_players(dt, index)
					self.update_ball(dt)
					self.time_left -= dt
					self.save_match(index)
				if self.matches[0]["time_left"] <= 0:
					self.finish_generation()

		self.load_match(self.candidate)
		self.draw()
		self.update_status()
		self.root.after(20, self.tick)

	def draw_field(self):
		canvas = self.canvas
		left, top, right, bottom = FIELD
		canvas.delete("all")
		canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill="#16261b", outline="")
		canvas.create_rectangle(left, top, right, bottom, fill=COLORS["green"], outline="")

		stripe_width = (right - left) / 10
		for index in range(10):
			if index % 2 == 0:
				x1 = left + index * stripe_width
				canvas.create_rectangle(x1, top, x1 + stripe_width, bottom, fill=COLORS["green_dark"], outline="")

		canvas.create_rectangle(left, top, right, bottom, outline=COLORS["line"], width=2)
		canvas.create_line(WIDTH / 2, top, WIDTH / 2, bottom, fill=COLORS["line"], width=2)
		canvas.create_oval(WIDTH / 2 - 68, HEIGHT / 2 - 68, WIDTH / 2 + 68, HEIGHT / 2 + 68, outline=COLORS["line"], width=2)
		canvas.create_oval(WIDTH / 2 - 3, HEIGHT / 2 - 3, WIDTH / 2 + 3, HEIGHT / 2 + 3, fill=COLORS["line"], outline="")

		box_top, box_bottom = HEIGHT / 2 - 152, HEIGHT / 2 + 152
		small_top, small_bottom = HEIGHT / 2 - 84, HEIGHT / 2 + 84
		canvas.create_rectangle(left, box_top, left + 145, box_bottom, outline=COLORS["line"], width=2)
		canvas.create_rectangle(right - 145, box_top, right, box_bottom, outline=COLORS["line"], width=2)
		canvas.create_rectangle(left, small_top, left + 54, small_bottom, outline=COLORS["line"], width=2)
		canvas.create_rectangle(right - 54, small_top, right, small_bottom, outline=COLORS["line"], width=2)
		canvas.create_oval(left + 94, HEIGHT / 2 - 4, left + 102, HEIGHT / 2 + 4, fill=COLORS["line"], outline="")
		canvas.create_oval(right - 102, HEIGHT / 2 - 4, right - 94, HEIGHT / 2 + 4, fill=COLORS["line"], outline="")
		canvas.create_arc(left + 98, HEIGHT / 2 - 68, left + 230, HEIGHT / 2 + 68, start=300, extent=120, style="arc", outline=COLORS["line"], width=2)
		canvas.create_arc(right - 230, HEIGHT / 2 - 68, right - 98, HEIGHT / 2 + 68, start=120, extent=120, style="arc", outline=COLORS["line"], width=2)

		canvas.create_rectangle(left - 20, GOAL_TOP, left, GOAL_BOTTOM, fill="#d8e5d0", outline=COLORS["line"], width=2)
		canvas.create_rectangle(right, GOAL_TOP, right + 20, GOAL_BOTTOM, fill="#d8e5d0", outline=COLORS["line"], width=2)
		canvas.create_rectangle(left + 1, GOAL_TOP + 4, left + 13, GOAL_BOTTOM - 4, fill="#b4c4b2", outline="")
		canvas.create_rectangle(right - 13, GOAL_TOP + 4, right - 1, GOAL_BOTTOM - 4, fill="#b4c4b2", outline="")

	def draw_player(self, player, index, ghost=False):
		color = COLORS[player.team]
		radius = 17 if player.keeper else 15
		canvas = self.canvas
		x, y = player.x, player.y
		if ghost:
			canvas.create_oval(
				x - radius,
				y - radius,
				x + radius,
				y + radius,
				fill=color,
				stipple="gray25",
				outline="",
			)
			return
		canvas.create_oval(x - radius + 2, y - radius + 4, x + radius + 2, y + radius + 4, fill="#193121", outline="")
		canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="#f3f5ee", width=2)
		if player.keeper:
			canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#f3f5ee", outline="")
		else:
			direction = 1 if player.team == "blue" else -1
			canvas.create_oval(x + direction * 3 - 3, y - 4, x + direction * 3 + 3, y + 2, fill="#193121", outline="")
			if player.team == "blue" and index == 0:
				canvas.create_oval(x - 22, y - 25, x + 22, y - 21, fill=COLORS["accent"], outline="")

	def draw_ball(self, ghost=False):
		if ghost:
			self.canvas.create_oval(
				self.ball_x - 9,
				self.ball_y - 9,
				self.ball_x + 9,
				self.ball_y + 9,
				fill=COLORS["gold"],
				stipple="gray25",
				outline="",
			)
			return
		self.canvas.create_oval(
			self.ball_x - 9,
			self.ball_y - 9,
			self.ball_x + 9,
			self.ball_y + 9,
			fill=COLORS["gold"],
			outline="#fff2bb",
			width=2,
		)
		self.canvas.create_oval(self.ball_x - 2, self.ball_y - 2, self.ball_x + 2, self.ball_y + 2, fill="#594b2b", outline="")

	def draw(self):
		self.draw_field()
		for index in range(POPULATION_SIZE):
			if index == self.candidate:
				continue
			self.load_match(index)
			for player_index, player in enumerate(self.blue_players):
				self.draw_player(player, player_index, ghost=True)
			for player_index, player in enumerate(self.red_players):
				self.draw_player(player, player_index, ghost=True)
			self.draw_ball(ghost=True)

		self.load_match(self.candidate)
		for index, player in enumerate(self.blue_players):
			self.draw_player(player, index)
		for index, player in enumerate(self.red_players):
			self.draw_player(player, index)
		self.draw_ball()

		self.canvas.create_rectangle(0, 0, WIDTH, 32, fill="#101813", outline="")
		self.canvas.create_text(
			WIDTH / 2,
			17,
			text=f"GENERACIÓN {self.generation:02d}     ·     {POPULATION_SIZE} PARTIDOS SIMULTÁNEOS     ·     VISTA {self.candidate + 1:02d}     ·     {max(0, int(self.time_left)):02d} s",
			fill=COLORS["text"],
			font=("Segoe UI", 10, "bold"),
		)
		if self.goal_flash > 0:
			self.canvas.create_text(
				WIDTH / 2,
				HEIGHT / 2 - 100,
				text="¡GOL!",
				fill=COLORS["gold"],
				font=("Segoe UI", 30, "bold"),
			)
		if self.paused:
			self.canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill="#101813", stipple="gray50", outline="")
			self.canvas.create_text(WIDTH / 2, HEIGHT / 2, text="PAUSA", fill=COLORS["text"], font=("Segoe UI", 28, "bold"))
		elif self.manual:
			self.canvas.create_text(
				110,
				HEIGHT - 20,
				text="MODO MANUAL  ·  WASD MOVER  ·  ESPACIO CHUTAR",
				fill=COLORS["text"],
				font=("Segoe UI", 9, "bold"),
			)

	def update_status(self):
		self.score_label.configure(text=f"{self.blue_score}  :  {self.red_score}")
		self.generation_label.configure(text=f"{self.generation:02d}")
		self.candidate_label.configure(text=f"{self.candidate + 1:02d} / {POPULATION_SIZE}")
		best = "—" if self.best_fitness == float("-inf") else f"{self.best_fitness:.1f} pts"
		self.fitness_label.configure(text=best)
		self.games_label.configure(text=str(self.games_played))
		if self.manual:
			status = "CONTROL MANUAL"
		elif self.paused:
			status = "SIMULACIÓN PAUSADA"
		else:
			status = "EVOLUCIÓN ACTIVA"
		self.status_label.configure(text=status)

	def update_chart(self):
		self.chart.delete("all")
		values = self.history
		if not values:
			self.chart.create_text(138, 47, text="La primera generación está jugando", fill=COLORS["muted"], font=("Segoe UI", 9))
			return
		low = min(values)
		high = max(values)
		spread = max(1.0, high - low)
		self.chart.create_line(12, 77, 266, 77, fill="#334438")
		points = []
		for index, value in enumerate(values):
			x = 14 + index * (250 / max(1, len(values) - 1))
			y = 72 - ((value - low) / spread) * 47
			points.extend((x, y))
		if len(points) >= 4:
			self.chart.create_line(*points, fill=COLORS["accent"], width=2, smooth=True)
		for x, y in zip(points[0::2], points[1::2]):
			self.chart.create_oval(x - 2, y - 2, x + 2, y + 2, fill=COLORS["accent"], outline="")
		self.chart.create_text(12, 88, anchor="w", text=f"MEDIA DE LA GENERACIÓN  {values[-1]:.1f}", fill=COLORS["muted"], font=("Segoe UI", 8))

	def on_key_down(self, event):
		key = event.keysym.lower()
		self.keys.add(key)
		if key == "escape":
			self.toggle_pause()

	def on_key_up(self, event):
		self.keys.discard(event.keysym.lower())

	def toggle_pause(self):
		self.paused = not self.paused
		self.pause_button.configure(text="▶  REANUDAR" if self.paused else "Ⅱ  PAUSAR")
		self.last_time = None

	def toggle_manual(self):
		self.manual = not self.manual
		self.manual_button.configure(text="⌨  VOLVER A LA RED" if self.manual else "⌨  JUGAR TÚ · WASD")
		self.keys.clear()
		self.reset_matches()

	def next_match(self):
		self.candidate = (self.candidate + 1) % POPULATION_SIZE
		self.load_match(self.candidate)
		self.update_status()
		self.draw()

	def cycle_speed(self):
		self.speed_index = (self.speed_index + 1) % len(self.speed_options)
		self.speed_button.configure(text=f"VELOCIDAD  ·  ×{self.speed_options[self.speed_index]}")

	def restart(self):
		self.population = [new_genome() for _ in range(POPULATION_SIZE)]
		self.generation = 1
		self.candidate = 0
		self.fitnesses = []
		self.best_fitness = float("-inf")
		self.history = []
		self.games_played = 0
		self.paused = False
		self.manual = False
		self.pause_button.configure(text="Ⅱ  PAUSAR")
		self.manual_button.configure(text="⌨  JUGAR TÚ · WASD")
		self.candidate = 0
		self.reset_matches()
		self.update_chart()


if __name__ == "__main__":
	random.seed()
	app_root = tk.Tk()
	game = EvolutionFootball(app_root)
	app_root.mainloop()
