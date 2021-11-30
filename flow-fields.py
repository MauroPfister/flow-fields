from time import perf_counter
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from functools import partial
from perlin2D import generate_fractal_noise_2d, generate_perlin_noise_2d

class Line(Line2D):
    """Custom line class.
    The line width is defined in the same units as the plotted data.
    E.g. a line of length 1.0 and width 1.0 looks like a square.
    In contrast, the width of a line plotted with plt.plot() is defined
    in points.
    """
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1) 
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def lerp(a, b, x):
    """Linear interpolation."""
    return a + x * (b - a)


def curl(field, x, y, eps=0.001):
    """Computes curl of a 2D vector field."""
    # grad = np.gradient(field(x, y))
    # curl = np.array([grad[0], -grad[1]]) 

    x_comp =   (field(x, y + eps) - field(x, y - eps)) / (2 * eps);
    y_comp = - (field(x + eps, y) - field(x - eps, y)) / (2 * eps);

    curl = np.array([x_comp, - y_comp]) 
    return curl


def get_velocity(point, field, bounds, interpolate):
    """Get velocity of field at query point (x, y).
    Linearly interpolates between grid cells if interpolate=True.
    Otherwise returns vector of closest grid cell.
    """
    x_min, y_min, x_max, y_max = bounds
    h, w = field.shape[0:2]

    i = (point[0] - x_min) / (x_max - x_min) * (w - 1)
    j = (point[1] - y_min) / (y_max - y_min) * (h - 1)

    cell_x = int(i)
    cell_y = int(j)
    frac_x = i % 1
    frac_y = j % 1

    if interpolate:
        vel = lerp(lerp(field[cell_y, cell_x    ], field[cell_y    , cell_x + 1], frac_x),
                   lerp(field[cell_y, cell_x + 1], field[cell_y + 1, cell_x + 1], frac_x),
                   frac_y)
    else:
        vel = field[cell_y, cell_x]
    
    return vel


def is_collision(point, width, lines, widths, safety_fac=1.5):
    """Check if point collides with points of another line."""
    if not lines:
        return False

    # Array with line width of every point on every line 
    widths_all_points = np.repeat(widths, [line.shape[0] for line in lines])
    all_points = np.vstack(lines)
    # Squared distance between query point and points of all the other lines
    dist = np.sum((all_points - point)**2, axis=1)
    is_collision = dist < (((widths_all_points + width) / 2)**2) * safety_fac

    return np.any(is_collision)


def out_of_bounds(point, width, bounds):
    """Check if point is out of bounds."""
    x_min, y_min, x_max, y_max = bounds
    r = width / 2
    out_of_bounds = (   point[0] - r < x_min 
                     or point[0] + r > x_max 
                     or point[1] - r < y_min 
                     or point[1] + r > y_max)
    return out_of_bounds


def generate_width_dist(name, max_width):
    """Generate a function f(x) that returns a line width.
    Typically x is proportional to the number of lines already generated.
    """
    # width = (0.05 - 0.005) * ((n_max_lines - len(lines)) / n_max_lines)**5  + 0.005

    width_dist = None
    if name == "uniform":
        width_dist = partial(uniform_width, width=max_width)
    elif name == "decreasing":
        width_dist = partial(decreasing_width, max_width=max_width)
         
    return width_dist


def uniform_width(x, width):
    return width


def decreasing_width(x, max_width):
    """Start with thick lines and gradually decrease width."""
    width = 2 / (x / 20 + 2) * max_width * (np.random.rand(1) * 0.5 + 0.5)
    return width


def trace_line(start_point, width, lines, widths, field, bounds, max_len, interpolate, step_size):
    """Trace single flow line of vector field."""
    line = [start_point]
    n_points = int(max_len / 2 / step_size)

    # Trace forward
    for _ in range(n_points):
        point = line[-1]
        vel = get_velocity(point, field, bounds, interpolate)
        point_new = point + vel * step_size
        if is_collision(point_new, width, lines, widths) or out_of_bounds(point_new, width, bounds):
            break
        line.append(point_new)

    # Trace backward
    line.reverse()
    for _ in range(n_points):
        point = line[-1]
        vel = get_velocity(point, field, bounds, interpolate)
        point_new = point - vel * step_size
        if is_collision(point_new, width, lines, widths) or out_of_bounds(point_new, width, bounds):
            break
        line.append(point_new)
    
    return np.array(line)


def trace_field(field, bounds, max_len, width_dist, n_max_lines, interpolate, step_size):
    """Trace flow lines of vector field."""
    lines = []
    widths = []

    max_attempts = 1000  # Max attempts at finding a valid starting point
    for i in range(n_max_lines):
        width = width_dist(i)
        width = max(float(width), step_size / 5)

        # Try to generate valid starting point for new line
        start_point_valid = False
        for _ in range(max_attempts):
            start_point = np.random.uniform(bounds[0:2], bounds[2:4])
            if not (   is_collision(start_point, width, lines, widths) 
                    or out_of_bounds(start_point, width, bounds)):
                start_point_valid = True
                break

        if start_point_valid:
            line = trace_line(start_point, width, lines, widths, field, bounds, max_len, interpolate, step_size)
            lines.append(line)
            widths.append(width)
        else:
            break

    return lines, widths


def fig_setup(col_bg="#ffffff"):
    """Set up figure for plotting."""
    fig = plt.figure(figsize=(8, 8))
    ax = plt.Axes(fig, [0, 0, 1, 1], facecolor=col_bg)
    fig.add_axes(ax)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    # Remove frame but keep background color
    for s in ax.spines.values():
        s.set_visible(False)

    return fig, ax


def plot(lines, widths, col_lines, col_bg, style="lines"):
    """Main plotting function."""
    fig, ax = fig_setup(col_bg)

    colors = np.random.choice(col_lines, len(lines))

    for line, width, color in zip(lines, widths, colors):
        if style == "lines":
            step = 1
            line_len = line.shape[0]
            main_line_end = int(line_len * (np.random.rand(1)*0.3 + 0.65))
            ids = range(main_line_end, line.shape[0], step)

            for id in reversed(ids):
                length = int(np.random.uniform(1, 10, 1))
                line_plot = Line(line[id:id+length, 0], line[id:id+length, 1],
                                 color=np.random.choice(col_lines, 1)[0],
                                 linewidth=width, 
                                 solid_capstyle="round")
                ax.add_line(line_plot)

            line_plot = Line(line[:main_line_end, 0], line[:main_line_end, 1],
                             linewidth=width,
                             color=color,
                             solid_capstyle="round")
            ax.add_line(line_plot)

        if style == "dots":
            step = int( width / step_size) + 1
            for point in line[::step]:
                circle = plt.Circle(tuple(point), width / 2, color=color)
                ax.add_patch(circle)
    
    return fig


color_palettes = {
    "autumn": [ "#03071e", "#370617", "#6a040f", "#9d0208", "#d00000", "#dc2f02",
                "#e85d04", "#f48c06", "#faa307", "#ffba08"],
    "violet-red": ["#ea698b","#d55d92","#c05299","#ac46a1","#973aa8","#822faf",
                   "#6d23b6","#6411ad","#571089","#47126b"],
    "blue-orange": ["#8ecae6", "#73bfdc", "#58b4d1", "#219ebc", "#126782", "#023047", 
                    "#ffb703", "#fd9e02", "#fb8500", "#fb9017"],
    "blue-berry": ["#b7094c", "#a01a58", "#892b64", "#723c70", "#5c4d7d", "#455e89", 
                   "#2e6f95", "#1780a1", "#0091ad"],
    "black-white": ["#000000", "#ffffff"],
    "dark": [ "#111111", "#222222"]
}


if __name__ == "__main__":

    # Hyper parameters
    step_size = 0.005
    max_len = 0.2
    bounds = [0, 0, 1, 1]
    max_width = 0.08
    n_max_lines = 300
    interpolate = True

    noise = generate_fractal_noise_2d((1000, 1000), (1, 1), octaves=2, persistence=2)
    angle_field = np.pi * ((noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1)
    angle_field = np.pi * noise
    # angle_field = np.pi * np.round(generate_perlin_noise_2d((1000, 1000), (2, 2)) * 4) / 4
    # angle_field = np.round(angle_field * np.pi / 4) / 4 * np.pi  # Discrete angles
    field = np.stack([np.cos(angle_field), np.sin(angle_field)], axis=2)

    width_dist = generate_width_dist("decreasing", max_width)
    t_start = perf_counter()
    lines, widths = trace_field(field, bounds, max_len, width_dist, n_max_lines, interpolate, step_size)
    t_end = perf_counter()
    print(f"Ellapsed time: {t_end - t_start: .2f} s")

    style = "lines"
    col_lines = color_palettes["autumn"]
    fig = plot(lines, widths, col_lines, "#ffffff", style="lines")

    plt.show()
    fig.savefig("flow-field_02.jpg", dpi=150)