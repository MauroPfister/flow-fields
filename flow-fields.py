from time import perf_counter
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
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

def get_vector_field(field, point):
    """Get vector field value at query point (x, y).
    Linearly interpolates between grid cells.
    """
    h, w = field.shape[0:2]

    i_x = point[0] * (w - 1)
    i_y = point[1] * (h - 1)

    cell_x = int(i_x)
    cell_y = int(i_y)
    frac_x = i_x % 1
    frac_y = i_y % 1
    
    return lerp(
           lerp(field[cell_y, cell_x    ], field[cell_y    , cell_x + 1], frac_x),
           lerp(field[cell_y, cell_x + 1], field[cell_y + 1, cell_x + 1], frac_x),
           frac_y)

def is_collision(point, width, lines, widths, safety_fac=1.5):
    if len(lines) < 1:
        return False

    # Array with line width of every point on every line 
    widths_all_points = np.repeat(widths, [line.shape[0] for line in lines])
    all_points = np.vstack(lines)
    # Squared distance between query point and points of all the other lines
    dist = np.sum((all_points - point)**2, axis=1)
    is_collision = dist < (((widths_all_points + width) / 2)**2) * safety_fac

    return np.any(is_collision)

def out_of_bounds(point, x_lim, y_lim):
    return point[0] < x_lim[0] or point[0] > x_lim[1] or point[1] < y_lim[0] or point[1] > y_lim[1]

def trace_line(start_point, width, lines, widths, field, step_size):
    x_lim = [0, 1]
    y_lim = [0, 1]
    line = [start_point]

    max_length = 0.1

    max_iter = int(max_length / step_size)
    for i in range(max_iter):
        point = line[-1]
        vel = get_vector_field(field, point)
        point_new = point + vel * step_size
        if is_collision(point_new, width, lines, widths) or out_of_bounds(point_new, x_lim, y_lim):
            break
        line.append(point_new)

    line.reverse()
    for i in range(max_iter):
        point = line[-1]
        vel = get_vector_field(field, point)
        point_new = point - vel * step_size
        if is_collision(point_new, width, lines, widths) or out_of_bounds(point_new, x_lim, y_lim):
            break
        line.append(point_new)
    
    return np.array(line)

def trace_field(field, step_size):
    lines = []
    widths = []

    width_scale = 0.16
    max_attempts = 1000  # Max attempts at finding a valid starting point
    n_lines_max = 300
    for _ in range(n_lines_max):

        # Start with thick lines and gradually decrease width
        scale = 1 / (len(lines) / 20 + 2) * width_scale * (np.random.rand(1) * 0.5 + 0.5)
        # scale = (0.05 - 0.005) * ((n_lines_max - len(lines)) / n_lines_max)**5  + 0.005
        width = max(float(scale), step_size / 5)

        # Try to generate valid starting point for new line
        start_point_valid = False
        for _ in range(max_attempts):
            start_point = np.random.rand(2)
            if not is_collision(start_point, width, lines, widths):
                start_point_valid = True
                break

        if start_point_valid:
            line = trace_line(start_point, width, lines, widths, field, step_size)
            lines.append(line)
            widths.append(width)
        else:
            break

    return lines, widths


color_palettes = {
    "autumn": [ "#03071e", "#370617", "#6a040f", "#9d0208", "#d00000", "#dc2f02",
                "#e85d04", "#f48c06", "#faa307", "#ffba08"],
    "violet-red": ["#ea698b","#d55d92","#c05299","#ac46a1","#973aa8","#822faf",
                   "#6d23b6","#6411ad","#571089","#47126b"],
    "blue-orange": ["#8ecae6", "#73bfdc", "#58b4d1", "#219ebc", "#126782", "#023047", 
                    "#ffb703", "#fd9e02", "#fb8500", "#fb9017"],
    "blue-berry": ["#b7094c", "#a01a58", "#892b64", "#723c70", "#5c4d7d", "#455e89", 
                   "#2e6f95", "#1780a1", "#0091ad"],
    "black-white": ["#000000", "#ffffff"]
}


if __name__ == "__main__":
    step_size = 0.005
    width = 0.01

    angle_field = np.pi * generate_fractal_noise_2d((1000, 1000), (1, 1), octaves=2, persistence=2)
    # angle_field = np.pi * np.round(generate_perlin_noise_2d((1000, 1000), (2, 2)) * 4) / 4
    # angle_field = np.round(angle_field * 1) / 2  # Discrete angles
    field = np.stack([np.cos(angle_field), np.sin(angle_field)], axis=2)

    # v_field = curl(perlin, x, y)
    # v_field = v_field / (np.sqrt(np.sum(v_field**2, axis=0)) + eps) # Normalize field

    t_start = perf_counter()
    lines, widths = trace_field(field, step_size)
    t_end = perf_counter()
    print(f"Ellapsed time: {t_end - t_start: .2f} s")

    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_axis_off()

    plot_style = "lines"

    accent = color_palettes["autumn"]
    selection = color_palettes["autumn"]
    colors = np.random.choice(selection, len(lines))

    for line, width, color in zip(lines, widths, colors):
        if plot_style == "lines":
            step = 1
            line_len = line.shape[0]
            main_line_end = int(line_len * (np.random.rand(1)*0.3 + 0.65))
            ids = range(main_line_end, line.shape[0], step)

            for id in reversed(ids):
                length = int(np.random.uniform(1, 10, 1))
                line_plot = Line(line[id:id+length, 0], line[id:id+length, 1],
                                 color=np.random.choice(accent, 1)[0],
                                 linewidth=width, 
                                 solid_capstyle="round")
                ax.add_line(line_plot)

            line_plot = Line(line[:main_line_end, 0], line[:main_line_end, 1],
                             linewidth=width,
                             color=color,
                             solid_capstyle="round")
            ax.add_line(line_plot)

        if plot_style == "dots":
            step = int( width / step_size) + 1
            for point in line[::step]:
                circle = plt.Circle(tuple(point), width / 2, color=color)
                ax.add_patch(circle)

    plt.show()
    fig.savefig("flow-field_02.jpg", dpi=150)