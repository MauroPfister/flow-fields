from time import perf_counter
import numpy as np
import numba
from matplotlib import pyplot as plt
from perlin2D import generate_fractal_noise_2d, generate_perlin_noise_2d

@numba.jit(fastmath=True, cache=True)
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

@numba.jit(nopython=True, fastmath=True, cache=True)
def get_v_field(v_field, x, y):
    """Get vector field value at query point (x, y).
    Linearly interpolates between grid cells.
    """
    h, w = v_field.shape[1:3]

    i_x = x * (w - 1)
    i_y = y * (h - 1)

    cell_x = int(i_x)
    cell_y = int(i_y)
    frac_x = i_x % 1
    frac_y = i_y % 1

    return lerp(
           lerp(v_field[:, cell_y, cell_x    ], v_field[:, cell_y    , cell_x + 1], frac_x),
           lerp(v_field[:, cell_y, cell_x + 1], v_field[:, cell_y + 1, cell_x + 1], frac_x),
           frac_y)

def is_collision(point, lines, widths, d_sep):
    if len(lines) < 1:
        return False

    # Array with line width of every point on every line 
    widths_vec = np.hstack([[width]*line.shape[0] for line, width in zip(lines, widths)])

    all_points = np.vstack(lines)
    dist = np.sum((all_points - point)**2, axis=1)
    mask = dist < ((widths_vec + d_sep)**2) * 1.5

    return np.any(mask)

def out_of_bounds(point, x_lim, y_lim):
    return point[0] < x_lim[0] or point[0] > x_lim[1] or point[1] < y_lim[0] or point[1] > y_lim[1]

def trace_line(seed_point, lines, widths, v_field, step_size, d_sep):
    x_lim = [0, 1]
    y_lim = [0, 1]
    line = [seed_point]

    max_length = 0.1

    max_iter = int(max_length / step_size)
    for i in range(max_iter):
        point = line[-1]
        vel = get_v_field(v_field, *point)
        point_new = point + vel * step_size
        if is_collision(point_new, lines, widths, d_sep) or out_of_bounds(point_new, x_lim, y_lim):
            break
        line.append(point_new)

    line.reverse()
    for i in range(max_iter):
        point = line[-1]
        vel = get_v_field(v_field, *point)
        point_new = point - vel * step_size
        if is_collision(point_new, lines, widths, d_sep) or out_of_bounds(point_new, x_lim, y_lim):
            break
        line.append(point_new)
    
    return np.array(line)


def trace_field(v_field, step_size=0.01, d_sep=0.05):

    lines = []
    widths = []
    queue = []
    start_point = np.random.rand(2) # Random starting point
    
    # Generate first line
    line = trace_line(start_point, lines, widths, v_field, step_size, d_sep)

    lines.append(line)
    widths.append(d_sep)
    queue.append(line)

    finished = False
    counter = 0

    current_line = queue.pop(0)

    # while not finished and counter < 10000:

    #     d_sep = np.asscalar(np.random.rand(1) * 0.06 + 0.01)

    #     # Generate normal vector from line segments
    #     normal = np.fliplr(np.diff(current_line, axis=0)) * np.array([[-1, 1]])
    #     # Offset vectors = normalized normal * d_sep
    #     offset = normal / np.sqrt(np.sum(normal**2, axis=1))[:, None] * d_sep * 1.2
    #     seed_found = False
    #     for i in range(current_line.shape[0] - 1):
    #         seed_point = current_line[i] + offset[i]

    #         if not is_collision(seed_point, lines, widths, d_sep):
    #             seed_found = True
    #             break

    #         seed_point = current_line[i] - offset[i]

    #         if not is_collision(seed_point, lines, widths, d_sep):
    #             seed_found = True
    #             break

    #     if seed_found:
    #         # Compute stream line
    #         line = trace_line(seed_point, lines, widths, v_field, step_size, d_sep)

    #         lines.append(line)
    #         widths.append(d_sep)
    #         queue.append(line)
    #     else:
    #         if len(queue) < 1:
    #             finished = True
    #         else:
    #             current_line = queue.pop(0)

    #     counter += 1
    lines = []
    widths = []

    finished = False
    max_itr = 300
    while not finished and counter < max_itr:

        scale = (0.05 - 0.005) * ((max_itr - counter) / max_itr)**5  + 0.005
        scale = max(1 / (counter/20 + 2) * 0.08, step_size)

        attempts = 0
        while attempts < 1000:
            d_sep = np.asscalar(np.random.rand(1) * 0.04 + 0.005) + scale
            d_sep = scale * np.asscalar(np.random.rand(1)* 0.5 + 0.5)
            # d_sep = np.asscalar(np.clip(np.random.exponential(0.2, 1), 0, 1)) * 0.07 + 0.003
            seed_point = np.random.rand(2)
            if is_collision(seed_point, lines, widths, d_sep):
                attempts += 1
            else:
                break

        if attempts < 1000:
            line = trace_line(seed_point, lines, widths, v_field, step_size, d_sep)

            lines.append(line)
            widths.append(d_sep)
            counter += 1
        else:
            finished = True

        # counter += 1
            

    return lines, widths



if __name__ == "__main__":
    step_size = 0.005
    d_sep = 0.01

    angle_field = np.pi * generate_fractal_noise_2d((1000, 1000), (1, 1), octaves=2, persistence=2)
    # angle_field = np.pi * np.round(generate_perlin_noise_2d((1000, 1000), (2, 2)) * 4) / 4
    # angle_field = np.round(angle_field * 1) / 2  # Discrete angles
    v_field = np.array([np.cos(angle_field), np.sin(angle_field)])

    # v_field = curl(perlin, x, y)
    # v_field = v_field / (np.sqrt(np.sum(v_field**2, axis=0)) + eps) # Normalize field

    lines, widths = trace_field(v_field, step_size=step_size, d_sep=d_sep)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("equal")
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks([])
    ax.set_yticks([])


    plot_style = "lines"
    outline = False

    cmap=plt.get_cmap("tab10")
    colors = cmap(np.random.rand(len(lines)))

    warm = ["#a47053", "#efca66", "#ecdab9", "#cec3c8", "#909cac"]
    sun = ["#7c7b89", "#f1e4de", "#f4d75e", "#e9723d", "#0b7fab"]
    sea = ["#002a29", "#007a79", "#c7cedf", "#976f4f", "#4d2a16"]
    blue = ["#012e67", "#9cacbf", "#2b6684", "#032e42", "#0a1417"]
    blue2 = ["#274b69", "#85a1c1", "#c6ccd8", "#3f4d63", "#202022"]
    pastel = ["#5b828e", "#bbcfd7", "#d2c8bc", "#ba9a88", "#ac7e62"]
    green = ["#487549", "#abba82", "#a7b5b7", "#037c87", "#102020"]
    pomegranade = ["#ed8b77", "#dc5322", "#a5142a"] # "#310003", "#0d1316"]
    gray = ["#181614", "#2b292b", "#92817b", "#aca6a6", "#e5ecf3"]
    orange = ["#c43d16", "#edc596", "#fcb500", "#dc6d02"]
    red = ["#ac0e28", "#bc4558", "#490009"]
    # accent = ["#363634", "#524636", "#ac7330", "#b19a78", "#d1c5ab"]

    warm = ["#ae1903", "#c43d16", "#c43d16", "#edc596", "#fcb500", "#dc6d02", "#770f10", "#621122", "#d40000", "#fab73d", "#e6b95f"]
    red_tones = ["#e28413", "#ec7415","#f16c16","#f56416","#ef5e17","#e95818","#dd4b1a","#e6391b","#ef271b"] 
    autumn_tones = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]
    violet = ["#97dffc","#93caf6","#8eb5f0","#858ae3","#7364d2","#613dc1","#5829a7","#4e148c","#461177","#3d0e61"]
    violet_red = ["#ea698b","#d55d92","#c05299","#ac46a1","#973aa8","#822faf","#6d23b6","#6411ad","#571089","#47126b"]

    red_blue = ["#b7094c","#a01a58","#892b64","#723c70","#5c4d7d","#455e89","#2e6f95","#1780a1","#0091ad"]
    blue_orange = ["#8ecae6","#73bfdc","#58b4d1","#219ebc","#126782","#023047","#ffb703","#fd9e02","#fb8500","#fb9017"]

    accent = autumn_tones
    selection = autumn_tones
    colors = np.random.choice(selection, len(lines))


    for line, d_sep, color in zip(lines, widths, colors):
        if plot_style == "lines":
            # ax.plot(line[:, 0], line[:, 1], linewidth=d_sep*200)
            if outline:
                ax.plot(line[:, 0], line[:, 1], 
                        color="black", 
                        linewidth=d_sep*900, 
                        solid_capstyle="round")

                # ax.plot(line[:, 0], line[:, 1], 
                #         color=color, 
                #         linewidth=d_sep*800, 
                #         solid_capstyle="round",
                #         path_effects=[pe.Stroke(linewidth=10, foreground='black'), pe.Normal()])

            step = int(2 * d_sep / step_size) + 1
            step = 1
            line_len = line.shape[0]

            main_line_end = int(line_len * (np.random.rand(1)*0.3 + 0.65))

            ids = range(main_line_end, line.shape[0], step)

            for id in reversed(ids):
                length = int(np.random.uniform(1, 10, 1))
                ax.plot(line[id:id+length, 0], line[id:id+length, 1], 
                        color=np.random.choice(accent, 1)[0],
                        linewidth=d_sep*800, 
                        solid_capstyle="round")

            ax.plot(line[:main_line_end, 0], line[:main_line_end, 1], 
                    color=color, 
                    linewidth=d_sep*800, 
                    solid_capstyle="round")

        if plot_style == "dots":
            step = int(2 * d_sep / step_size) + 1
            for point in line[::step]:
                circle = plt.Circle(tuple(point), d_sep * 0.9, color='black')
                ax.add_patch(circle)


    plt.show()
    ax.axis('off')
    fig.savefig("flow-field_02.jpg", bbox_inches='tight', pad_inches = 0)
    # fig.savefig("flow-field_02.svg", bbox_inches='tight', pad_inches = 0)