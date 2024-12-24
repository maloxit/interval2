import json
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from intvalpy import IntLinIncR2, Interval, Tol, precision
from intvalpy_fix import IntLinIncR2
import numpy as np
from os import makedirs

precision.extendedPrecisionQ = True

def load_data(directory, side):
    values_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    loaded_data = []
    for offset, value_x in enumerate(values_x):
        data = {}
        with open(directory + "/" + str(value_x) + "lvl_side_" + side + "_fast_data.json", "rt") as f:
            data = json.load(f)
        for i in range(8):
            loaded_data.append([])
            for j in range(1024):
                loaded_data[i].append([(0, 0) for k in range(100 * len(values_x))])
                for k in range(len(data["sensors"][i][j])):
                    loaded_data[i][j][offset * 100 + k] = (value_x, data["sensors"][i][j][k])

    return loaded_data, values_x

def generate_data(rad, offset, offset_id, values_x):
    gen = []
    gen_offset = []
    for id, value_x in enumerate(values_x):
            data = np.random.uniform(-rad, rad, 100) + value_x
            for k in range(len(data)):
                gen.append((value_x, data[k]))
                gen_offset.append((value_x, data[k] + (offset if id == offset_id else 0)))

    return gen, gen_offset

# using Tol
def regression_type_1(points):
    x, y = zip(*points)
    # build intervals out of given points
    weights = [1 / 16384] * len(y)
    # we know that y_i = b_0 + b_1 * x_i
    # or, in other words
    # Y = X * b, where X is a matrix with row (x_i, 1), and b is a vector (b_1, b_0)
    X_mat = Interval([[[x_el, x_el], [1, 1]] for x_el in x])
    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    # find argmax for Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)
    updated = 0
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([
                [[x[i], x[i]], [1, 1]]
            ])
            Y_vec_small = Interval([[y[i], weights[i]]], midRadQ=True)
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                weights[i] = abs(y[i] - (x[i] * b_vec[0] + b_vec[1])) + 1e-8
                updated += 1

    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    # find argmax for Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)

    return b_vec, weights, updated



def points_to_twin(points, values_x):
    x, y = zip(*points)
    eps = 1 / 16384

    # first of all, lets build y_ex and y_in
    y_ex = []
    y_in = []

    for i in range(len(values_x)):
        y_list = list(y[i * 100 : (i + 1) * 100])
        y_list.sort()
        y_in.append((y_list[25] - eps, y_list[75] + eps))
        y_ex.append((max(y_list[25] - 1.5 * (y_list[75] - y_list[25]), y_list[0]), min(y_list[75] + 1.5 * (y_list[75] - y_list[25]), y_list[-1])))
    
    return y_in, y_ex


# using twin arithmetics
def regression_type_2(values_x, y_in, y_ex, plot = True):


    X_mat = []
    Y_vec = []
    for i in range(len(values_x)):
        x_el = values_x[i]
        # y_ex_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex[i][0], y_ex[i][1]])
        # y_in_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex[i][0], y_in[i][1]])
        # y_ex_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in[i][0], y_ex[i][1]])
        # y_in_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in[i][0], y_in[i][1]])

    # now we have matrix X * b = Y, but with some "additional" rows
    # we can walk over all rows and if some of them is less than 0, we can just remove it at all
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)
    to_remove = []
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([X_mat[i]])
            Y_vec_small = Interval([Y_vec[i]])
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            del X_mat[i]
            del Y_vec[i]

    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)

    vertices1 = IntLinIncR2(X_mat_interval, Y_vec_interval)
    vertices2 = IntLinIncR2(X_mat_interval, Y_vec_interval, consistency='tol')

    if plot:
        plt.xlabel("b0")
        plt.ylabel("b1")
    b_uni_vertices = []
    for v in vertices1:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            b_uni_vertices += [(x[i], y[i]) for i in range(len(x))]
            if plot:
                plt.fill(x, y, linestyle='-', linewidth=1, color='gray', alpha=0.5, label="Uni")
                plt.scatter(x, y, s=0, color='black', alpha=1)


    b_tol_vertices = []
    for v in vertices2:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            b_tol_vertices += [(x[i], y[i]) for i in range(len(x))]
            if plot:
                plt.fill(x, y, linestyle='-', linewidth=1, color='blue', alpha=0.3, label="Tol")
                plt.scatter(x, y, s=10, color='black', alpha=1)

    if plot:
        plt.scatter([b_vec[0]], [b_vec[1]], s=10, color='red', alpha=1, label="argmax Tol")
        plt.legend()
    return b_vec, to_remove, b_uni_vertices, b_tol_vertices

def build_plots(data, coord_x, coord_y, values_x):
    makedirs(f'results/{coord_x}_{coord_y}', exist_ok=True)
    # method 1
    b_vec, rads, to_remove = regression_type_1(data)
    x, y = zip(*data)
    plt.figure()
    plt.title("Y(x) method 1 for " + str((coord_x, coord_y)))
    plt.scatter(x, y, label="medians", color='green')
    plt.plot([-0.5, 0.5], [b_vec[1] + b_vec[0] * -0.5, b_vec[1] + b_vec[0] * 0.5], label="Argmax Tol")
    plt.legend()
    print((coord_x, coord_y), 1, b_vec[0], b_vec[1], to_remove)
    plt.savefig(f'results/{coord_x}_{coord_y}/calibration.png')

    plt.figure()
    plt.title("Y(x) - b_0*x - b_1 method 1 for " + str((coord_x, coord_y)))
    for i in range(len(y)):
        plt.plot([i, i], [y[i] - rads[i] - b_vec[1] - b_vec[0] * x[i],
                          y[i] + rads[i] - b_vec[1] - b_vec[0] * x[i]], color="lightblue", zorder=1)
        plt.plot([i, i], [y[i] - 1 / 16384 - b_vec[1] - b_vec[0] * x[i],
                          y[i] + 1 / 16384 - b_vec[1] - b_vec[0] * x[i]], color="green", zorder=2)
    plt.savefig(f'results/{coord_x}_{coord_y}/calibration_dif.png')

    # method 2
    plt.figure()
    plt.title("Uni and Tol method 2 for " + str((coord_x, coord_y)))
    y_in, y_ex = points_to_twin(data, values_x)
    b_vec2, to_remove, b_uni_vertices, b_tol_vertices = regression_type_2(values_x, y_in, y_ex)
    plt.savefig(f'results/{coord_x}_{coord_y}/uni_tol.png')

    print((coord_x, coord_y), 2, b_vec2[0], b_vec2[1], len(to_remove))
    x2 = values_x
    plt.figure()
    plt.title("Y(x) method 2 for " + str((coord_x, coord_y)))

    x2 = [-3] + x2 + [3]

    for i in range(len(x2) - 1):
        x0 = x2[i]
        x1 = x2[i + 1]
        max_idx = 0
        min_idx = 0
        max_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        min_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_uni_vertices)):
            val = b_uni_vertices[j][1] + b_uni_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x0
        y1_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x1
        y0_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x0
        y1_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="lightgray", alpha=0.3, linewidth=0)

        max_idx = 0
        min_idx = 0
        max_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        min_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_tol_vertices)):
            val = b_tol_vertices[j][1] + b_tol_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x0
        y1_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x1
        y0_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x0
        y1_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="lightblue", alpha=0.3, linewidth=0)

    for i in range(len(values_x)):
        plt.plot([values_x[i], values_x[i]], [y_ex[i][0], y_ex[i][1]], color="gray", zorder=1)
        plt.plot([values_x[i], values_x[i]], [y_in[i][0], y_in[i][1]], color="blue", zorder=2)

    plt.plot([-0.5, 0.5], [b_vec2[1] + b_vec2[0] * -0.5, b_vec2[1] + b_vec2[0] * 0.5], label="Argmax Tol", color="red",
             zorder=1000)
    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))
    plt.savefig(f'results/{coord_x}_{coord_y}/method2.png')


def build_example_plots(y_in, y_ex, values_x, offset):
    makedirs(f'results/example_{offset}', exist_ok=True)
    plt.figure()
    plt.title("Uni and Tol method 2 for example")
    b_vec2, to_remove, b_uni_vertices, b_tol_vertices = regression_type_2(values_x, y_in, y_ex)
    print(offset, b_vec2[0], b_vec2[1], len(to_remove))
    x2 = values_x
    plt.savefig(f'results/example_{offset}/uni_tol.png')
    plt.figure()
    plt.title("Y(x) method 2 for example")

    x2 = [-3] + x2 + [3]

    for i in range(len(x2) - 1):
        x0 = x2[i]
        x1 = x2[i + 1]
        max_idx = 0
        min_idx = 0
        max_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        min_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_uni_vertices)):
            val = b_uni_vertices[j][1] + b_uni_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x0
        y1_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x1
        y0_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x0
        y1_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="gray", alpha=0.1, linewidth=0)

        max_idx = 0
        min_idx = 0
        max_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        min_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_tol_vertices)):
            val = b_tol_vertices[j][1] + b_tol_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x0
        y1_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x1
        y0_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x0
        y1_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="blue", alpha=0.1, linewidth=0)

    for i in range(len(values_x)):
        plt.plot([values_x[i], values_x[i]], [y_ex[i][0], y_ex[i][1]], color="gray", zorder=1)
        plt.plot([values_x[i], values_x[i]], [y_in[i][0], y_in[i][1]], color="blue", zorder=2)

    plt.plot([-0.5, 0.5], [b_vec2[1] + b_vec2[0] * -0.5, b_vec2[1] + b_vec2[0] * 0.5], label="Argmax Tol", color="red",
             zorder=1000)
    for j in range(len(b_tol_vertices)):
        plt.plot([-0.5, 0.5], [b_tol_vertices[j][1] + b_tol_vertices[j][0] * -0.5, b_tol_vertices[j][1] + b_tol_vertices[j][0] * 0.5], color="green",
                zorder=1000)
        val = b_tol_vertices[j][1] + b_tol_vertices[j][0] * (x0 + x1) / 2
    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))
    plt.savefig(f'results/example_{offset}/method2.png')


def example(b0, b1, rad_in, rad_ex, offset):
    y_in = [
        (values_x[0] * b0 + b1 - rad_in, values_x[0] * b0 + b1 + rad_in), 
        (values_x[1] * b0 + b1 - rad_in + offset, values_x[1] * b0 + b1 + rad_in + offset), 
        (values_x[2] * b0 + b1 - rad_in, values_x[2] * b0 + b1 + rad_in)
        ]

    y_ex = [
        (values_x[0] * b0 + b1 - rad_ex, values_x[0] * b0 + b1 + rad_ex), 
        (values_x[1] * b0 + b1 - rad_ex + offset, values_x[1] * b0 + b1 + rad_ex + offset), 
        (values_x[2] * b0 + b1 - rad_ex, values_x[2] * b0 + b1 + rad_ex)
        ]

    build_example_plots(y_in, y_ex, values_x, offset)


def interp(data, coord_x, coord_y, values_x):
    res = []
    # method 1
    b_vec, rads, to_remove = regression_type_1(data)
    res.append((float(b_vec[0]), float(b_vec[1]), to_remove))

    # method 2
    y_in, y_ex = points_to_twin(data, values_x)
    b_vec2, to_remove, b_uni_vertices, b_tol_vertices = regression_type_2(values_x, y_in, y_ex, False)
    res.append((float(b_vec2[0]), float(b_vec2[1]), len(to_remove)))
    return res



if __name__ == "__main__":
    makedirs(f'results', exist_ok=True)
    side_a_1, values_x = load_data("bin/04_10_2024_070_068", "a")

    '''
    res = dict()
    for i in range(0, 8, 2):
        res[i] = dict()
        for j in range(0, 1024, 8):
            res[i][j] = interp(side_a_1[i][j], i, j, values_x)
            print(i, j)
    with open('results/result.json', 'w') as fp:
        json.dump(res, fp)
    '''
    
    build_plots(side_a_1[0][0], 0, 0, values_x)
    build_plots(side_a_1[1][24], 1, 24, values_x)
    build_plots(side_a_1[2][72], 2, 72, values_x)

    
    values_x = [-0.4, 0, 0.4]

    #gen, gen_offset = generate_data(0.15, 0.2, 1, values_x)
    #build_plots(gen_offset, -1, -1, values_x)

    b0 = 1
    b1 = 0.05
    rad_in = 0.1
    rad_ex = 0.2

    offset = 0.0
    example(b0, b1, rad_in, rad_ex, offset)

    offset = 0.15
    example(b0, b1, rad_in, rad_ex, offset)

    offset = 0.25
    example(b0, b1, rad_in, rad_ex, offset)
