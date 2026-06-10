"""
Convolution basics algorithm.
Pure NumPy 2D convolution with stride/padding/dilation,
step-by-step trace for visualization, kernel training via gradient descent.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _apply_padding(input_2d, padding):
    """Zero-pad a 2D array."""
    if padding == 0:
        return np.array(input_2d, dtype=np.float64)
    return np.pad(np.array(input_2d, dtype=np.float64), padding, mode='constant', constant_values=0)


def _effective_kernel(kernel, dilation):
    """Expand kernel by inserting zeros for dilated convolution."""
    k_np = np.array(kernel, dtype=np.float64)
    k_size = k_np.shape[0]
    if dilation == 1:
        return k_np
    eff_size = (k_size - 1) * dilation + 1
    eff = np.zeros((eff_size, eff_size), dtype=np.float64)
    for i in range(k_size):
        for j in range(k_size):
            eff[i * dilation, j * dilation] = k_np[i, j]
    return eff


def compute_output_shape(h_in, w_in, k_size, stride, padding, dilation=1):
    """Compute convolution output dimensions.
    Formula: H_out = floor((H_in + 2P - D*(K-1) - 1) / S + 1)
    """
    h_out = (h_in + 2 * padding - dilation * (k_size - 1) - 1) // stride + 1
    w_out = (w_in + 2 * padding - dilation * (k_size - 1) - 1) // stride + 1
    return h_out, w_out


def conv2d_with_trace(input_2d, kernel, stride=1, padding=0, dilation=1):
    """
    Single-channel 2D convolution with element-by-element trace.

    Returns:
        dict with 'output' (2D list), 'output_shape' [h,w], 'trace' (list of per-position details)
    """
    inp = _apply_padding(input_2d, padding)
    k_np = np.array(kernel, dtype=np.float64)
    k_size = k_np.shape[0]
    h_out, w_out = compute_output_shape(len(input_2d), len(input_2d[0]), k_size, stride, padding, dilation)

    if dilation > 1:
        eff = _effective_kernel(kernel, dilation)
    else:
        eff = k_np

    windows = sliding_window_view(inp, (k_size, k_size))
    trace = []
    output = np.zeros((h_out, w_out), dtype=np.float64)

    for i in range(0, windows.shape[0] - (k_size - 1) * (dilation - 1), stride):
        for j in range(0, windows.shape[1] - (k_size - 1) * (dilation - 1), stride):
            oi, oj = i // stride, j // stride
            if oi >= h_out or oj >= w_out:
                continue

            if dilation > 1:
                window_vals = []
                elem_products = []
                for di in range(k_size):
                    for dj in range(k_size):
                        val = inp[i + di * dilation, j + dj * dilation]
                        window_vals.append(int(val))
                        elem_products.append(round(val * k_np[di, dj], 2))
                sum_val = sum(elem_products)
            else:
                w = windows[i, j]
                window_vals = w.astype(np.int32).tolist()
                ep = (w.astype(np.float64) * k_np).round(2).tolist()
                if isinstance(ep[0], list):
                    elem_products = [v for row in ep for v in row]
                else:
                    elem_products = ep
                sum_val = round(float(np.sum(w.astype(np.float64) * k_np)), 2)

            output[oi, oj] = sum_val
            trace.append({
                'pos': [oi, oj],
                'window_start': [i, j],
                'window': window_vals,
                'elem_product': elem_products,
                'sum': sum_val,
            })

    return {
        'output': output.round(2).tolist(),
        'output_shape': [h_out, w_out],
        'trace': trace,
    }


def conv2d_multi_channel(input_3d, kernel_3d, stride=1, padding=0):
    """Multi-channel convolution: each input channel has its own kernel, results are summed."""
    c_in = len(input_3d)
    per_channel = []
    summed = None
    for c in range(c_in):
        r = conv2d_with_trace(input_3d[c], kernel_3d[c], stride=stride, padding=padding)
        out = np.array(r['output'])
        per_channel.append(out.tolist())
        summed = out if summed is None else summed + out
    return {
        'per_channel': per_channel,
        'summed': summed.round(2).tolist() if summed is not None else [],
    }


def generate_kernel(size, preset='random', seed=None):
    """Generate a convolution kernel with a predefined pattern."""
    rng = np.random.default_rng(seed)
    if preset == 'edge_detect':
        k = np.full((size, size), -1.0, dtype=np.float64)
        center = size // 2
        k[center, center] = size * size - 1
        return k.round(2).tolist()
    elif preset == 'blur':
        return np.full((size, size), 1.0 / (size * size)).round(3).tolist()
    elif preset == 'sharpen':
        k = np.full((size, size), -1.0, dtype=np.float64)
        center = size // 2
        k[center, center] = size * size
        return k.round(2).tolist()
    else:
        return rng.integers(-3, 4, size=(size, size)).astype(np.float64).tolist()


def generate_matrix(h, w, seed=None):
    """Generate a random integer matrix as convolution input."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=(h, w)).astype(np.int32).tolist()


def train_kernel(target_preset, kernel_size, input_size, lr=0.05, iterations=100):
    """
    Train a convolution kernel via gradient descent to match a target kernel.
    Demonstrates how CNNs learn filters from data.
    """
    rng = np.random.default_rng()
    target = np.array(generate_kernel(kernel_size, preset=target_preset), dtype=np.float64)
    current = rng.normal(0, 0.3, size=(kernel_size, kernel_size)).astype(np.float64)
    inp = rng.integers(0, 10, size=(input_size, input_size)).astype(np.float64)
    inp_norm = inp / 9.0

    windows = sliding_window_view(inp_norm, (kernel_size, kernel_size))
    target_out = np.zeros((input_size - kernel_size + 1, input_size - kernel_size + 1))
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            target_out[i, j] = np.sum(windows[i, j] * target)

    trace = []
    n_positions = windows.shape[0] * windows.shape[1]
    for step in range(iterations + 1):
        current_out = np.zeros_like(target_out)
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                current_out[i, j] = np.sum(windows[i, j] * current)
        loss = float(np.mean((current_out - target_out) ** 2))

        grad = np.zeros_like(current)
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                error = current_out[i, j] - target_out[i, j]
                grad += error * windows[i, j]
        grad /= n_positions

        grad_norm = np.sqrt(np.sum(grad ** 2))
        if grad_norm > 1.0:
            grad /= grad_norm

        current -= lr * grad

        if step % max(1, iterations // 50) == 0 or step == iterations:
            trace.append({
                'step': step,
                'kernel': current.round(3).tolist(),
                'loss': round(loss, 6),
            })

    return {
        'iterations': trace,
        'final_kernel': current.round(3).tolist(),
        'target_kernel': target.tolist(),
        'input_matrix': inp.astype(np.int32).tolist(),
    }
