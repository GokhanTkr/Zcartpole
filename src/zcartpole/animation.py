"""Render the cart and any number of serial pendulum links as an animated GIF."""
from pathlib import Path

import numpy as np


def save_animation(model, sol, output='swingup.gif', *, fps=15, speed=1.0,
                   x_limit=None):
    """Save a closed-loop simulation as GIF and return its path.

    ``speed`` is simulated seconds per playback second. Matplotlib and Pillow
    are imported only when this function is called (install ``.[viz]``).
    """
    if not isinstance(fps, int) or fps < 1 or not np.isfinite(speed) or speed <= 0:
        raise ValueError('fps and speed must be positive')
    if x_limit is not None and (not np.isfinite(x_limit) or x_limit <= 0):
        raise ValueError('x_limit must be positive')
    t = np.asarray(sol.t, dtype=float)
    Y = np.asarray(sol.y, dtype=float)
    U = np.asarray(sol.u, dtype=float)
    if (t.ndim != 1 or t.size < 2 or Y.shape != (2+3*model.n, t.size)
            or U.shape != (t.size,) or not np.isfinite(t).all()
            or not np.isfinite(Y).all() or not np.isfinite(U).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Invalid simulation trajectory')

    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    output = Path(output)
    if output.suffix.lower() != '.gif':
        raise ValueError('output must be a .gif path')
    total = float(np.sum(model.l))
    half_width = max(0.18, 0.15*total)
    reach = float(np.max(np.abs(Y[0]))) + total + half_width + 0.15
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    ax.set(xlim=(-reach, reach), ylim=(-total-0.25, total+0.25),
           xlabel='x (m)', ylabel='y (m)', title=f'{model.n}-link cart-pole swing-up')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.axhline(-0.19, color='#888888', linewidth=2, zorder=0)
    if x_limit is not None:
        for boundary in (-x_limit, x_limit):
            ax.axvline(boundary, color='#d1495b', linestyle='--',
                       alpha=0.7, linewidth=1.2)
    cart = Rectangle((-half_width, -0.17), 2*half_width, 0.17,
                     color='#2574a9', zorder=3)
    ax.add_patch(cart)
    links, = ax.plot([], [], '-o', linewidth=3, markersize=8,
                     color='#e76f51', markerfacecolor='#264653', zorder=4)
    label = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top',
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.85,
                                           edgecolor='none'))

    frame_t = np.arange(float(t[0]), float(t[-1]), speed/fps)
    frame_t = np.r_[frame_t, t[-1]]
    indices = np.clip(np.searchsorted(t, frame_t), 0, t.size-1)

    def draw(frame):
        j = indices[frame]
        x = float(Y[0, j])
        angles = np.arctan2(Y[3::3, j], Y[2::3, j])
        xx = np.r_[x, x + np.cumsum(model.l*np.sin(angles))]
        yy = np.r_[0.0, np.cumsum(model.l*np.cos(angles))]
        cart.set_x(x-half_width)
        links.set_data(xx, yy)
        label.set_text(f't = {t[j]:.2f} s   x = {x:+.3f} m   u = {U[j]:+.2f} N')
        return cart, links, label

    animation = FuncAnimation(fig, draw, frames=len(indices),
                              interval=1000/fps, blit=False,
                              cache_frame_data=False)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(output), writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)
    return output
