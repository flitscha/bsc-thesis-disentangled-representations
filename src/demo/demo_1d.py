import tkinter as tk
from tkinter import ttk
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.experiment import Experiment
from core.pipeline import build_graph, build_spline_from_graph
from core.traversal import traverse_graph

from visualization.visualize import (
    visualize_gmm,
    visualize_traversal
)


class Demo_1d:
    """
    This demo allows you to test and visualize the disentanglement of artificial 1D data.

    An undirected graph is created from the result of multi-factor analysis (MFA).
    This graph is then traversed.

    Using the resulting sequence, these points can be interpolated with cubic splines.
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Learn Coordinates of 1d Manifold")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        setting_frame = ttk.Frame(self.root, padding=(3, 3, 12, 12))
        setting_frame.grid(column=0, row=0)
        plot_frame = ttk.Frame(self.root, padding=(3, 3, 12, 12))
        plot_frame.grid(column=1, row=0)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.ax = None
        self._init_settings(setting_frame, plot_frame)
        self._set_padding(setting_frame)
        self._init_plot(plot_frame)

        self.experiment = None
        self.graph_edges = None

        self.traversal_order = None

        self.spline = None
        self.spline_line = None
        self.spline_point = None
        self.t = 0.0


    def _init_settings(self, setting_frame, plot_frame):
        # data combo box
        ttk.Label(master=setting_frame, text="Data: ").grid(column=0, row=1, sticky=tk.W)
        self.data_var = tk.StringVar(value='circle')
        data_box = ttk.Combobox(master=setting_frame, width=9, textvariable=self.data_var)
        data_box['values'] = ('circle', 'curve_in_3d')
        data_box.state(["readonly"])
        data_box.grid(column=1, row=1)

        # number of data points
        ttk.Label(master=setting_frame, text="Number of data points: ").grid(
            column=0, row=2, sticky=tk.W)
        self.num_points = tk.IntVar(value=100)
        ttk.Entry(master=setting_frame, width=10,
                  textvariable=self.num_points).grid(column=1, row=2)

        # embed dim
        ttk.Label(master=setting_frame, text="Embed data into dimension: ").grid(
            column=0, row=3, sticky=tk.W)
        self.embed_dim = tk.IntVar()
        ttk.Entry(master=setting_frame, width=10,
                  textvariable=self.embed_dim).grid(column=1, row=3)

        # number of components
        ttk.Label(master=setting_frame, text="Number of Components: ").grid(
            column=0, row=4, sticky=tk.W)
        self.num_components = tk.IntVar(value=15)
        ttk.Entry(master=setting_frame, width=10,
                  textvariable=self.num_components).grid(column=1, row=4)

        # k nearest neighbors
        ttk.Label(master=setting_frame, text="k (neighbors for graph):").grid(
            column=0, row=5, sticky=tk.W)
        self.k_neighbors = tk.IntVar(value=2)
        ttk.Entry(master=setting_frame, width=10,
                  textvariable=self.k_neighbors).grid(column=1, row=5)

        # Train button (trains everything: mfa + graph + spline)
        ttk.Button(master=setting_frame, text="Train",
                   command=lambda: self._train(plot_frame)).grid(column=1, row=6)


        # radio button to select what gets drawn
        draw_setting_label = ttk.Label(master=setting_frame, text="--- Draw Setting ---")
        draw_setting_label.grid(column=0, row=7, sticky=tk.W)

        self.draw_mode = tk.StringVar(value="data")
        ttk.Radiobutton(
            master=setting_frame, text="Data", variable=self.draw_mode,
            value="data", command=lambda: self._update_plot(plot_frame)
        ).grid(column=0, row=8, sticky=tk.W)
        ttk.Radiobutton(
            master=setting_frame, text="MFA", variable=self.draw_mode,
            value="mfa", command=lambda: self._update_plot(plot_frame)
        ).grid(column=1, row=8, sticky=tk.W)
        ttk.Radiobutton(
            master=setting_frame, text="Graph", variable=self.draw_mode,
            value="graph", command=lambda: self._update_plot(plot_frame)
        ).grid(column=0, row=9, sticky=tk.W)
        ttk.Radiobutton(
            master=setting_frame, text="Spline", variable=self.draw_mode,
            value="spline", command=lambda: self._update_plot(plot_frame)
        ).grid(column=1, row=9, sticky=tk.W)

        # Spline Settings
        ttk.Label(master=setting_frame, text="t (spline interpolation)").grid(
            column=0, row=10, sticky=tk.W)
        self.t_slider = ttk.Scale(
            master=setting_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=lambda val: self._slider_change(float(val), plot_frame)
        )
        self.t_slider.grid(column=1, row=10, sticky=tk.EW)

        # TODO: sliders to change distance-metric

        # TODO: input-box for random-seed

        # TODO: find some interesting seeds, and make sure, that the outputs are deterministic



    def _init_plot(self, plot_frame, is_3d=False):
        # destroy existing plot
        for widget in plot_frame.winfo_children():
            widget.destroy()
        if self.ax:
            self.ax.cla()

        # create the new plot
        self.fig = Figure(figsize=(12, 12), dpi=100)
        if is_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        canvas.draw()

        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _set_padding(self, setting_frame):
        for child in setting_frame.winfo_children():
            child.grid_configure(padx=6, pady=8)

    def _train(self, plot_frame):
        # train mfa
        self.experiment = Experiment(
            data_type=self.data_var.get(),
            N=self.num_points.get(),
            C=self.num_components.get(),
            H=1,
            cov_type="mfa",
            shared=False,
            embed_dim=self.embed_dim.get(),
        )
        self.experiment.generate_data()
        self.experiment.train()
        print("Trained mfa")

        # graph
        graph = build_graph(self.experiment.model, n_tangents=1, k=self.k_neighbors.get())
        self.graph_edges = graph["edges"]
        print("Trained graph")

        # spline
        self.traversal_order = traverse_graph(graph["adjacency"])
        self.spline = build_spline_from_graph(self.experiment.model, graph)
        self._init_spline_plot(plot_frame)

        # visualize results
        self._update_plot(plot_frame)


    def _init_spline_plot(self, plot_frame):
        t_vals_full = np.linspace(0, 1, 200)
        curve_full = self.spline(t_vals_full)

        # init plot
        self._init_plot(self.root.children[list(self.root.children.keys())[1]])

        # Set axes limits so you actually see the spline
        margin = 0.1 * (curve_full.max(axis=0) - curve_full.min(axis=0))
        self.ax.set_xlim(curve_full[:, 0].min() - margin[0], curve_full[:, 0].max()+margin[0])
        self.ax.set_ylim(curve_full[:, 1].min() - margin[1], curve_full[:, 1].max()+margin[1])

        self.spline_line, = self.ax.plot([], [], c="blue", linewidth=2)
        self.spline_point = self.ax.scatter([], [], s=80, c="red", zorder=5)

        self._slider_change(0.0, plot_frame)


    def _slider_change(self, value, plot_frame):
        self.t = value

        if self.spline is None or self.draw_mode.get() != "spline":
            return

        t_vals = np.linspace(0, self.t, 200)
        curve = self.spline(t_vals)

        self.spline_line.set_data(curve[:, 0], curve[:, 1])

        p = self.spline(self.t)
        self.spline_point.set_offsets([p[0], p[1]])

        self.fig.canvas.draw_idle()


    def _update_plot(self, plot_frame):
        if self.experiment is None:
            return

        is_3d = self.data_var.get() in ["curve_in_3d"]
        self._init_plot(plot_frame, is_3d=is_3d)

        model = self.experiment.model
        proj = self.experiment.projection_matrix
        mode = self.draw_mode.get()

        if mode == "data":
            visualize_gmm(
                ax=self.ax, data=self.experiment.data, means=model.means,
                covariances=model.covariances, priors=model.prior,
                projection_matrix=proj, draw_points=True,
                visualisation_mode="none", draw_means=False
            )

        elif mode == "mfa":
            visualize_gmm(
                ax=self.ax, data=self.experiment.data, means=model.means,
                covariances=model.covariances, priors=model.prior,
                projection_matrix=proj, draw_points=True,
                visualisation_mode="line", draw_means=True
            )

        elif mode == "graph" and self.graph_edges is not None:
            means = model.means @ proj if proj is not None else model.means
            visualize_traversal(ax=self.ax, means=means, order=self.traversal_order)

        if mode == "spline" and self.spline is not None:
            # TODO: splines in 3D do not work yet
            self.ax.scatter(self.experiment.data[:, 0], self.experiment.data[:, 1], c='grey', alpha=0.2, s=5)
            self.spline_line, = self.ax.plot([], [], c="blue", linewidth=2, zorder=10)
            self.spline_point = self.ax.scatter([], [], s=80, c="red", zorder=11)
            self._slider_change(self.t, plot_frame)

        self.fig.canvas.draw()

    def main_loop(self):
        self.root.mainloop()


if __name__ == "__main__":
    demo = Demo_1d()
    demo.main_loop()

