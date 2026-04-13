import tkinter as tk
from tkinter import ttk
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from experiment import Experiment

from manifold_graph import (
    extract_tangent_directions,
    compute_score_matrix,
    build_knn_graph,
    adjacency_to_edges,
)
from visualization.visualize import visualize_gmm, visualize_graph_on_mfa, visualize_traversal


from graph_traversal import traverse_graph
from manifold_interpolation import build_closed_spline


class Demo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MFA Demo")
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
        ttk.Label(master=setting_frame, text="Data: ").grid(
            column=0, row=1, sticky=tk.W)
        self.data_var = tk.StringVar(value='circle')
        data_box = ttk.Combobox(master=setting_frame,
                                width=9, textvariable=self.data_var)
        data_box['values'] = (
            'line', 'circle', 'swiss_roll', 'torus', 'curve_in_3d')
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

        # Model
        ttk.Label(master=setting_frame, text="Model: ").grid(
            column=0, row=4, sticky=tk.W)
        self.cov_type = tk.StringVar(value='mfa')
        model_box = ttk.Combobox(master=setting_frame,
                                 width=9, textvariable=self.cov_type)
        model_box['values'] = ('isotropic', 'diagonal', 'mfa', 'full')
        model_box.state(["readonly"])
        model_box.grid(column=1, row=4)

        # Shared covariances
        self.shared_cov = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            master=setting_frame, text="Shared Covariances", onvalue=True, offvalue=False, variable=self.shared_cov
        ).grid(column=1, row=5, sticky=tk.W)

        # number of components
        ttk.Label(master=setting_frame, text="Number of Components: ").grid(
            column=0, row=6, sticky=tk.W)
        self.num_components = tk.IntVar(value=15)
        ttk.Entry(master=setting_frame, width=10,
                  textvariable=self.num_components).grid(column=1, row=6)

        # Manifold dimension
        ttk.Label(master=setting_frame, text="Manifold Dimension: ").grid(
            column=0, row=7, sticky=tk.W)
        self.manifold_dimension = tk.IntVar(value=1)
        ttk.Entry(master=setting_frame, width=10,
                  textvariable=self.manifold_dimension).grid(column=1, row=7)

        # Train button
        ttk.Button(master=setting_frame, text="Train",
                   command=lambda: self._train(plot_frame)).grid(column=1, row=8)

        # draw points toggle
        self.draw_points = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            master=setting_frame, text="Draw Points", onvalue=True, offvalue=False,
            variable=self.draw_points, command=lambda: self._update_plot(
                plot_frame)
        ).grid(column=0, row=9, sticky=tk.W)

        # visualisation mode for the gaussian-components
        ttk.Label(master=setting_frame, text="Visualisation Mode:").grid(
            column=0, row=10, sticky=tk.W)
        self.visualisation_mode = tk.StringVar(value="ellipsoid")
        ttk.Radiobutton(
            master=setting_frame, text="Ellipsoid", variable=self.visualisation_mode,
            value="ellipsoid", command=lambda: self._update_plot(plot_frame)
        ).grid(column=0, row=11, sticky=tk.W)
        ttk.Radiobutton(
            master=setting_frame, text="Line", variable=self.visualisation_mode,
            value="line", command=lambda: self._update_plot(plot_frame)
        ).grid(column=0, row=12, sticky=tk.W)
        ttk.Radiobutton(
            master=setting_frame, text="Plane", variable=self.visualisation_mode,
            value="plane", command=lambda: self._update_plot(plot_frame)
        ).grid(column=1, row=11, sticky=tk.W)
        ttk.Radiobutton(
            master=setting_frame, text="None", variable=self.visualisation_mode,
            value="none", command=lambda: self._update_plot(plot_frame)
        ).grid(column=1, row=12, sticky=tk.W)

        # draw means toggle
        self.draw_means = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            master=setting_frame, text="Draw Means", onvalue=True, offvalue=False,
            variable=self.draw_means, command=lambda: self._update_plot(
                plot_frame)
        ).grid(column=1, row=9, sticky=tk.W)

        # TODO: setting for error range in embedded data?

        # -------------------------------------
        # Graph Settings
        self.graph_label = ttk.Label(
            master=setting_frame, text="--- Graph Settings ---")
        self.graph_label.grid(column=0, row=13, sticky=tk.W)

        # compute graph button
        self.compute_graph_button = ttk.Button(
            master=setting_frame,
            text="Compute Graph",
            command=self._compute_graph
        )
        self.compute_graph_button.grid(column=0, row=14, sticky=tk.W)

        # draw graph toggle
        self.draw_graph = tk.BooleanVar(value=True)
        self.draw_graph_checkbox = ttk.Checkbutton(
            master=setting_frame,
            text="Draw Graph",
            variable=self.draw_graph,
            command=lambda: self._update_plot(plot_frame)
        )
        self.draw_graph_checkbox.grid(column=1, row=14, sticky=tk.W)

        # k nearest neighbors
        ttk.Label(master=setting_frame, text="k (neighbors):").grid(
            column=0, row=15, sticky=tk.W)
        self.k_neighbors = tk.IntVar(value=2)
        ttk.Entry(master=setting_frame, width=10,
                  textvariable=self.k_neighbors).grid(column=1, row=15)

        self.traverse_button = ttk.Button(
            master=setting_frame,
            text="Traverse Graph",
            command=self._traverse_graph
        )
        self.traverse_button.grid(column=0, row=16, sticky=tk.W)

        # -------------------------------------
        # Spline Settings
        ttk.Label(master=setting_frame,
                  text="--- Spline ---").grid(column=0, row=17, sticky=tk.W)

        self.calc_spline_button = ttk.Button(
            master=setting_frame,
            text="Calculate Spline",
            command=self._calculate_splines
        )
        self.calc_spline_button.grid(column=0, row=18, sticky=tk.W)

        ttk.Label(master=setting_frame, text="t (interpolation parameter)").grid(
            column=0, row=19, sticky=tk.W)
        self.t_slider = ttk.Scale(
            master=setting_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=lambda val: self._slider_change(float(val))
        )
        self.t_slider.grid(column=1, row=19, sticky=tk.EW)

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
        self.experiment = Experiment(
            data_type=self.data_var.get(),
            N=self.num_points.get(),
            C=self.num_components.get(),
            H=self.manifold_dimension.get(),
            cov_type=self.cov_type.get(),
            shared=self.shared_cov.get(),
            embed_dim=self.embed_dim.get(),
        )
        self.experiment.generate_data()
        self.experiment.train()

        # visualize results
        print("obj:", self.experiment.obj)
        self._update_plot(plot_frame)

    def _compute_graph(self):
        if self.experiment is None:
            return

        means = self.experiment.model.means
        covariances = self.experiment.model.covariances

        tangents = extract_tangent_directions(covariances)
        score_matrix = compute_score_matrix(means, tangents)

        adjacency = build_knn_graph(score_matrix, k=self.k_neighbors.get())
        self.graph_edges = adjacency_to_edges(adjacency)

        print("Graph computed with k =", self.k_neighbors.get())

        self._update_plot(
            self.root.children[list(self.root.children.keys())[1]])

    def _traverse_graph(self):
        if self.experiment is None:
            return

        means = self.experiment.model.means
        covariances = self.experiment.model.covariances

        tangents = extract_tangent_directions(covariances)
        score_matrix = compute_score_matrix(means, tangents)

        adjacency = build_knn_graph(score_matrix, k=self.k_neighbors.get())

        self.traversal_order = traverse_graph(adjacency)

        print("Traversal length:", len(self.traversal_order))

        self._update_plot(
            self.root.children[list(self.root.children.keys())[1]]
        )

    def _calculate_splines(self):
        points_ordered = self.experiment.model.means[self.traversal_order]
        self.spline = build_closed_spline(points_ordered)

        t_vals_full = np.linspace(0, 1, 200)
        curve_full = self.spline(t_vals_full)

        # init plot
        self._init_plot(self.root.children[list(self.root.children.keys())[1]])

        # Set axes limits so you actually see the spline
        margin = 0.1 * (curve_full.max(axis=0) - curve_full.min(axis=0))
        self.ax.set_xlim(curve_full[:, 0].min() -
                         margin[0], curve_full[:, 0].max()+margin[0])
        self.ax.set_ylim(curve_full[:, 1].min() -
                         margin[1], curve_full[:, 1].max()+margin[1])

        self.spline_line, = self.ax.plot([], [], c="blue", linewidth=2)
        self.spline_point = self.ax.scatter([], [], s=80, c="red", zorder=5)

        self._slider_change(0.0)

    def _slider_change(self, value):
        self.t = value
        if self.spline is not None:
            t_vals = np.linspace(0, self.t, 200)
            curve = self.spline(t_vals)
            self.spline_line.set_data(curve[:, 0], curve[:, 1])

            p = self.spline(self.t)
            self.spline_point.set_offsets([p[0], p[1]])

            self.fig.canvas.draw_idle()

    def _update_plot(self, plot_frame):
        # return, if there is no data to plot
        if self.experiment is None:
            return

        # init plot
        is_3d = False
        if self.data_var.get() in ["swiss_roll", "torus", "curve_in_3d"]:
            is_3d = True
        self._init_plot(plot_frame, is_3d=is_3d)

        # plot results
        if self.draw_graph.get() and self.graph_edges is not None:
            proj = self.experiment.projection_matrix

            if proj is not None:
                means_proj = self.experiment.model.means @ proj
                covs_proj = [
                    proj.T @ cov @ proj
                    for cov in self.experiment.model.covariances
                ]
            else:
                means_proj = self.experiment.model.means
                covs_proj = self.experiment.model.covariances

            visualize_graph_on_mfa(
                ax=self.ax,
                means=means_proj,
                covariances=covs_proj,
                priors=self.experiment.model.prior,
                edges=self.graph_edges
            )
        else:
            visualize_gmm(
                ax=self.ax,
                data=self.experiment.data,
                means=self.experiment.model.means,
                covariances=self.experiment.model.covariances,
                priors=self.experiment.model.prior,
                projection_matrix=self.experiment.projection_matrix,
                draw_points=self.draw_points.get(),
                visualisation_mode=self.visualisation_mode.get(),
                draw_means=self.draw_means.get()
            )

        if self.traversal_order is not None:
            visualize_traversal(self.ax, means_proj, self.traversal_order)

        # Draw spline curve
        if self.spline is not None:
            import numpy as np
            t_vals = np.linspace(0, self.t, 200)
            curve = self.spline(t_vals)
            self.ax.plot(curve[:, 0], curve[:, 1], c="blue",
                         linewidth=2, label="Spline")

            # current point
            p = self.spline(self.t)
            self.ax.scatter(p[0], p[1], s=80, c="red",
                            zorder=5, label="Current point")

    def main_loop(self):
        self.root.mainloop()
