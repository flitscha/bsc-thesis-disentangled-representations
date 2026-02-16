import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from experiment import Experiment
from visualization.visualize import visualize_gmm, visualize_gmm_higher_dimension

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


    def _init_settings(self, setting_frame, plot_frame):
        # data combo box
        ttk.Label(master=setting_frame, text="Data: ").grid(column=0, row=1, sticky=tk.W)
        self.data_var = tk.StringVar(value='circle')
        data_box = ttk.Combobox(master=setting_frame, width=9, textvariable=self.data_var)
        data_box['values'] = ('line', 'circle', 'swiss_roll')
        data_box.state(["readonly"])
        data_box.grid(column=1, row=1)

        # number of data points
        ttk.Label(master=setting_frame, text="Number of data points: ").grid(column=0, row=2, sticky=tk.W)
        self.num_points = tk.IntVar(value=100)
        ttk.Entry(master=setting_frame, width=10, textvariable=self.num_points).grid(column=1, row=2)

        # embed dim
        ttk.Label(master=setting_frame, text="Embed data into dimension: ").grid(column=0, row=3, sticky=tk.W)
        self.embed_dim = tk.IntVar()
        ttk.Entry(master=setting_frame, width=10, textvariable=self.embed_dim).grid(column=1, row=3)

        # Model
        ttk.Label(master=setting_frame, text="Model: ").grid(column=0, row=4, sticky=tk.W)
        self.cov_type = tk.StringVar(value='mfa')
        model_box = ttk.Combobox(master=setting_frame, width=9, textvariable=self.cov_type)
        model_box['values'] = ('isotropic', 'diagonal', 'mfa', 'full')
        model_box.state(["readonly"])
        model_box.grid(column=1, row=4)

        # Shared covariances
        self.shared_cov = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            master=setting_frame, text="Shared Covariances", onvalue=True, offvalue=False, variable=self.shared_cov
        ).grid(column=1, row=5, sticky=tk.W)

        # number of components
        ttk.Label(master=setting_frame, text="Number of Components: ").grid(column=0, row=6, sticky=tk.W)
        self.num_components = tk.IntVar(value=15)
        ttk.Entry(master=setting_frame, width=10, textvariable=self.num_components).grid(column=1, row=6)

        # Manifold dimension
        ttk.Label(master=setting_frame, text="Manifold Dimension: ").grid(column=0, row=7, sticky=tk.W)
        self.manifold_dimension = tk.IntVar(value=1)
        ttk.Entry(master=setting_frame, width=10, textvariable=self.manifold_dimension).grid(column=1, row=7)

        # Train button
        ttk.Button(master=setting_frame, text="Train", command=lambda: self._train(plot_frame)).grid(column=1, row=8)


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
        experiment = Experiment(
            data_type=self.data_var.get(),
            N=self.num_points.get(),
            C=self.num_components.get(),
            H=self.manifold_dimension.get(),
            cov_type=self.cov_type.get(),
            shared=self.shared_cov.get(),
            embed_dim=self.embed_dim.get(),
        )
        experiment.generate_data()
        experiment.train()

        # visualize results
        print("obj:", experiment.obj)
        
        # init plot
        is_3d = False
        if self.data_var.get() == "swiss_roll":
            is_3d = True
        self._init_plot(plot_frame, is_3d=is_3d)

        # plot results
        if self.embed_dim.get() == 0:
            visualize_gmm(self.ax, experiment.data, experiment.model.means, experiment.model.covariances, experiment.model.prior)
        else:
            visualize_gmm_higher_dimension(
                self.ax, experiment.data, experiment.model.means, experiment.model.covariances, experiment.model.prior, experiment.projection_matrix
            )
            


    def main_loop(self):
        self.root.mainloop()

